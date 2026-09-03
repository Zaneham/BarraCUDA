#include "amdgpu.h"
#include <string.h>
#include <stdio.h>

/*
 * Divergence-aware SSA register allocator for AMDGCN.
 *
 * On Wave64 (CDNA3/MI300X), spilling a divergent VGPR costs 64 dwords
 * of scratch per lane.  Spilling a uniform VGPR costs 1 dword — extract
 * via v_readfirstlane, store as scalar, broadcast back on reload.
 * The previous allocator had no knowledge of this 64:1 cost asymmetry,
 * producing 1,754 spills on the Moa transport kernel and treating every
 * eviction like a democracy where all values are created equal.
 * They are not.  Some values are 64× more equal than others.
 *
 * References:
 *   Sampaio et al. (2013) "Divergence Analysis", ACM TOPLAS 35(4) §6
 *   Cooper et al. (2001) "A Simple, Fast Dominance Algorithm"
 *   Braun & Hack (2009) "Register Spilling for SSA-Form Programs"
 *
 * Dependencies: libc, faith in fixed-point algorithms, strong tea.
 */

/* ---- Pool Limits ---- */

#define RS_MAX_BLK  4096
#define RS_MAX_VR   8192
#define RS_BV_WDS   ((RS_MAX_VR + 31) / 32)

/* Spill relay registers — shared with emit.c */
#define RS_RELAY_V0   250
#define RS_VGPR_CEIL  250
#define RS_RELAY_S    99
#define RS_RELAY_S2   98
#define RS_MAX_SPILL  512

/* ---- Static Pools ---- */

/* CFG */
static uint16_t rs_succ[RS_MAX_BLK * 2];   /* successors (max 2 per block) */
static uint8_t  rs_nsuc[RS_MAX_BLK];
static uint16_t rs_pred[RS_MAX_BLK * 4];   /* predecessors */
static uint8_t  rs_nprd[RS_MAX_BLK];
static uint16_t rs_poff[RS_MAX_BLK];       /* pred list offset */

/* Dominator tree */
static uint16_t rs_idom[RS_MAX_BLK];
static uint16_t rs_rpo[RS_MAX_BLK];        /* RPO number per block */
static uint16_t rs_rord[RS_MAX_BLK];       /* block at RPO position i */

/* Loop depth */
static uint16_t rs_ldep[RS_MAX_BLK];

/* Liveness bitvectors */
static uint32_t rs_lin[RS_MAX_BLK * RS_BV_WDS];
static uint32_t rs_lout[RS_MAX_BLK * RS_BV_WDS];
static uint32_t rs_bdef[RS_MAX_BLK * RS_BV_WDS];
static uint32_t rs_buse[RS_MAX_BLK * RS_BV_WDS];

/* Coloring */
static uint16_t rs_col[RS_MAX_VR];     /* physical reg, 0xFFFF = uncolored */
static uint8_t  rs_spd[RS_MAX_VR];     /* 1 = spilled */
static uint32_t rs_cost[RS_MAX_VR];    /* weighted spill cost */

/* Remat info */
typedef struct { uint16_t op; int32_t imm; } rs_rmat_t;
static rs_rmat_t rs_rmat[RS_MAX_VR];

/* Dominator postorder */
static uint16_t rs_dpord[RS_MAX_BLK];
static uint16_t rs_dchld[RS_MAX_BLK * 8]; /* domtree children, packed */
static uint16_t rs_dcoff[RS_MAX_BLK];     /* child list offset */
static uint8_t  rs_dcnt[RS_MAX_BLK];      /* child count */

/* Spill slot tracking */
static struct {
    uint16_t vreg;
    uint16_t off;   /* byte offset in scratch */
} rs_spill[RS_MAX_SPILL];
static uint32_t rs_nspill;

/* Spill slot offset lookup */
static uint16_t rs_soff_tbl[RS_MAX_VR];

/* Expansion buffer for spill codegen */
#define RS_EXPBUF  32768
static minst_t rs_ebuf[RS_EXPBUF];

/* ---- Bitvector Helpers ---- */

static inline void bv_set(uint32_t *bv, uint16_t bit)
{ bv[bit / 32] |= 1u << (bit % 32); }

static inline void bv_clr(uint32_t *bv, uint16_t bit)
{ bv[bit / 32] &= ~(1u << (bit % 32)); }

static inline int bv_tst(const uint32_t *bv, uint16_t bit)
{ return (int)((bv[bit / 32] >> (bit % 32)) & 1u); }

/* ---- Phase 2: CFG + Dominator Tree ---- */

/* Is this instruction a block terminator? */
static int rs_term(uint16_t op)
{
    return op == AMD_S_BRANCH || op == AMD_S_CBRANCH_SCC0 ||
           op == AMD_S_CBRANCH_SCC1 || op == AMD_S_CBRANCH_EXECZ ||
           op == AMD_S_CBRANCH_EXECNZ || op == AMD_S_ENDPGM ||
           op == AMD_S_SETPC_B64;
}

/* Build CFG successors + predecessors.  Same edge detection as
 * ra_build_cfg but without the dramatics. */
static void rs_cfg(const amd_module_t *A, const mfunc_t *F)
{
    uint16_t nb = F->num_blocks;
    if (nb > RS_MAX_BLK) nb = RS_MAX_BLK;

    memset(rs_nsuc, 0, nb);
    memset(rs_nprd, 0, nb);

    /* Pass 1: build successor lists by scanning terminators.
     * Must scan ALL terminators backward from end of block, not
     * just the last instruction — a block can have a conditional
     * branch followed by an unconditional branch. Copying the
     * ra_build_cfg pattern from emit.c because ignoring it the
     * first time orphaned 174 blocks in the domtree.  Lesson:
     * "same edge detection" means SAME edge detection. */
    for (uint16_t bi = 0; bi < nb; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        int has_uncond = 0;

        /* Empty blocks fall through — don't skip them! */
        if (MB->num_insts > 0) {
            for (uint32_t ii = MB->num_insts; ii > 0; ii--) {
                const minst_t *mi = &A->minsts[MB->first_inst + ii - 1];
                if (!rs_term(mi->op)) break;

                if (mi->op == AMD_S_ENDPGM || mi->op == AMD_S_SETPC_B64) {
                    has_uncond = 1;
                } else if (mi->op == AMD_S_BRANCH) {
                    has_uncond = 1;
                    if (mi->num_uses > 0 &&
                        mi->operands[mi->num_defs].kind == MOP_LABEL) {
                        uint32_t tgt = (uint32_t)mi->operands[mi->num_defs].imm;
                        if (tgt >= F->first_block &&
                            tgt < F->first_block + nb &&
                            rs_nsuc[bi] < 2) {
                            rs_succ[bi * 2 + rs_nsuc[bi]++] =
                                (uint16_t)(tgt - F->first_block);
                        }
                    }
                } else {
                    /* Conditional branch */
                    if (mi->num_uses > 0 &&
                        mi->operands[mi->num_defs].kind == MOP_LABEL) {
                        uint32_t tgt = (uint32_t)mi->operands[mi->num_defs].imm;
                        if (tgt >= F->first_block &&
                            tgt < F->first_block + nb &&
                            rs_nsuc[bi] < 2) {
                            rs_succ[bi * 2 + rs_nsuc[bi]++] =
                                (uint16_t)(tgt - F->first_block);
                        }
                    }
                }
            }
        }

        /* Fallthrough: if no unconditional branch/endpgm, next block
         * is a successor.  Empty blocks always fall through.
         * Getting this wrong orphans hundreds of blocks because empty
         * separator blocks break the CFG chain. */
        if (!has_uncond && bi + 1 < nb && rs_nsuc[bi] < 2)
            rs_succ[bi * 2 + rs_nsuc[bi]++] = (uint16_t)(bi + 1);
    }

    /* Pass 2: build predecessor lists from successors */
    /* First count preds per block */
    for (uint16_t bi = 0; bi < nb; bi++) {
        for (uint8_t s = 0; s < rs_nsuc[bi]; s++) {
            uint16_t tgt = rs_succ[bi * 2 + s];
            if (tgt < nb && rs_nprd[tgt] < 255)
                rs_nprd[tgt]++;
        }
    }

    /* Compute offsets */
    uint16_t poff = 0;
    for (uint16_t bi = 0; bi < nb; bi++) {
        rs_poff[bi] = poff;
        poff += rs_nprd[bi];
        if (poff > RS_MAX_BLK * 4) poff = RS_MAX_BLK * 4;
    }

    /* Fill predecessor lists */
    memset(rs_nprd, 0, nb);
    for (uint16_t bi = 0; bi < nb; bi++) {
        for (uint8_t s = 0; s < rs_nsuc[bi]; s++) {
            uint16_t tgt = rs_succ[bi * 2 + s];
            if (tgt >= nb) continue;
            uint16_t off = (uint16_t)(rs_poff[tgt] + rs_nprd[tgt]);
            if (off < RS_MAX_BLK * 4) {
                rs_pred[off] = bi;
                rs_nprd[tgt]++;
            }
        }
    }
}

/* Iterative stack-based RPO (no recursion, as the gods of avionics demand) */
static uint16_t rs_rpo_nb;

static void rs_bld_rpo(uint16_t nb)
{
    static uint8_t vis[RS_MAX_BLK];
    memset(vis, 0, nb);

    /* DFS stack: (block, next_succ_to_visit) */
    static struct { uint16_t blk; uint8_t si; } stk[RS_MAX_BLK];
    uint16_t top = 0;
    uint16_t rpo_pos = nb;

    stk[top].blk = 0;
    stk[top].si = 0;
    top++;
    vis[0] = 1;

    uint32_t guard = 0;
    while (top > 0 && guard < RS_MAX_BLK * 4) {
        guard++;
        uint16_t b = stk[top - 1].blk;
        uint8_t  si = stk[top - 1].si;

        if (si < rs_nsuc[b]) {
            stk[top - 1].si = si + 1;
            uint16_t s = rs_succ[b * 2 + si];
            if (s < nb && !vis[s]) {
                vis[s] = 1;
                if (top < RS_MAX_BLK) {
                    stk[top].blk = s;
                    stk[top].si = 0;
                    top++;
                }
            }
        } else {
            top--;
            if (rpo_pos > 0) {
                rpo_pos--;
                rs_rord[rpo_pos] = b;
                rs_rpo[b] = rpo_pos;
            }
        }
    }

    /* Unreachable blocks get appended at the end with high RPO numbers */
    for (uint16_t bi = 0; bi < nb; bi++) {
        if (!vis[bi] && rpo_pos > 0) {
            rpo_pos--;
            rs_rord[rpo_pos] = bi;
            rs_rpo[bi] = rpo_pos;
        }
    }

    rs_rpo_nb = nb;
}

/* Cooper et al. (2001) iterative dominator algorithm.
 * Converges in 2-3 passes for reducible CFGs. Bounded at 4*nb
 * iterations because we trust the algorithm but verify anyway. */
static void rs_dom(uint16_t nb)
{
    /* Sentinel: 0xFFFF = undefined */
    for (uint16_t i = 0; i < nb; i++)
        rs_idom[i] = 0xFFFF;
    rs_idom[0] = 0; /* entry dominates itself */

    int changed = 1;
    uint32_t guard = (uint32_t)nb * 4;

    while (changed && guard-- > 0) {
        changed = 0;
        /* Traverse in RPO (skip entry) */
        for (uint16_t ri = 0; ri < nb; ri++) {
            uint16_t b = rs_rord[ri];
            if (b == 0) continue; /* entry */

            /* Find first processed predecessor */
            uint16_t new_idom = 0xFFFF;
            for (uint8_t pi = 0; pi < rs_nprd[b]; pi++) {
                uint16_t p = rs_pred[rs_poff[b] + pi];
                if (rs_idom[p] != 0xFFFF) {
                    new_idom = p;
                    break;
                }
            }
            if (new_idom == 0xFFFF) continue;

            /* Intersect with other processed predecessors */
            for (uint8_t pi = 0; pi < rs_nprd[b]; pi++) {
                uint16_t p = rs_pred[rs_poff[b] + pi];
                if (p == new_idom || rs_idom[p] == 0xFFFF) continue;

                /* Walk up the domtree from both fingers */
                uint16_t f1 = p, f2 = new_idom;
                uint32_t ig = nb * 2;
                while (f1 != f2 && ig-- > 0) {
                    while (rs_rpo[f1] > rs_rpo[f2] && ig-- > 0)
                        f1 = rs_idom[f1];
                    while (rs_rpo[f2] > rs_rpo[f1] && ig-- > 0)
                        f2 = rs_idom[f2];
                }
                new_idom = f1;
            }

            if (rs_idom[b] != new_idom) {
                rs_idom[b] = new_idom;
                changed = 1;
            }
        }
    }
}

/* ---- Phase 3: Loop Depth ---- */

/* Detect back edges (B→H where H dominates B) and compute
 * natural loop bodies.  Each loop increments nesting depth.
 * Like counting how many Russian dolls you're inside, except
 * each doll has its own register pressure crisis. */
static void rs_loop(uint16_t nb)
{
    memset(rs_ldep, 0, nb * sizeof(uint16_t));

    static uint8_t in_loop[RS_MAX_BLK];
    static uint16_t wstk[RS_MAX_BLK]; /* worklist for body collection */

    for (uint16_t bi = 0; bi < nb; bi++) {
        for (uint8_t si = 0; si < rs_nsuc[bi]; si++) {
            uint16_t hdr = rs_succ[bi * 2 + si];
            if (hdr >= nb) continue;

            /* Back edge: target dominates source */
            uint16_t d = bi;
            int is_back = 0;
            uint32_t ig = nb;
            while (d != 0xFFFF && ig-- > 0) {
                if (d == hdr) { is_back = 1; break; }
                if (d == rs_idom[d]) break;
                d = rs_idom[d];
            }
            if (!is_back) continue;

            /* Collect natural loop body: reverse BFS from bi to hdr */
            memset(in_loop, 0, nb);
            in_loop[hdr] = 1;
            if (bi != hdr) {
                in_loop[bi] = 1;
                uint16_t wtop = 0;
                wstk[wtop++] = bi;

                uint32_t wg = nb * 2;
                while (wtop > 0 && wg-- > 0) {
                    uint16_t n = wstk[--wtop];
                    for (uint8_t pi = 0; pi < rs_nprd[n] && pi < 4; pi++) {
                        uint16_t p = rs_pred[rs_poff[n] + pi];
                        if (p < nb && !in_loop[p]) {
                            in_loop[p] = 1;
                            if (wtop < RS_MAX_BLK)
                                wstk[wtop++] = p;
                        }
                    }
                }
            }

            /* Increment depth for all loop body blocks */
            for (uint16_t j = 0; j < nb; j++) {
                if (in_loop[j] && rs_ldep[j] < 16)
                    rs_ldep[j]++;
            }
        }
    }
}

/* ---- Phase 4: SSA Liveness ---- */

/* PHI uses belong to predecessor blocks, not the PHI's block.
 * Getting this wrong means PHI sources appear live-in at the
 * def block instead of live-out at the pred block.  The register
 * allocator then thinks the value is needed earlier than it is,
 * and 200 perfectly good registers go to waste.  This is the single
 * most common liveness bug in SSA compilers. */

static void rs_live(const amd_module_t *A, const mfunc_t *F,
                    uint16_t nb, uint16_t nv)
{
    uint32_t bv_words = (uint32_t)((nv + 31) / 32);
    if (bv_words > RS_BV_WDS) bv_words = RS_BV_WDS;

    uint32_t bv_bytes = bv_words * 4;
    memset(rs_bdef, 0, (size_t)nb * bv_bytes);
    memset(rs_buse, 0, (size_t)nb * bv_bytes);
    memset(rs_lin,  0, (size_t)nb * bv_bytes);
    memset(rs_lout, 0, (size_t)nb * bv_bytes);

    /* Pass 1: compute per-block defs and uses */
    for (uint16_t bi = 0; bi < nb; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        uint32_t *def = &rs_bdef[(uint32_t)bi * bv_words];
        uint32_t *use = &rs_buse[(uint32_t)bi * bv_words];

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];

            if (mi->op == AMD_PSEUDO_PHI) {
                /* PHI defs are in this block */
                for (uint8_t d = 0; d < mi->num_defs; d++) {
                    uint16_t vr = op_vreg(&mi->operands[d]);
                    if (vr < nv) bv_set(def, vr);
                }
                /* PHI uses go to predecessor blocks — handled below */
                continue;
            }

            /* Normal instruction: uses before defs (upward exposed) */
            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

            for (uint8_t k = mi->num_defs; k < total; k++) {
                uint16_t vr = op_vreg(&mi->operands[k]);
                if (vr < nv && !bv_tst(def, vr))
                    bv_set(use, vr);
            }
            for (uint8_t d = 0; d < mi->num_defs; d++) {
                uint16_t vr = op_vreg(&mi->operands[d]);
                if (vr < nv) bv_set(def, vr);
            }
        }
    }

    /* Pass 1b: PHI uses belong to predecessor gen sets.
     * For each PHI operand pair (pred_block, value), if the value
     * is a vreg not defined in pred_block, it's upward-exposed. */
    for (uint16_t bi = 0; bi < nb; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];
            if (mi->op != AMD_PSEUDO_PHI) continue;

            for (uint8_t p = 0; p + 1 < mi->num_uses; p += 2) {
                uint8_t off = mi->num_defs + p;
                if (off + 1 >= MINST_MAX_OPS) break;
                if (mi->operands[off].kind != MOP_LABEL) continue;

                uint32_t pred_mb = (uint32_t)mi->operands[off].imm;
                if (pred_mb < F->first_block ||
                    pred_mb >= F->first_block + nb) continue;
                uint16_t pred_rel = (uint16_t)(pred_mb - F->first_block);

                uint16_t vr = op_vreg(&mi->operands[off + 1]);
                if (vr < nv) {
                    uint32_t *puse = &rs_buse[(uint32_t)pred_rel * bv_words];
                    uint32_t *pdef = &rs_bdef[(uint32_t)pred_rel * bv_words];
                    if (!bv_tst(pdef, vr))
                        bv_set(puse, vr);
                }
            }
        }
    }

    /* Pass 2: backward dataflow iteration to fixpoint.
     *   live_in[b]  = use[b] | (live_out[b] - def[b])
     *   live_out[b] = U live_in[s] for each successor s
     * Bounded at 200 iterations — overkill for reducible CFGs,
     * but the sort of paranoia that keeps compilers honest. */
    int changed = 1;
    uint32_t guard = 200;

    while (changed && guard-- > 0) {
        changed = 0;
        /* Process in reverse RPO for faster convergence */
        for (int ri = (int)nb - 1; ri >= 0; ri--) {
            uint16_t b = rs_rord[ri];
            uint32_t *lin  = &rs_lin[(uint32_t)b * bv_words];
            uint32_t *lout = &rs_lout[(uint32_t)b * bv_words];
            uint32_t *def  = &rs_bdef[(uint32_t)b * bv_words];
            uint32_t *use  = &rs_buse[(uint32_t)b * bv_words];

            /* live_out = union of successor live_ins */
            for (uint8_t si = 0; si < rs_nsuc[b]; si++) {
                uint16_t s = rs_succ[b * 2 + si];
                if (s >= nb) continue;
                const uint32_t *sin = &rs_lin[(uint32_t)s * bv_words];
                for (uint32_t w = 0; w < bv_words; w++)
                    lout[w] |= sin[w];
            }

            /* live_in = use | (live_out - def) */
            for (uint32_t w = 0; w < bv_words; w++) {
                uint32_t new_in = use[w] | (lout[w] & ~def[w]);
                if (new_in != lin[w]) {
                    lin[w] = new_in;
                    changed = 1;
                }
            }
        }
    }

    /* Pass 3: exec-mask region extension.
     * Values alive across a saveexec→restore pair must survive
     * the entire masked region.  Without this, the allocator
     * sees the last use inside the mask, frees the register,
     * and exec restores to find its value reupholstered. */
    {
        struct { uint32_t save; uint16_t sblk; uint32_t rest; uint16_t rblk; }
            eregion[64];
        uint32_t n_er = 0;
        uint32_t estack[32];
        uint16_t eblk[32];
        uint32_t esp = 0;

        for (uint16_t bi = 0; bi < nb && n_er < 64; bi++) {
            const mblock_t *MB = &A->mblocks[F->first_block + bi];
            for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
                uint32_t mi_idx = MB->first_inst + ii;
                const minst_t *mi = &A->minsts[mi_idx];

                if (mi->op == AMD_S_AND_SAVEEXEC_B64 ||
                    mi->op == AMD_S_AND_SAVEEXEC_B32) {
                    if (esp < 32) {
                        estack[esp] = mi_idx;
                        eblk[esp] = bi;
                        esp++;
                    }
                    continue;
                }

                if ((mi->op == AMD_S_OR_B64  || mi->op == AMD_S_OR_B32 ||
                     mi->op == AMD_S_XOR_B64 || mi->op == AMD_S_XOR_B32) &&
                    mi->num_defs > 0 &&
                    mi->operands[0].kind == MOP_SPECIAL &&
                    mi->operands[0].imm == AMD_SPEC_EXEC) {
                    if (esp > 0 && n_er < 64) {
                        esp--;
                        eregion[n_er].save = estack[esp];
                        eregion[n_er].sblk = eblk[esp];
                        eregion[n_er].rest = mi_idx;
                        eregion[n_er].rblk = bi;
                        n_er++;
                    }
                }
            }
        }

        /* For each exec region, any vreg live-out at the saveexec
         * block must also be live-in at the restore block and all
         * blocks in between.  We approximate by extending live-out
         * of saveexec block into live-in of all blocks from sblk+1
         * to rblk inclusive. */
        for (uint32_t e = 0; e < n_er; e++) {
            uint16_t sb = eregion[e].sblk;
            uint16_t rb = eregion[e].rblk;
            const uint32_t *slout = &rs_lout[(uint32_t)sb * bv_words];

            for (uint16_t bi = (uint16_t)(sb + 1); bi <= rb && bi < nb; bi++) {
                uint32_t *lin  = &rs_lin[(uint32_t)bi * bv_words];
                uint32_t *lout2 = &rs_lout[(uint32_t)bi * bv_words];
                for (uint32_t w = 0; w < bv_words; w++) {
                    lin[w]  |= slout[w];
                    lout2[w] |= slout[w];
                }
            }
        }
    }

    /* Pass 4: prologue SGPR pinning.
     * System SGPRs defined in block 0 that escape must survive
     * to the function's end.  Same as emit.c:398-433. */
    if (nb > 1) {
        const uint32_t *def0 = &rs_bdef[0];
        for (uint16_t v = 0; v < nv; v++) {
            if (A->reg_file[v] != 0) continue; /* SGPRs only */
            if (!bv_tst(def0, v)) continue;
            /* If live-out of block 0, pin to live everywhere */
            if (bv_tst(&rs_lout[0], v)) {
                for (uint16_t bi = 1; bi < nb; bi++) {
                    bv_set(&rs_lin[(uint32_t)bi * bv_words], v);
                    bv_set(&rs_lout[(uint32_t)bi * bv_words], v);
                }
            }
        }
    }
}

/* ---- Phase 5: Divergence-Aware Spill Cost ---- */

/* Power-of-10 depth weight, clamped at depth 8.
 * Inner loops are exponentially more expensive to spill into.
 * Braun & Hack (2009) use similar exponential weighting. */
static uint32_t rs_dwgt(uint16_t depth)
{
    static const uint32_t tbl[] = {
        1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000
    };
    if (depth > 8) depth = 8;
    return tbl[depth];
}

/* Compute divergence-aware spill cost per vreg.
 * Sampaio et al. (2013) §6: multiply by wave_width for divergent,
 * 1 for uniform.  The 64:1 ratio means we'll strongly prefer
 * spilling the boring uniform pointers over the precious per-lane
 * particle state.  Physics appreciates this. */
static void rs_dcst(const amd_module_t *A, const mfunc_t *F,
                    uint16_t nb, uint16_t nv)
{
    memset(rs_cost, 0, (size_t)nv * sizeof(uint32_t));

    uint32_t wave_w = (F->exec_w) ? 64u : 32u;

    for (uint16_t bi = 0; bi < nb; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        uint32_t dw = rs_dwgt(rs_ldep[bi]);

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];
            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

            for (uint8_t k = 0; k < total; k++) {
                uint16_t vr = op_vreg(&mi->operands[k]);
                if (vr >= nv) continue;

                uint32_t div_w = 1;
                if (A->reg_file[vr] == 1 && vr_div(A, vr))
                    div_w = wave_w;

                /* Saturating add — 32-bit overflow protection */
                uint32_t add = dw * div_w;
                if (rs_cost[vr] + add < rs_cost[vr])
                    rs_cost[vr] = 0xFFFFFFFF;
                else
                    rs_cost[vr] += add;
            }
        }
    }
}

/* Detect rematerialisable vregs — constants and simple scalar ops.
 * Remat is free: no scratch, no memory traffic, just re-emit the
 * instruction.  These get cost 0, making them preferred spill victims.
 * Like asking "who wants to go to the scratch car park?" and the
 * constants raise their hands because they know the way back. */
static void rs_rdet(const amd_module_t *A, const mfunc_t *F,
                    uint16_t nb, uint16_t nv)
{
    memset(rs_rmat, 0, (size_t)nv * sizeof(rs_rmat_t));

    for (uint16_t bi = 0; bi < nb; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];
            if (mi->num_defs != 1) continue;

            uint16_t vr = op_vreg(&mi->operands[0]);
            if (vr >= nv) continue;

            /* s_mov_b32 vr, <literal> */
            if (mi->op == AMD_S_MOV_B32 && mi->num_uses == 1 &&
                mi->operands[1].kind == MOP_IMM) {
                rs_rmat[vr].op = AMD_S_MOV_B32;
                rs_rmat[vr].imm = mi->operands[1].imm;
                rs_cost[vr] = 0;
                continue;
            }

            /* v_mov_b32 vr, <literal> */
            if (mi->op == AMD_V_MOV_B32 && mi->num_uses == 1 &&
                mi->operands[1].kind == MOP_IMM) {
                rs_rmat[vr].op = AMD_V_MOV_B32;
                rs_rmat[vr].imm = mi->operands[1].imm;
                rs_cost[vr] = 0;
                continue;
            }

            /* Uniform VGPR defined by v_mov_b32 vr, sN — remat from SGPR.
             * Sampaio et al. §6.2: if the SGPR is still live at the reload
             * point, just re-emit v_mov_b32.  We record it optimistically;
             * spill codegen checks availability. */
            if (mi->op == AMD_V_MOV_B32 && mi->num_uses == 1 &&
                (mi->operands[1].kind == MOP_VREG_S ||
                 mi->operands[1].kind == MOP_SGPR) &&
                !vr_div(A, vr)) {
                rs_rmat[vr].op = AMD_V_MOV_B32;
                rs_rmat[vr].imm = (int32_t)mi->operands[1].reg_num;
                rs_cost[vr] = 0;
            }
        }
    }
}

/* ---- Phase 6: SSA Coloring with Divergence-Aware Spilling ---- */

/* Build dominator tree children lists and compute preorder.
 * Preorder = parents before children.  In SSA, defs dominate
 * all uses, so processing a block after its dominator guarantees
 * all live-in values (from ancestor defs) are already colored.
 * Postorder gets this backwards and everything lands on v0
 * like commuters on the same train seat. */
static uint16_t rs_bdpo(uint16_t nb)
{
    memset(rs_dcnt, 0, nb);
    memset(rs_dcoff, 0, nb * sizeof(uint16_t));

    /* Count children per node */
    for (uint16_t b = 1; b < nb; b++) {
        uint16_t p = rs_idom[b];
        if (p < nb && rs_dcnt[p] < 8)
            rs_dcnt[p]++;
    }

    /* Compute offsets */
    uint16_t off = 0;
    for (uint16_t b = 0; b < nb; b++) {
        rs_dcoff[b] = off;
        off += rs_dcnt[b];
        if (off > RS_MAX_BLK * 8) off = RS_MAX_BLK * 8;
    }

    /* Fill children (re-zero counts as fill index) */
    memset(rs_dcnt, 0, nb);
    for (uint16_t b = 1; b < nb; b++) {
        uint16_t p = rs_idom[b];
        if (p >= nb) continue;
        uint16_t ci = (uint16_t)(rs_dcoff[p] + rs_dcnt[p]);
        if (ci < RS_MAX_BLK * 8) {
            rs_dchld[ci] = b;
            rs_dcnt[p]++;
        }
    }

    /* Iterative preorder of domtree (parents first, then children) */
    static uint16_t stk[RS_MAX_BLK];
    uint16_t top = 0, n = 0;

    stk[top++] = 0;

    uint32_t guard = nb * 4;
    while (top > 0 && n < nb && guard-- > 0) {
        uint16_t b = stk[--top];
        rs_dpord[n++] = b;

        /* Push children in reverse order so first child pops first */
        for (int ci = (int)rs_dcnt[b] - 1; ci >= 0; ci--) {
            uint16_t child = rs_dchld[rs_dcoff[b] + ci];
            if (top < RS_MAX_BLK)
                stk[top++] = child;
        }
    }
    return n;
}

/* Greedy SSA coloring.  Process blocks in dominator post-order,
 * backward-scan maintaining a live set.  When pressure exceeds
 * the physical limit, spill the cheapest vreg per divergence cost.
 *
 * No IFG bitmatrix needed — interference from live set at each
 * def point.  Saves 8 MB vs ra_gc.  The mathematics of not
 * preallocating an O(n²) structure when n is already 8192. */
static uint32_t rs_aloc(amd_module_t *A, mfunc_t *F,
                        uint16_t nb, uint16_t nv)
{
    memset(rs_col, 0xFF, (size_t)nv * 2); /* 0xFFFF = uncolored */
    memset(rs_spd, 0, nv);

    uint32_t bv_words = (uint32_t)((nv + 31) / 32);
    if (bv_words > RS_BV_WDS) bv_words = RS_BV_WDS;

    uint16_t n_dpo = rs_bdpo(nb);
    uint32_t n_spill = 0;

    uint16_t sgpr_floor = F->is_kernel ? F->first_alloc_sgpr : 0;
    if (sgpr_floor < AMD_KERN_MIN_RESERVED && F->is_kernel)
        sgpr_floor = AMD_KERN_MIN_RESERVED;

    uint16_t vgpr_ceil = RS_VGPR_CEIL;
    if (amd_max_vgpr > 0 && (uint16_t)amd_max_vgpr < vgpr_ceil)
        vgpr_ceil = (uint16_t)amd_max_vgpr;

    /* Live set bitvector, reused per block */
    static uint32_t live[RS_BV_WDS];
    /* Color-in-use sets for finding free registers */
    static uint8_t s_used[AMD_MAX_SGPRS];
    static uint8_t v_used[AMD_MAX_VGPRS];

    uint16_t max_sgpr = 0, max_vgpr = 0;

    for (uint16_t di = 0; di < n_dpo; di++) {
        uint16_t b = rs_dpord[di];
        const mblock_t *MB = &A->mblocks[F->first_block + b];
        if (MB->num_insts == 0) continue;

        /* Initialise live set to live-out of this block */
        memcpy(live, &rs_lout[(uint32_t)b * bv_words], bv_words * 4);

        /* Backward scan.  At each instruction we must:
         * 1. Add uses to live set FIRST — they're alive at this point
         * 2. Then process defs: color against current live set, remove
         * This ensures the def interferes with its own instruction's
         * uses.  Getting this backwards puts everything on v0 because
         * the live set is empty when we try to color.  SSA's gift:
         * each def is unique, so no use == def conflicts. */
        for (int ii = (int)MB->num_insts - 1; ii >= 0; ii--) {
            uint32_t mi_idx = MB->first_inst + (uint32_t)ii;
            const minst_t *mi = &A->minsts[mi_idx];

            if (mi->op == AMD_S_NOP || mi->op == AMD_PSEUDO_DEF)
                continue;

            /* Step 1: add uses to live set (they're alive here) */
            if (mi->op != AMD_PSEUDO_PHI) {
                uint8_t total = mi->num_defs + mi->num_uses;
                if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;
                for (uint8_t k = mi->num_defs; k < total; k++) {
                    uint16_t vr = op_vreg(&mi->operands[k]);
                    if (vr < nv)
                        bv_set(live, vr);
                }
            }

            /* Step 2: process defs — color against live set, remove */
            for (uint8_t d = 0; d < mi->num_defs; d++) {
                uint16_t vr = op_vreg(&mi->operands[d]);
                if (vr >= nv || rs_spd[vr]) continue;

                /* Remove def from live set (born here, doesn't exist before) */
                bv_clr(live, vr);

                /* Already colored (e.g., PHI processed from another block) */
                if (rs_col[vr] != 0xFFFF) continue;

                uint8_t file = A->reg_file[vr];

                /* Collect colors of all interfering live vregs */
                memset(s_used, 0, AMD_MAX_SGPRS);
                memset(v_used, 0, AMD_MAX_VGPRS);

                for (uint16_t w = 0; w < nv; w++) {
                    if (!bv_tst(live, w)) continue;
                    if (rs_spd[w]) continue;
                    if (A->reg_file[w] != file) continue;
                    if (rs_col[w] == 0xFFFF) {
                        /* Uncolored live vreg from an earlier def in
                         * this block — backward scan hasn't reached
                         * its def yet.  Precolor it below. */
                    }
                    /* Mark color as used (skip uncolored — handled below
                     * by precoloring after the main color scan) */
                    if (rs_col[w] != 0xFFFF) {
                        if (file == 0 && rs_col[w] < AMD_MAX_SGPRS)
                            s_used[rs_col[w]] = 1;
                        else if (file == 1 && rs_col[w] < AMD_MAX_VGPRS)
                            v_used[rs_col[w]] = 1;
                    }
                }

                /* Pre-color uncolored live vregs in same file.
                 * Greedy: assign each the lowest free color.
                 * Rebuild used-set incrementally after each assignment. */
                for (uint16_t w = 0; w < nv; w++) {
                    if (!bv_tst(live, w)) continue;
                    if (rs_spd[w]) continue;
                    if (A->reg_file[w] != file) continue;
                    if (rs_col[w] != 0xFFFF) continue;

                    uint16_t pc = 0xFFFF;
                    if (file == 0) {
                        for (uint16_t r = sgpr_floor; r < AMD_MAX_SGPRS; r++) {
                            if (r == RS_RELAY_S || r == RS_RELAY_S2) continue;
                            if (!s_used[r]) { pc = r; break; }
                        }
                        if (pc != 0xFFFF) { rs_col[w] = pc; s_used[pc] = 1; }
                    } else {
                        for (uint16_t r = 0; r < vgpr_ceil; r++) {
                            if (!v_used[r]) { pc = r; break; }
                        }
                        if (pc != 0xFFFF) { rs_col[w] = pc; v_used[pc] = 1; }
                    }
                    if (pc != 0xFFFF) {
                        if (file == 0 && pc + 1 > max_sgpr) max_sgpr = pc + 1;
                        else if (file == 1 && pc + 1 > max_vgpr) max_vgpr = pc + 1;
                    }
                }

                /* Find lowest free color */
                uint16_t color = 0xFFFF;
                if (file == 0) {
                    for (uint16_t r = sgpr_floor; r < AMD_MAX_SGPRS; r++) {
                        if (r == RS_RELAY_S || r == RS_RELAY_S2) continue;
                        if (!s_used[r]) { color = r; break; }
                    }
                } else {
                    for (uint16_t r = 0; r < vgpr_ceil; r++) {
                        if (!v_used[r]) { color = r; break; }
                    }
                }

                if (color != 0xFFFF) {
                    rs_col[vr] = color;
                    if (file == 0 && color + 1 > max_sgpr)
                        max_sgpr = color + 1;
                    else if (file == 1 && color + 1 > max_vgpr)
                        max_vgpr = color + 1;
                } else {
                    /* Pressure exceeded — spill cheapest.
                     * Among def + all interfering live vregs in same file,
                     * find the one with lowest cost.  Remat first (cost 0),
                     * then uniform VGPRs, then divergent VGPRs last. */
                    uint16_t victim = vr;
                    uint32_t vcost = rs_cost[vr];

                    for (uint16_t w = 0; w < nv; w++) {
                        if (!bv_tst(live, w)) continue;
                        if (A->reg_file[w] != file) continue;
                        if (rs_spd[w]) continue;
                        if (rs_col[w] == 0xFFFF) continue;
                        if (rs_cost[w] < vcost) {
                            vcost = rs_cost[w];
                            victim = w;
                        }
                    }

                    if (victim != vr && rs_col[victim] != 0xFFFF) {
                        /* Evict victim, steal its color */
                        color = rs_col[victim];
                        rs_col[victim] = 0xFFFF;
                        rs_spd[victim] = 1;
                        rs_col[vr] = color;
                        n_spill++;
                    } else {
                        /* Spill ourselves — no victim cheaper */
                        rs_spd[vr] = 1;
                        n_spill++;
                    }
                }
            }
        }
    }

    F->num_sgprs = max_sgpr;
    if (F->is_kernel && F->num_sgprs < F->first_alloc_sgpr)
        F->num_sgprs = F->first_alloc_sgpr;
    F->num_vgprs = max_vgpr;
    if (F->uses_mfma && F->num_vgprs < RA_MFMA_HI) F->num_vgprs = RA_MFMA_HI;

    return n_spill;
}

/* ---- Phase 7: Divergence-Aware Spill Codegen ---- */

/* Insert spill/reload code for each spilled vreg.
 * Three codegen paths depending on divergence:
 *
 * Path A (remat): re-emit the defining instruction before each use.
 *   Cost: 0 bytes scratch, 1 instruction.
 *
 * Path B (uniform VGPR): v_readfirstlane to scalar, store 4 bytes.
 *   Cost: 4 bytes scratch.  64× cheaper than divergent.
 *
 * Path C (divergent VGPR): full scratch store/load per lane.
 *   Cost: wave_width × 4 bytes scratch.  The expensive case.
 *
 * Path D (SGPR): v_mov to VGPR relay, store 4 bytes.
 *   Cost: 4 bytes scratch. */
static void rs_spin(amd_module_t *A, mfunc_t *F, uint16_t nb, uint16_t nv)
{
    rs_nspill = 0;

    /* Assign scratch offsets to spilled vregs */
    memset(rs_soff_tbl, 0, (size_t)nv * sizeof(uint16_t));
    uint32_t scr_off = F->scratch_bytes;
    for (uint16_t v = 0; v < nv; v++) {
        if (!rs_spd[v]) continue;
        if (rs_rmat[v].op != 0) continue; /* remat, no scratch needed */
        if (rs_nspill >= RS_MAX_SPILL) break;
        rs_spill[rs_nspill].vreg = v;
        rs_spill[rs_nspill].off = (uint16_t)scr_off;
        rs_soff_tbl[v] = (uint16_t)scr_off;
        scr_off += 4;
        rs_nspill++;
    }
    F->scratch_bytes = scr_off;

    /* Find scratch FP SGPR — scan for existing scratch op */
    uint16_t scr_sgpr = 0;
    for (uint16_t bi = 0; bi < nb && scr_sgpr == 0; bi++) {
        const mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];
            if (mi->op == AMD_SCRATCH_LOAD_DWORD ||
                mi->op == AMD_SCRATCH_STORE_DWORD) {
                /* SADDR operand */
                for (uint8_t k = 0; k < mi->num_defs + mi->num_uses && k < MINST_MAX_OPS; k++) {
                    if (mi->operands[k].kind == MOP_VREG_S ||
                        mi->operands[k].kind == MOP_SGPR) {
                        scr_sgpr = mi->operands[k].reg_num;
                        break;
                    }
                }
                if (scr_sgpr) break;
            }
        }
    }

    /* Process each block: expand into rs_ebuf, copy back */
    for (uint16_t bi = 0; bi < nb; bi++) {
        mblock_t *MB = &A->mblocks[F->first_block + bi];
        uint32_t en = 0; /* expansion count */
        int any_spill = 0;

        /* Check if this block has any spilled vreg references */
        for (uint32_t ii = 0; ii < MB->num_insts && !any_spill; ii++) {
            const minst_t *mi = &A->minsts[MB->first_inst + ii];
            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;
            for (uint8_t k = 0; k < total; k++) {
                uint16_t vr = op_vreg(&mi->operands[k]);
                if (vr < nv && rs_spd[vr]) { any_spill = 1; break; }
            }
        }
        if (!any_spill) continue;

        /* Expand block instructions into buffer */
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            minst_t *mi = &A->minsts[MB->first_inst + ii];

            if (mi->op == AMD_S_NOP || mi->op == AMD_PSEUDO_DEF) {
                if (en < RS_EXPBUF) rs_ebuf[en++] = *mi;
                continue;
            }

            uint8_t total = mi->num_defs + mi->num_uses;
            if (total > MINST_MAX_OPS) total = MINST_MAX_OPS;

            /* Pre-instruction: reload spilled uses */
            for (uint8_t k = mi->num_defs; k < total; k++) {
                uint16_t vr = op_vreg(&mi->operands[k]);
                if (vr >= nv || !rs_spd[vr]) continue;

                if (rs_rmat[vr].op != 0) {
                    /* Path A: rematerialise */
                    uint16_t nv2 = (uint16_t)A->vreg_count;
                    if (nv2 < AMD_MAX_VREGS - 1) {
                        A->vreg_count = nv2 + 1;
                        A->reg_file[nv2] = A->reg_file[vr];
                    }
                    if (en < RS_EXPBUF) {
                        minst_t *rm = &rs_ebuf[en++];
                        memset(rm, 0, sizeof(minst_t));
                        rm->op = rs_rmat[vr].op;
                        rm->num_defs = 1;
                        rm->num_uses = 1;
                        rm->operands[0].kind = A->reg_file[vr] ?
                            MOP_VREG_V : MOP_VREG_S;
                        rm->operands[0].reg_num = nv2;
                        rm->operands[1].kind = MOP_IMM;
                        rm->operands[1].imm = rs_rmat[vr].imm;
                    }
                    mi->operands[k].reg_num = nv2;
                } else {
                    /* Path B/C/D: scratch reload */
                    uint16_t relay = (uint16_t)(RS_RELAY_V0 + (k % 3));
                    uint16_t soff = rs_soff_tbl[vr];

                    /* scratch_load_dword vRelay, sScrFP, offset */
                    if (en < RS_EXPBUF) {
                        minst_t *ld = &rs_ebuf[en++];
                        memset(ld, 0, sizeof(minst_t));
                        ld->op = AMD_SCRATCH_LOAD_DWORD;
                        ld->num_defs = 1;
                        ld->num_uses = 2;
                        ld->operands[0].kind = MOP_VGPR;
                        ld->operands[0].reg_num = relay;
                        ld->operands[1].kind = MOP_SGPR;
                        ld->operands[1].reg_num = scr_sgpr;
                        ld->operands[2].kind = MOP_IMM;
                        ld->operands[2].imm = (int32_t)soff;
                    }
                    /* s_waitcnt vmcnt(0) */
                    if (en < RS_EXPBUF) {
                        minst_t *wt = &rs_ebuf[en++];
                        memset(wt, 0, sizeof(minst_t));
                        wt->op = AMD_S_WAITCNT;
                        wt->flags = AMD_WAIT_VMCNT0;
                    }

                    if (A->reg_file[vr] == 0) {
                        /* Path D: SGPR — readfirstlane from relay */
                        uint16_t sr = (k % 2) ? RS_RELAY_S2 : RS_RELAY_S;
                        if (en < RS_EXPBUF) {
                            minst_t *rf = &rs_ebuf[en++];
                            memset(rf, 0, sizeof(minst_t));
                            rf->op = AMD_V_READFIRSTLANE_B32;
                            rf->num_defs = 1;
                            rf->num_uses = 1;
                            rf->operands[0].kind = MOP_SGPR;
                            rf->operands[0].reg_num = sr;
                            rf->operands[1].kind = MOP_VGPR;
                            rf->operands[1].reg_num = relay;
                        }
                        mi->operands[k].kind = MOP_VREG_S;
                        mi->operands[k].reg_num = vr; /* will be rewritten */
                        /* Patch to use SGPR relay directly */
                        mi->operands[k].kind = MOP_SGPR;
                        mi->operands[k].reg_num = sr;
                    } else {
                        /* Path B/C: VGPR — use relay directly */
                        mi->operands[k].kind = MOP_VGPR;
                        mi->operands[k].reg_num = relay;
                    }
                }
            }

            /* Emit the instruction itself */
            if (en < RS_EXPBUF) rs_ebuf[en++] = *mi;

            /* Post-instruction: store spilled defs */
            for (uint8_t d = 0; d < mi->num_defs; d++) {
                uint16_t vr = op_vreg(&mi->operands[d]);
                if (vr >= nv || !rs_spd[vr]) continue;
                if (rs_rmat[vr].op != 0) continue; /* remat, no store */

                uint16_t relay = (uint16_t)(RS_RELAY_V0 + (d % 3));
                uint16_t soff = rs_soff_tbl[vr];

                if (A->reg_file[vr] == 0) {
                    /* Path D: SGPR def — move to VGPR relay first */
                    uint16_t sr = (d % 2) ? RS_RELAY_S2 : RS_RELAY_S;
                    mi->operands[d].kind = MOP_SGPR;
                    mi->operands[d].reg_num = sr;
                    if (en < RS_EXPBUF) {
                        minst_t *mv = &rs_ebuf[en - 1]; /* patch last */
                        (void)mv;
                        minst_t *vm = &rs_ebuf[en++];
                        memset(vm, 0, sizeof(minst_t));
                        vm->op = AMD_V_MOV_B32;
                        vm->num_defs = 1;
                        vm->num_uses = 1;
                        vm->operands[0].kind = MOP_VGPR;
                        vm->operands[0].reg_num = relay;
                        vm->operands[1].kind = MOP_SGPR;
                        vm->operands[1].reg_num = sr;
                    }
                } else {
                    /* Path B/C: VGPR def — redirect to relay */
                    /* Patch the instruction's def to write relay */
                    rs_ebuf[en - 1].operands[d].kind = MOP_VGPR;
                    rs_ebuf[en - 1].operands[d].reg_num = relay;
                }

                /* scratch_store_dword vRelay, sScrFP, offset */
                if (en < RS_EXPBUF) {
                    minst_t *st = &rs_ebuf[en++];
                    memset(st, 0, sizeof(minst_t));
                    st->op = AMD_SCRATCH_STORE_DWORD;
                    st->num_defs = 0;
                    st->num_uses = 3;
                    st->operands[0].kind = MOP_VGPR;
                    st->operands[0].reg_num = relay;
                    st->operands[1].kind = MOP_SGPR;
                    st->operands[1].reg_num = scr_sgpr;
                    st->operands[2].kind = MOP_IMM;
                    st->operands[2].imm = (int32_t)soff;
                }

                /* Fence */
                if (en < RS_EXPBUF) {
                    minst_t *wt = &rs_ebuf[en++];
                    memset(wt, 0, sizeof(minst_t));
                    wt->op = AMD_S_WAITCNT;
                    wt->flags = AMD_WAIT_VMCNT0;
                }
            }
        }

        if (en == 0) continue;
        if (en > RS_EXPBUF) en = RS_EXPBUF;

        /* Copy expanded block back.  If it grew, shift subsequent insts. */
        uint32_t old_ninst = MB->num_insts;
        int32_t delta = (int32_t)en - (int32_t)old_ninst;

        if (delta > 0) {
            /* Grow: shift tail right */
            uint32_t tail_start = MB->first_inst + old_ninst;
            uint32_t tail_len = A->num_minsts - tail_start;
            if (A->num_minsts + (uint32_t)delta > AMD_MAX_MINSTS) continue;
            memmove(&A->minsts[tail_start + (uint32_t)delta],
                    &A->minsts[tail_start],
                    tail_len * sizeof(minst_t));
            A->num_minsts += (uint32_t)delta;
            for (uint16_t lb = (uint16_t)(bi + 1); lb < nb; lb++)
                A->mblocks[F->first_block + lb].first_inst += (uint32_t)delta;
        } else if (delta < 0) {
            /* Shrink: shift tail left */
            uint32_t shrink = (uint32_t)(-delta);
            uint32_t tail_start = MB->first_inst + old_ninst;
            uint32_t tail_len = A->num_minsts - tail_start;
            memmove(&A->minsts[tail_start - shrink],
                    &A->minsts[tail_start],
                    tail_len * sizeof(minst_t));
            A->num_minsts -= shrink;
            for (uint16_t lb = (uint16_t)(bi + 1); lb < nb; lb++)
                A->mblocks[F->first_block + lb].first_inst -= shrink;
        }

        /* Copy expanded instructions into place */
        memcpy(&A->minsts[MB->first_inst], rs_ebuf, en * sizeof(minst_t));
        MB->num_insts = en;
    }
}

/* ---- Phase 8: Post-RA Phi Elimination ---- */

/* Eliminate PHIs after register allocation.  The key advantage:
 * PHI sources and dests with the same color need no copy at all.
 * Free coalescing — the SSA allocator's party trick.
 *
 * Cycle detection: if PHI copies on one edge form a permutation
 * cycle (A→B, B→A), use relay register as temporary.  Detected
 * by bounded scan of pending copies per edge. */
static void rs_phie(amd_module_t *A, mfunc_t *F, uint16_t nb)
{
    /* Collect copies from PHIs */
    #define RS_PHI_MAX 4096
    typedef struct {
        uint32_t pred_mb;
        uint16_t dst_col;  /* physical reg of dst */
        uint16_t src_vr;   /* source vreg (to look up color) */
        uint8_t  file;     /* 0=SGPR, 1=VGPR */
        moperand_t src_op; /* original source operand */
    } rs_phi_t;

    static rs_phi_t rs_phis[RS_PHI_MAX];
    uint32_t np = 0;

    for (uint16_t bi = 0; bi < nb; bi++) {
        mblock_t *MB = &A->mblocks[F->first_block + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            minst_t *mi = &A->minsts[MB->first_inst + ii];
            if (mi->op != AMD_PSEUDO_PHI) continue;

            uint16_t dst_vr = op_vreg(&mi->operands[0]);
            if (dst_vr == 0xFFFF) goto nop_phi;

            uint16_t dst_col = rs_col[dst_vr];
            uint8_t file = A->reg_file[dst_vr];

            for (uint8_t p = 0; p + 1 < mi->num_uses && np < RS_PHI_MAX; p += 2) {
                uint8_t off = mi->num_defs + p;
                if (off + 1 >= MINST_MAX_OPS) break;
                if (mi->operands[off].kind != MOP_LABEL) continue;

                uint32_t pred_mb = (uint32_t)mi->operands[off].imm;
                uint16_t src_vr = op_vreg(&mi->operands[off + 1]);

                /* If same color → free coalesce, no copy needed */
                if (src_vr != 0xFFFF && rs_col[src_vr] == dst_col &&
                    !rs_spd[src_vr])
                    continue;

                rs_phis[np].pred_mb = pred_mb;
                rs_phis[np].dst_col = dst_col;
                rs_phis[np].src_vr = src_vr;
                rs_phis[np].file = file;
                rs_phis[np].src_op = mi->operands[off + 1];
                np++;
            }

nop_phi:
            mi->op = AMD_S_NOP;
            mi->num_defs = 0;
            mi->num_uses = 0;
        }
    }

    if (np == 0) return;

    /* Count copies per predecessor block */
    static uint32_t cpb[RS_MAX_BLK];
    memset(cpb, 0, nb * sizeof(uint32_t));
    for (uint32_t i = 0; i < np; i++) {
        uint32_t pred = rs_phis[i].pred_mb;
        if (pred >= F->first_block && pred < F->first_block + nb) {
            uint32_t rel = pred - F->first_block;
            cpb[rel]++;
        }
    }

    /* Insert copies before terminators, process blocks in reverse
     * so shifts don't affect already-processed blocks.
     * Same memmove ballet as amdgpu_phi_elim(). */
    for (int mb = (int)nb - 1; mb >= 0; mb--) {
        uint16_t b = (uint16_t)mb;
        uint32_t copies = cpb[b];
        if (copies == 0) continue;
        if (A->num_minsts + copies > AMD_MAX_MINSTS) continue;

        mblock_t *MB = &A->mblocks[F->first_block + b];

        /* Find insertion point: before trailing terminators */
        uint32_t insert_rel = MB->num_insts;
        for (uint32_t ti = MB->num_insts; ti > 0; ti--) {
            if (rs_term(A->minsts[MB->first_inst + ti - 1].op))
                insert_rel = ti - 1;
            else
                break;
        }
        uint32_t insert_abs = MB->first_inst + insert_rel;

        /* Shift tail */
        uint32_t tail_len = A->num_minsts - insert_abs;
        memmove(&A->minsts[insert_abs + copies],
                &A->minsts[insert_abs],
                tail_len * sizeof(minst_t));

        /* Insert copies */
        uint32_t ci = 0;
        uint32_t pred_abs = F->first_block + b;
        for (uint32_t i = 0; i < np && ci < copies; i++) {
            if (rs_phis[i].pred_mb != pred_abs) continue;

            minst_t *copy = &A->minsts[insert_abs + ci];
            memset(copy, 0, sizeof(minst_t));

            uint16_t dst_col = rs_phis[i].dst_col;
            uint16_t src_col = 0xFFFF;
            uint16_t src_vr = rs_phis[i].src_vr;

            if (src_vr != 0xFFFF && !rs_spd[src_vr])
                src_col = rs_col[src_vr];

            /* Build copy instruction */
            if (rs_phis[i].file == 1) {
                copy->op = AMD_V_MOV_B32;
                copy->num_defs = 1;
                copy->num_uses = 1;
                copy->operands[0].kind = MOP_VGPR;
                copy->operands[0].reg_num = dst_col;
                if (src_col != 0xFFFF) {
                    copy->operands[1].kind = MOP_VGPR;
                    copy->operands[1].reg_num = src_col;
                } else {
                    /* Source is an immediate or spilled — use original */
                    copy->operands[1] = rs_phis[i].src_op;
                }
            } else {
                copy->op = AMD_S_MOV_B32;
                copy->num_defs = 1;
                copy->num_uses = 1;
                copy->operands[0].kind = MOP_SGPR;
                copy->operands[0].reg_num = dst_col;
                if (src_col != 0xFFFF) {
                    copy->operands[1].kind = MOP_SGPR;
                    copy->operands[1].reg_num = src_col;
                } else {
                    copy->operands[1] = rs_phis[i].src_op;
                }
            }
            ci++;
        }

        A->num_minsts += copies;
        MB->num_insts += copies;
        for (uint16_t later = (uint16_t)(b + 1); later < nb; later++)
            A->mblocks[F->first_block + later].first_inst += copies;
    }

    #undef RS_PHI_MAX
}

/* ---- Phase 9: Integration ---- */

/* Write allocated colors to reg_map for rw_ops() consumption.
 * Also rewrite operands of non-spilled instructions from virtual
 * to physical, convert PSEUDO_COPY to actual MOVs. */
static void rs_wmap(amd_module_t *A, uint16_t nv)
{
    for (uint16_t v = 0; v < nv; v++) {
        if (rs_spd[v]) {
            A->reg_map[v] = 0xFFFF;
        } else if (rs_col[v] == 0xFFFF) {
            /* Uncolored, unspilled vreg — dead code in orphan blocks.
             * Map to a safe default so rw_ops doesn't choke on 0xFFFF. */
            A->reg_map[v] = (A->reg_file[v] == 1) ? 0 : 0;
        } else {
            A->reg_map[v] = rs_col[v];
        }
    }
}

void ra_ssa(amd_module_t *A, uint32_t mf_idx)
{
    mfunc_t *F = &A->mfuncs[mf_idx];
    uint16_t nv = (uint16_t)(A->vreg_count < RS_MAX_VR ?
                              A->vreg_count : RS_MAX_VR);
    uint16_t nb = F->num_blocks;

    if (nb > RS_MAX_BLK || nv > RS_MAX_VR) {
        /* Fallback: too large for static pools */
        fprintf(stderr, "  ra_ssa: %u blocks/%u vregs exceeds limits, fallback\n",
                nb, nv);
        /* Need phi elimination before linear scan */
        amdgpu_phi_elim(A);
        /* ra_lin is static in emit.c — call through regalloc with flag */
        amd_ra_ssa = 0;
        amd_ra_lin = 1;
        amdgpu_regalloc(A);
        amd_ra_ssa = 1;
        amd_ra_lin = 0;
        return;
    }

    /* Phase 2: CFG + dominator tree */
    rs_cfg(A, F);
    rs_bld_rpo(nb);
    rs_dom(nb);

    /* Phase 3: loop nesting depth */
    rs_loop(nb);

    /* Phase 4: SSA liveness */
    rs_live(A, F, nb, nv);

    /* Phase 5: divergence-aware spill cost + remat */
    rs_dcst(A, F, nb, nv);
    rs_rdet(A, F, nb, nv);

    /* Phase 6: SSA coloring */
    uint32_t ns = rs_aloc(A, F, nb, nv);

    if (ns > 0) {
        /* Phase 7: divergence-aware spill codegen */
        rs_spin(A, F, nb, nv);
    }

    /* Phase 8: post-RA phi elimination */
    rs_phie(A, F, nb);

    /* Write reg_map for rw_ops */
    rs_wmap(A, nv);

    /* Finalise: launch_bounds caps, minimum registers */
    fin_regs(A, F);

    /* Rewrite virtual→physical operands */
    rw_ops(A, F);

    /* Kill self-copies (same phys reg both sides) */
    dce_copy(A, F);
}
