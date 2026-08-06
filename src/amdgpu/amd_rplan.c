#include "amdgpu.h"
#include <string.h>
#include <stdio.h>

/*
 * Resource planning pass for AMDGPU kernels.
 *
 * Scans BIR, stamps target-specific constants onto mfunc_t.
 * Downstream passes read decisions, never ask questions.
 * Modelled on tensix/coarsen.c — same philosophy, different planet.
 *
 * The GPU hardware has strong opinions and no sense of humour.
 * Our comments compensate.
 */

/* ---- Scan Statistics ---- */

typedef struct {
    uint32_t tid[3];       /* thread ID uses per dim */
    uint32_t bid[3];       /* block ID uses per dim */
    uint32_t n_alloca;     /* scratch allocations */
    uint32_t n_loads;
    uint32_t n_stores;
    uint32_t n_barr;       /* barriers */
    uint32_t n_atom;       /* atomics */
    uint32_t n_shfl;       /* warp shuffles */
    uint8_t  max_dim;      /* highest dim used */
    uint8_t  has_disp;     /* uses blockDim/gridDim */
} rp_stat_t;

/* ---- BIR Scan ---- */

/* Walk the BIR for one kernel, count what it uses.
 * Like a building inspector, except the building is made of
 * register pressure and false confidence. */
static void rp_scan(const bir_module_t *bir, const bir_func_t *F,
                     rp_stat_t *st)
{
    memset(st, 0, sizeof(*st));
    int guard = 262144;

    for (uint32_t bi = 0; bi < F->num_blocks && guard > 0; bi++, guard--) {
        const bir_block_t *B = &bir->blocks[F->first_block + bi];

        for (uint32_t ii = 0; ii < B->num_insts && guard > 0; ii++, guard--) {
            const bir_inst_t *I = &bir->insts[B->first_inst + ii];
            uint32_t dim = I->subop < 3 ? I->subop : 0;

            switch (I->op) {
            case BIR_THREAD_ID:
                st->tid[dim]++;
                if (dim > st->max_dim) st->max_dim = (uint8_t)dim;
                break;
            case BIR_BLOCK_ID:
                st->bid[dim]++;
                if (dim > st->max_dim) st->max_dim = (uint8_t)dim;
                break;
            case BIR_BLOCK_DIM:
            case BIR_GRID_DIM:
                st->has_disp = 1;
                if (dim > st->max_dim) st->max_dim = (uint8_t)dim;
                break;
            case BIR_ALLOCA:        st->n_alloca++; break;
            case BIR_LOAD:          st->n_loads++;  break;
            case BIR_STORE:         st->n_stores++; break;
            case BIR_BARRIER:
            case BIR_BARRIER_GROUP: st->n_barr++;   break;

            case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB:
            case BIR_ATOMIC_AND: case BIR_ATOMIC_OR: case BIR_ATOMIC_XOR:
            case BIR_ATOMIC_MIN: case BIR_ATOMIC_MAX:
            case BIR_ATOMIC_XCHG: case BIR_ATOMIC_CAS:
            case BIR_ATOMIC_LOAD: case BIR_ATOMIC_STORE:
                st->n_atom++; break;

            case BIR_SHFL: case BIR_SHFL_UP:
            case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
            case BIR_BALLOT: case BIR_VOTE_ANY: case BIR_VOTE_ALL:
                st->n_shfl++; break;

            default: break;
            }
        }
    }
    if (st->max_dim > 2) st->max_dim = 2;
}

/* ---- Target Allocation ---- */

/* Stamp target-specific constants.  Every field here is a decision
 * that used to be an is_cdna() call scattered across isel and emit.
 * Now the argument happens once, up front, and everyone else just
 * reads the memo. */
static void rp_alloc(mfunc_t *MF, amd_target_t tgt, const rp_stat_t *st)
{
    int cdna = (tgt <= AMD_TARGET_GFX942);

    MF->exec_w   = cdna ? 1 : 0;
    MF->smem_hz  = cdna ? 1 : 0;
    MF->scr_afs  = cdna ? 1 : 0;
    MF->rp_pad   = 0;
    MF->imp_sgp  = cdna ? 6 : 0;
    MF->sgp_min  = cdna ? 2 : 0;
    MF->wavefront_size = cdna ? AMD_WAVE64 : AMD_WAVE_SIZE;

    /* RSRC1 static mode bits — computed once, OR'd into rsrc1 by emit.
     * The per-kernel VGPR/SGPR block fields come from regalloc and
     * get combined separately.  Target constants and per-kernel
     * arithmetic stay in their proper lanes, like civilised traffic. */
    MF->r1_mode = (3u << 16) |    /* FLOAT_DENORM_32 = preserve */
                  (3u << 18) |    /* FLOAT_DENORM_16_64 = preserve */
                  (1u << 21) |    /* DX10_CLAMP */
                  (1u << 23);     /* IEEE_MODE */
    if (!cdna) {
        MF->r1_mode |= (1u << 26) |  /* WGP_MODE */
                        (1u << 27);   /* MEM_ORDERED */
    }

    /* Stamp dispatch/dim info from scan */
    MF->needs_dispatch = st->has_disp;
    MF->max_dim = st->max_dim;
}

/* ---- Public Entry ---- */

void amd_rplan(amd_module_t *A)
{
    int guard = 8192;
    for (uint32_t fi = 0; fi < A->num_mfuncs && guard > 0; fi++, guard--) {
        mfunc_t *MF = &A->mfuncs[fi];
        if (!MF->is_kernel) continue;

        rp_stat_t st;
        rp_scan(A->bir, &A->bir->funcs[MF->bir_func], &st);
        rp_alloc(MF, A->target, &st);

        const char *name = A->bir->strings + MF->name;
        /* stderr, not stdout: --amdgpu writes assembly to stdout when no -o is
         * given, and a diagnostic in that stream is a .s that won't assemble. */
        fprintf(stderr, "  rplan %s: wave%u, sgp_imp=%u, scratch=%s, "
                "%u loads, %u stores\n",
                name, MF->wavefront_size, MF->imp_sgp,
                st.n_alloca ? "yes" : "no",
                st.n_loads, st.n_stores);
    }
}
