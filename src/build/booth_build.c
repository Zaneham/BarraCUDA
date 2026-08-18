/* booth_build.c -- see booth_build.h. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bir.h"
#include "backend.h"
#include "backend_cfg.h"
#include "booth_build.h"

/* Bound only; a runaway caller is refused rather than scribbling past M. */
#define BB_MAX_BLK 4096

struct bb {
    bir_module_t *M;
    uint32_t      func;   /* index into M->funcs, or BB_NONE when none is open */
    uint32_t      cur;    /* block being filled, or BB_NONE */
    uint32_t      line;   /* source line stamped on each instruction emitted */
    uint32_t      nblk;   /* blocks reserved in the open function */
};

/* ---- Lifecycle ---- */

bb_t *bb_new(void)
{
    bb_t *B = calloc(1, sizeof *B);
    if (B == NULL) return NULL;
    B->M = malloc(sizeof *B->M);
    if (B->M == NULL) { free(B); return NULL; }
    bir_module_init(B->M);
    B->func = BB_NONE;
    B->cur  = BB_NONE;
    return B;
}

void bb_free(bb_t *B)
{
    if (B == NULL) return;
    free(B->M);
    free(B);
}

int bb_full(const bb_t *B) { return B != NULL && B->M->pool_full != 0; }

struct bir_module *bb_module(bb_t *B)
{
    return (B != NULL) ? (struct bir_module *)B->M : NULL;
}

/* ---- Types ---- */

uint32_t bb_void(bb_t *B)                     { return bir_type_void(B->M); }
uint32_t bb_int (bb_t *B, int bits)           { return bir_type_int(B->M, bits); }
uint32_t bb_flt (bb_t *B, int bits)           { return bir_type_float(B->M, bits); }
uint32_t bb_ptr (bb_t *B, uint32_t p, int as) { return bir_type_ptr(B->M, p, as); }

/* ---- Emission ---- */

static bb_val emit(bb_t *B, uint16_t op, uint8_t subop, uint32_t ty,
                   const uint32_t *ops, uint8_t n)
{
    bir_module_t *M = B->M;
    if (M->num_insts >= BIR_MAX_INSTS) {
        bir_pfull(M, BIR_P_INSTS);
        return BB_NONE;
    }
    uint32_t i = M->num_insts++;
    bir_inst_t *I = &M->insts[i];
    memset(I, 0, sizeof *I);
    I->op = op;
    I->subop = subop;
    I->type = ty;
    I->num_operands = n;
    for (uint8_t k = 0; k < n && k < BIR_OPERANDS_INLINE; k++)
        I->operands[k] = ops[k];
    M->inst_lines[i] = B->line;
    return i;
}

void bb_line(bb_t *B, uint32_t line) { B->line = line; }

/* ---- Immediates ---- */

bb_val bb_ci(bb_t *B, uint32_t ty, int64_t v)
{
    return BIR_MAKE_CONST(bir_const_int(B->M, ty, v));
}

bb_val bb_cf(bb_t *B, uint32_t ty, double v)
{
    return BIR_MAKE_CONST(bir_const_float(B->M, ty, v));
}

int bb_isflt(const bb_t *B, uint32_t ty)
{
    if (B == NULL || ty >= B->M->num_types) return 0;
    uint8_t k = B->M->types[ty].kind;
    return k == BIR_TYPE_FLOAT || k == BIR_TYPE_BFLOAT;
}

/* ---- Functions ---- */

int bb_func(bb_t *B, const char *name, uint32_t ret,
            const uint32_t *params, int nparams, int kernel)
{
    bir_module_t *M = B->M;
    if (B->func != BB_NONE) return -1;              /* already open */
    if (M->num_funcs >= BIR_MAX_FUNCS) { bir_pfull(M, BIR_P_FUNCS); return -1; }

    uint32_t t = bir_type_func(M, ret, params, nparams);
    uint32_t f = M->num_funcs++;
    bir_func_t *F = &M->funcs[f];
    memset(F, 0, sizeof *F);
    F->name        = bir_add_string(M, name, (uint32_t)strlen(name));
    F->type        = t;
    F->first_block = M->num_blocks;
    F->num_params  = (uint16_t)nparams;
    F->cuda_flags  = kernel ? CUDA_GLOBAL : CUDA_DEVICE;

    B->func = f;
    B->nblk = 0;
    return 0;
}

void bb_fend(bb_t *B)
{
    if (B->func == BB_NONE) return;
    bir_module_t *M = B->M;
    bir_func_t *F = &M->funcs[B->func];
    F->num_blocks  = (uint16_t)(M->num_blocks - F->first_block);
    F->total_insts = M->num_insts;
    B->func = BB_NONE;
}

/* ---- Blocks ---- */

uint32_t bb_blk(bb_t *B, const char *name)
{
    bir_module_t *M = B->M;
    if (B->nblk >= BB_MAX_BLK || M->num_blocks >= BIR_MAX_BLOCKS) {
        bir_pfull(M, BIR_P_BLOCKS);
        return BB_NONE;
    }
    /* Index is claimed now so a branch can name it; bb_close fills the body. */
    B->nblk++;
    uint32_t b = M->num_blocks++;
    M->blocks[b].name       = bir_add_string(M, name, (uint32_t)strlen(name));
    M->blocks[b].first_inst = 0;
    M->blocks[b].num_insts  = 0;
    return b;   /* absolute index, which is what a branch operand holds */
}

void bb_open(bb_t *B, uint32_t blk)
{
    B->cur = blk;
    B->M->blocks[blk].first_inst = B->M->num_insts;
}

void bb_close(bb_t *B)
{
    if (B->cur == BB_NONE) return;
    bir_block_t *bl = &B->M->blocks[B->cur];
    bl->num_insts = B->M->num_insts - bl->first_inst;
    B->cur = BB_NONE;
}

/* ---- Instructions ---- */

bb_val bb_param(bb_t *B, int idx, uint32_t ty)
{ return emit(B, BIR_PARAM, (uint8_t)idx, ty, NULL, 0); }

bb_val bb_tid (bb_t *B, int d) { return emit(B, BIR_THREAD_ID, (uint8_t)d, bb_int(B,32), NULL, 0); }
bb_val bb_bid (bb_t *B, int d) { return emit(B, BIR_BLOCK_ID,  (uint8_t)d, bb_int(B,32), NULL, 0); }
bb_val bb_bdim(bb_t *B, int d) { return emit(B, BIR_BLOCK_DIM, (uint8_t)d, bb_int(B,32), NULL, 0); }
bb_val bb_gdim(bb_t *B, int d) { return emit(B, BIR_GRID_DIM,  (uint8_t)d, bb_int(B,32), NULL, 0); }

static bb_val bin(bb_t *B, uint16_t op, uint32_t ty, bb_val a, bb_val b)
{
    uint32_t o[2] = { BIR_MAKE_VAL(a), BIR_MAKE_VAL(b) };
    return emit(B, op, 0, ty, o, 2);
}

bb_val bb_add (bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_ADD,  t, a, b); }
bb_val bb_sub (bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_SUB,  t, a, b); }
bb_val bb_mul (bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_MUL,  t, a, b); }
bb_val bb_fadd(bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_FADD, t, a, b); }
bb_val bb_fsub(bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_FSUB, t, a, b); }
bb_val bb_fmul(bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_FMUL, t, a, b); }

bb_val bb_icmp(bb_t *B, int pred, bb_val a, bb_val b)
{
    uint32_t o[2] = { BIR_MAKE_VAL(a), BIR_MAKE_VAL(b) };
    return emit(B, BIR_ICMP, (uint8_t)pred, bb_int(B, 1), o, 2);
}

bb_val bb_gep(bb_t *B, uint32_t ty, bb_val base, bb_val idx)
{
    uint32_t o[2] = { BIR_MAKE_VAL(base), BIR_MAKE_VAL(idx) };
    return emit(B, BIR_GEP, 0, ty, o, 2);
}

bb_val bb_load(bb_t *B, uint32_t ty, bb_val addr)
{
    uint32_t o[1] = { BIR_MAKE_VAL(addr) };
    return emit(B, BIR_LOAD, 0, ty, o, 1);
}

void bb_store(bb_t *B, bb_val v, bb_val addr)
{
    uint32_t o[2] = { BIR_MAKE_VAL(v), BIR_MAKE_VAL(addr) };
    emit(B, BIR_STORE, 0, bb_void(B), o, 2);
}

void bb_br(bb_t *B, uint32_t blk)
{
    uint32_t o[1] = { blk };
    emit(B, BIR_BR, 0, bb_void(B), o, 1);
}

void bb_brif(bb_t *B, bb_val c, uint32_t t, uint32_t f, uint32_t merge)
{
    uint32_t o[4] = { BIR_MAKE_VAL(c), t, f, merge };
    emit(B, BIR_BR_COND, 0, bb_void(B), o, 4);
}

void bb_ret(bb_t *B) { emit(B, BIR_RET, 0, bb_void(B), NULL, 0); }

/* ---- Output ---- */

int bb_print(bb_t *B, const char *path)
{
    FILE *f = (path == NULL) ? stdout : fopen(path, "w");
    if (f == NULL) return -1;
    bir_print_module(B->M, f);
    fprintf(f, "\n; %u functions, %u globals, %u instructions\n",
            B->M->num_funcs, B->M->num_globals, B->M->num_insts);
    if (f != stdout) fclose(f);
    return 0;
}

/* Until the registry takes a name, selection goes through its flag. These are
   the flags that set `on`; the others each backend declares only pick a target
   variant, so passing one selects nothing and be_run then does nothing. */
static const char *flag_for(const char *target)
{
    if (strcmp(target, "cpu")     == 0) return "--cpu";
    if (strcmp(target, "rv64")    == 0) return "--rv64";
    if (strcmp(target, "amdgpu")  == 0) return "--amdgpu";
    if (strcmp(target, "nvptx")   == 0) return "--nvidia-ptx";
    if (strcmp(target, "tensix")  == 0) return "--tensix";
    if (strcmp(target, "metal")   == 0) return "--metal";
    if (strcmp(target, "intel")   == 0) return "--intel-spirv";
    return NULL;
}

int bb_emit(bb_t *B, const char *target, const char *path)
{
    if (bir_pchk(B->M, "booth_build") != 0) return -1;

    const char *flag = flag_for(target);
    if (flag == NULL) {
        fprintf(stderr, "booth_build: unknown target '%s'\n", target);
        return -1;
    }

    int used = 0;
    if (be_parse_flag(flag, NULL, &used) != 1) {
        fprintf(stderr, "booth_build: no backend claimed '%s'\n", flag);
        return -1;
    }

    be_cfg_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.output_file = path;

    return be_run((const struct bir_module *)B->M,
                  (const struct be_cfg *)&cfg);
}
