/* booth_build.h -- construct a BIR module without linking against the tree. */
#ifndef BOOTH_BUILD_H
#define BOOTH_BUILD_H

#include <stdint.h>
#include <stddef.h>

typedef struct bb bb_t;

typedef uint32_t bb_val;
#define BB_NONE 0xFFFFFFFFu

/* Mirrors bir_addrspace_t, so a caller needs this header only. */
enum { BB_AS_PRIVATE, BB_AS_SHARED, BB_AS_GLOBAL, BB_AS_CONST, BB_AS_GENERIC };

/* Ordered in the same sequence as bir_cmp_kind_t, so pred passes straight
   through as the subop. The O predicates are the ordered float ones. */
enum { BB_EQ, BB_NE, BB_SLT, BB_SLE, BB_SGT, BB_SGE,
       BB_ULT, BB_ULE, BB_UGT, BB_UGE,
       BB_OEQ, BB_ONE, BB_OLT, BB_OLE, BB_OGT, BB_OGE };

/* The module is tens of megabytes, so the builder heap-allocates it. */
bb_t *bb_new(void);
void  bb_free(bb_t *B);

uint32_t bb_void (bb_t *B);
uint32_t bb_int  (bb_t *B, int bits);
uint32_t bb_flt  (bb_t *B, int bits);
uint32_t bb_ptr  (bb_t *B, uint32_t pointee, int addrspace);
uint32_t bb_arr  (bb_t *B, uint32_t elem, uint32_t count);
uint32_t bb_vec  (bb_t *B, uint32_t elem, uint32_t lanes);

/* One function open at a time. */
int  bb_func (bb_t *B, const char *name, uint32_t ret,
              const uint32_t *params, int nparams, int kernel);
void bb_fend (bb_t *B);

/* Reserve every block before emitting: a branch names blocks with no body yet. */
uint32_t bb_blk   (bb_t *B, const char *name);
void     bb_open  (bb_t *B, uint32_t blk);
void     bb_close (bb_t *B);

/* Do not nest these. Each appends, and argument evaluation order is
   unspecified in both C and OCaml. One call per statement. */
bb_val bb_param (bb_t *B, int idx, uint32_t ty);

/* Immediates are operands, not instructions, so these emit nothing. */
bb_val bb_ci    (bb_t *B, uint32_t ty, int64_t v);
bb_val bb_cf    (bb_t *B, uint32_t ty, double v);
int    bb_isflt (const bb_t *B, uint32_t ty);

bb_val bb_tid   (bb_t *B, int dim);   /* dim: 0=x 1=y 2=z */
bb_val bb_bid   (bb_t *B, int dim);
bb_val bb_bdim  (bb_t *B, int dim);
bb_val bb_gdim  (bb_t *B, int dim);

bb_val bb_add   (bb_t *B, uint32_t ty, bb_val a, bb_val b);
bb_val bb_sub   (bb_t *B, uint32_t ty, bb_val a, bb_val b);
bb_val bb_mul   (bb_t *B, uint32_t ty, bb_val a, bb_val b);
bb_val bb_fadd  (bb_t *B, uint32_t ty, bb_val a, bb_val b);
bb_val bb_fsub  (bb_t *B, uint32_t ty, bb_val a, bb_val b);
bb_val bb_fmul  (bb_t *B, uint32_t ty, bb_val a, bb_val b);
bb_val bb_icmp  (bb_t *B, int pred, bb_val a, bb_val b);
bb_val bb_fcmp  (bb_t *B, int pred, bb_val a, bb_val b);

/* Mirrors of the BIR opcode groups, so a caller needs only this header.
   Names match what bir_print writes, which is what bir_parse reads back. */
enum bb_op2 {
    BB_ADD, BB_SUB, BB_MUL, BB_SDIV, BB_UDIV, BB_SREM, BB_UREM,
    BB_FADD, BB_FSUB, BB_FMUL, BB_FDIV, BB_FREM,
    BB_AND, BB_OR, BB_XOR, BB_SHL, BB_LSHR, BB_ASHR
};

enum bb_fn {
    BB_SQRT, BB_RSQ, BB_RCP, BB_EXP2, BB_LOG2, BB_SIN, BB_COS,
    BB_FABS, BB_FLOOR, BB_CEIL, BB_FTRUNC, BB_RNDNE,   /* one operand */
    BB_FMAX, BB_FMIN                                   /* two */
};

enum bb_cvt {
    BB_TRUNC, BB_ZEXT, BB_SEXT, BB_FPTRUNC, BB_FPEXT,
    BB_FPTOSI, BB_FPTOUI, BB_SITOFP, BB_UITOFP, BB_BITCAST
};

bb_val bb_op    (bb_t *B, int op, uint32_t ty, bb_val a, bb_val b);
bb_val bb_fn1   (bb_t *B, int fn, uint32_t ty, bb_val a);
bb_val bb_fn2   (bb_t *B, int fn, uint32_t ty, bb_val a, bb_val b);
bb_val bb_cvt   (bb_t *B, int c, uint32_t dst, bb_val a);
bb_val bb_sel   (bb_t *B, uint32_t ty, bb_val c, bb_val t, bb_val f);

/* Intrinsics print without a type, so the reader has to ask what it just read. */
uint32_t bb_tyof(const bb_t *B, bb_val v);

/* ty is the pointer type, not the pointee. mem2reg promotes these away. */
bb_val bb_alca  (bb_t *B, uint32_t ty);

/* Per-block scratchpad, one allocation for every thread in the block. */
bb_val bb_shal  (bb_t *B, uint32_t ty);
void   bb_barr  (bb_t *B);

/* Read, modify, write, returning the old value. Ordering mirrors
   bir_order_t; min and max are deliberately absent, see bb_atom. */
enum bb_order { BB_RELAXED, BB_ACQUIRE, BB_RELEASE, BB_ACQ_REL, BB_SEQ_CST };
enum bb_atom  { BB_A_ADD, BB_A_SUB, BB_A_AND, BB_A_OR, BB_A_XOR, BB_A_XCHG };
bb_val bb_atom  (bb_t *B, int op, int order, uint32_t ty, bb_val p, bb_val v);

bb_val bb_gep   (bb_t *B, uint32_t ty, bb_val base, bb_val idx);
bb_val bb_load  (bb_t *B, uint32_t ty, bb_val addr);
void   bb_store (bb_t *B, bb_val v, bb_val addr);

void   bb_br    (bb_t *B, uint32_t blk);
void   bb_brif  (bb_t *B, bb_val cond, uint32_t t, uint32_t f, uint32_t merge);
void   bb_ret   (bb_t *B);
void   bb_retv  (bb_t *B, uint32_t ty, bb_val v);

/* Up to 5 arguments, which is what fits beside the callee in one instruction.
   bb_ffind returns the index bb_call wants, or -1 when the name is unknown. */
#define BB_MAX_ARGS 5
int    bb_ffind (const bb_t *B, const char *name);
bb_val bb_call  (bb_t *B, uint32_t ty, int fidx, const bb_val *args, int nargs);

/* Source line stamped on instructions emitted after this. */
void   bb_line  (bb_t *B, uint32_t line);

int bb_print (bb_t *B, const char *path);

/* target: "cpu", "rv64", "amdgpu", "ptx", "tensix", "metal", "intel".
   Skips the IR passes; run_bir_backends is the path that runs them. */
int bb_emit (bb_t *B, const char *target, const char *path);

/* Pools refuse quietly by design, so check before emitting. */
int bb_full (const bb_t *B);

struct bir_module;
struct bir_module *bb_module(bb_t *B);   /* borrowed, dies with the builder */

#endif /* BOOTH_BUILD_H */
