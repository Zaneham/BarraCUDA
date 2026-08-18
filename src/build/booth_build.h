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

enum { BB_EQ, BB_NE, BB_SLT, BB_SLE, BB_SGT, BB_SGE,
       BB_ULT, BB_ULE, BB_UGT, BB_UGE };

/* The module is tens of megabytes, so the builder heap-allocates it. */
bb_t *bb_new(void);
void  bb_free(bb_t *B);

uint32_t bb_void (bb_t *B);
uint32_t bb_int  (bb_t *B, int bits);
uint32_t bb_flt  (bb_t *B, int bits);
uint32_t bb_ptr  (bb_t *B, uint32_t pointee, int addrspace);

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

bb_val bb_gep   (bb_t *B, uint32_t ty, bb_val base, bb_val idx);
bb_val bb_load  (bb_t *B, uint32_t ty, bb_val addr);
void   bb_store (bb_t *B, bb_val v, bb_val addr);

void   bb_br    (bb_t *B, uint32_t blk);
void   bb_brif  (bb_t *B, bb_val cond, uint32_t t, uint32_t f, uint32_t merge);
void   bb_ret   (bb_t *B);

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
