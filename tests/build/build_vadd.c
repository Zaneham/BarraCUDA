/* build_vadd.c -- vadd through booth_build; should print what the CUDA
 * frontend prints for examples/cmake/vadd.cu. */

#include <stdio.h>
#include "booth_build.h"

int main(int argc, char **argv)
{
    bb_t *B = bb_new();
    if (B == NULL) { fprintf(stderr, "out of memory\n"); return 1; }

    uint32_t v    = bb_void(B);
    uint32_t i32  = bb_int(B, 32);
    uint32_t f32  = bb_flt(B, 32);
    uint32_t pf32 = bb_ptr(B, f32, BB_AS_GLOBAL);

    uint32_t params[4] = { pf32, pf32, pf32, i32 };
    bb_func(B, "vadd", v, params, 4, 1);

    /* Reserve before emitting, so branches can name them. */
    uint32_t entry = bb_blk(B, "entry");
    uint32_t then_ = bb_blk(B, "if.then");
    uint32_t end   = bb_blk(B, "if.end");

    bb_line(B, 3);
    bb_open(B, entry);
    bb_val a = bb_param(B, 0, pf32);
    bb_val b = bb_param(B, 1, pf32);
    bb_val c = bb_param(B, 2, pf32);
    bb_val n = bb_param(B, 3, i32);

    /* One call per statement; nesting would reorder the emission. */
    bb_val bid  = bb_bid(B, 0);
    bb_val bdim = bb_bdim(B, 0);
    bb_val m    = bb_mul(B, i32, bid, bdim);
    bb_val tid  = bb_tid(B, 0);
    bb_val i    = bb_add(B, i32, m, tid);

    bb_line(B, 4);
    bb_val cond = bb_icmp(B, BB_SLT, i, n);
    bb_brif(B, cond, then_, end, end);
    bb_close(B);

    bb_open(B, then_);
    bb_val pc = bb_gep(B, pf32, c, i);
    bb_val pa = bb_gep(B, pf32, a, i);
    bb_val va = bb_load(B, f32, pa);
    bb_val pb = bb_gep(B, pf32, b, i);
    bb_val vb = bb_load(B, f32, pb);
    bb_val s  = bb_fadd(B, f32, va, vb);
    bb_store(B, s, pc);
    bb_br(B, end);
    bb_close(B);

    bb_open(B, end);
    bb_ret(B);
    bb_close(B);

    bb_fend(B);

    if (bb_full(B)) { fprintf(stderr, "a pool overflowed\n"); bb_free(B); return 1; }

    int rc = 0;
    if (argc > 1) {
        rc = bb_emit(B, "cpu", argv[1]);
        fprintf(stderr, "bb_emit -> %d\n", rc);
    } else {
        rc = bb_print(B, NULL);
    }

    bb_free(B);
    return rc == 0 ? 0 : 1;
}
