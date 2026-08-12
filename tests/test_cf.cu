/* test_cf.cu — Constant folding test cases.
 *
 * Each kernel targets a specific constant folding edge case.
 * The test harness compiles with --ir and checks which
 * instructions survived. */

/* Integer arithmetic with constant operands */
__global__ void cf_int_arith(int *out, int x) {
    out[0] = x + (3 + 4);
    out[1] = x + (10 - 3);
}

/* Chained folding: a=2+3, b=a*4, c=b+x */
__global__ void cf_chain(int *out, int x) {
    int a = 2 + 3;
    int b = a * 4;
    out[0] = b + x;
}

/* Integer comparison + select folds away */
__global__ void cf_icmp_select(int *out, int x) {
    out[0] = (2 < 5) ? x : 0;
}

/* Integer division by zero must not fold (undefined behavior) */
__global__ void cf_divzero(int *out) {
    out[0] = 1 / 0;
}

/* Integer/float conversions with constant operands */
__global__ void cf_conv(int *iout, float *fout) {
    fout[0] = (float)42;      /* sitofp */
    iout[0] = (int)3.14f;     /* fptosi */
}

/* Float arithmetic with constant operands */
__global__ void cf_float(float *out) {
    out[0] = 1.5f + 2.5f;
}


/* Subtraction, folding to a value nothing else in this file produces. The
 * existing 10-3 folds to the same 7 as the 3+4 above it, so a fold that runs
 * its operands backwards was invisible. */
__global__ void cf_sub(int *out, int x) {
    out[0] = x + (20 - 6);
}
