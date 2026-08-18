/* run_vadd.c -- link a Booth --cpu kernel and check the arithmetic.
 * ABI is the kernel's params then a hidden nthreads; the body runs once per
 * thread_id from 0 to nthreads.
 *
 *   kath --bir-in --cpu vadd.bir -o vadd.o
 *   gcc -no-pie tests/build/run_vadd.c vadd.o -o run_vadd
 */

#include <stdio.h>
#include <math.h>

extern void vadd(const float *a, const float *b, float *c, int n, int nthreads);

#define N 16

int main(void)
{
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(100 * i);
        c[i] = -1.0f;
    }

    vadd(a, b, c, N, N);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        float want = a[i] + b[i];
        if (fabsf(c[i] - want) > 0.0f) {
            printf("c[%2d] = %-10g want %-10g  MISMATCH\n", i, c[i], want);
            bad++;
        }
    }

    if (bad == 0) {
        printf("all %d elements correct, c[0]=%g c[%d]=%g\n",
               N, c[0], N - 1, c[N - 1]);
        printf("OK: a kernel written in OCaml computed the right answer.\n");
        return 0;
    }
    printf("FAILED: %d of %d wrong\n", bad, N);
    return 1;
}
