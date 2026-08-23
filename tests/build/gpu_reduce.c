/* gpu_reduce.c -- run the OCaml reduction, which needs a loop and a ref.
 * a[i] = 1, so thread t must get exactly n - t. Every thread differs, so a
 * wrong index or an off-by-one in the loop shows up rather than averaging out.
 *
 *   kath --bir-in --nvidia-ptx reduce.bir -o reduce.ptx
 *   gcc tests/build/gpu_reduce.c src/nvidia/nv_rt.c -Isrc/nvidia -o gpu_reduce -lm
 *   ./gpu_reduce reduce.ptx
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "booth/nv_rt.h"

#define N 4096

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "reduce.ptx";

    nv_dev_t dev;
    if (nv_rt_init(&dev) != NV_RT_OK) { fprintf(stderr, "no CUDA device\n"); return 1; }

    nv_kern_t k;
    if (nv_rt_load(&dev, ptx, "reduce", &k) != NV_RT_OK) {
        fprintf(stderr, "could not load %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }

    float *a = malloc(N * sizeof *a);
    float *o = malloc(N * sizeof *o);
    for (int i = 0; i < N; i++) { a[i] = 1.0f; o[i] = -1.0f; }

    CUdevptr da = nv_rt_alloc(&dev, N * sizeof *a);
    CUdevptr dobuf = nv_rt_alloc(&dev, N * sizeof *o);
    nv_rt_h2d(&dev, da, a, N * sizeof *a);

    int n = N;
    void *args[] = { &da, &dobuf, &n };

    unsigned block = 256;
    if (nv_rt_launch(&dev, &k, (N + block - 1) / block, 1, 1, block, 1, 1, 0, args)
        != NV_RT_OK) {
        fprintf(stderr, "launch failed\n");
        nv_rt_shut(&dev);
        return 1;
    }
    nv_rt_sync(&dev);
    nv_rt_d2h(&dev, o, dobuf, N * sizeof *o);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        float want = (float)(N - i);
        if (fabsf(o[i] - want) > 0.0f) {
            if (bad < 3) printf("o[%d] = %g want %g\n", i, o[i], want);
            bad++;
        }
    }

    if (bad == 0)
        printf("all %d sums correct, o[0]=%g o[%d]=%g\n"
               "OK: an OCaml loop with an accumulator ran on the GPU.\n",
               N, o[0], N - 1, o[N - 1]);
    else
        printf("FAILED: %d of %d wrong\n", bad, N);

    nv_rt_free(&dev, da); nv_rt_free(&dev, dobuf);
    nv_rt_unload(&dev, &k);
    nv_rt_shut(&dev);
    free(a); free(o);
    return bad == 0 ? 0 : 1;
}
