/* gpu_vadd.c -- run the OCaml-authored kernel on a real NVIDIA GPU.
 * PTX is JITed by the driver, so no CUDA SDK is needed.
 *
 *   kath --bir-in --nvidia-ptx vadd.bir -o vadd.ptx
 *   gcc tests/build/gpu_vadd.c src/nvidia/nv_rt.c -Isrc/nvidia -o gpu_vadd
 *   ./gpu_vadd vadd.ptx
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nv_rt.h"

#define N 4096

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "vadd.ptx";

    nv_dev_t dev;
    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "no CUDA device\n");
        return 1;
    }

    nv_kern_t k;
    if (nv_rt_load(&dev, ptx, "vadd", &k) != NV_RT_OK) {
        fprintf(stderr, "could not load %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }
    printf("loaded %s, kernel vadd\n", ptx);

    float *a = malloc(N * sizeof *a);
    float *b = malloc(N * sizeof *b);
    float *c = malloc(N * sizeof *c);
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(1000 - i);
        c[i] = -1.0f;
    }

    CUdevptr da = nv_rt_alloc(&dev, N * sizeof *a);
    CUdevptr db = nv_rt_alloc(&dev, N * sizeof *b);
    CUdevptr dc = nv_rt_alloc(&dev, N * sizeof *c);

    nv_rt_h2d(&dev, da, a, N * sizeof *a);
    nv_rt_h2d(&dev, db, b, N * sizeof *b);

    int n = N;
    void *args[] = { &da, &db, &dc, &n };

    unsigned block = 256;
    unsigned grid  = (N + block - 1) / block;
    if (nv_rt_launch(&dev, &k, grid, 1, 1, block, 1, 1, 0, args) != NV_RT_OK) {
        fprintf(stderr, "launch failed\n");
        nv_rt_shut(&dev);
        return 1;
    }
    nv_rt_sync(&dev);
    nv_rt_d2h(&dev, c, dc, N * sizeof *c);

    int bad = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(c[i] - (a[i] + b[i])) > 0.0f) {
            if (bad < 3)
                printf("c[%d] = %g want %g\n", i, c[i], a[i] + b[i]);
            bad++;
        }

    if (bad == 0)
        printf("all %d elements correct, c[0]=%g c[%d]=%g\n"
               "OK: an OCaml kernel ran on the GPU.\n", N, c[0], N - 1, c[N - 1]);
    else
        printf("FAILED: %d of %d wrong\n", bad, N);

    nv_rt_free(&dev, da); nv_rt_free(&dev, db); nv_rt_free(&dev, dc);
    nv_rt_unload(&dev, &k);
    nv_rt_shut(&dev);
    free(a); free(b); free(c);
    return bad == 0 ? 0 : 1;
}
