/* gpu_scale.c -- check that float immediates survive the BIR text round-trip.
 * An immediate read back as an integer would turn 2.5 into 2 and still run.
 *
 *   kath --bir-in --nvidia-ptx scale.bir -o scale.ptx
 *   gcc tests/build/gpu_scale.c src/nvidia/nv_rt.c -Isrc/nvidia -o gpu_scale -lm
 *   ./gpu_scale scale.ptx
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nv_rt.h"

#define N 4096

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "scale.ptx";

    nv_dev_t dev;
    if (nv_rt_init(&dev) != NV_RT_OK) { fprintf(stderr, "no CUDA device\n"); return 1; }

    nv_kern_t k;
    if (nv_rt_load(&dev, ptx, "scale", &k) != NV_RT_OK) {
        fprintf(stderr, "could not load %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }

    float *a = malloc(N * sizeof *a);
    float *b = malloc(N * sizeof *b);
    for (int i = 0; i < N; i++) { a[i] = (float)(i % 17); b[i] = -1.0f; }

    CUdevptr da = nv_rt_alloc(&dev, N * sizeof *a);
    CUdevptr db = nv_rt_alloc(&dev, N * sizeof *b);
    nv_rt_h2d(&dev, da, a, N * sizeof *a);

    int n = N;
    void *args[] = { &da, &db, &n };

    unsigned block = 256;
    if (nv_rt_launch(&dev, &k, (N + block - 1) / block, 1, 1, block, 1, 1, 0, args)
        != NV_RT_OK) {
        fprintf(stderr, "launch failed\n");
        nv_rt_shut(&dev);
        return 1;
    }
    nv_rt_sync(&dev);
    nv_rt_d2h(&dev, b, db, N * sizeof *b);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        float want = a[i] * 2.5f + 1.0f;
        if (fabsf(b[i] - want) > 1e-5f) {
            if (bad < 3) printf("b[%d] = %g want %g\n", i, b[i], want);
            bad++;
        }
    }

    if (bad == 0)
        printf("all %d elements correct, b[3]=%g (3*2.5+1)\n"
               "OK: float immediates survived.\n", N, b[3]);
    else
        printf("FAILED: %d of %d wrong\n", bad, N);

    nv_rt_free(&dev, da); nv_rt_free(&dev, db);
    nv_rt_unload(&dev, &k);
    nv_rt_shut(&dev);
    free(a); free(b);
    return bad == 0 ? 0 : 1;
}
