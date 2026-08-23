/* tnv_rt.c — BarraCUDA NVIDIA Runtime Test
 *
 * Loads vector_add.ptx (compiled by BarraCUDA) and actually runs it
 * on whatever NVIDIA GPU is sitting in the machine. The full loop:
 * open-source compiler → closed-source driver → actual silicon.
 * The ouroboros of GPU computing.
 *
 * Usage:
 *   gcc -O2 -Isrc/nvidia tnv_rt.c nv_rt.c -o tnv_rt
 *   ./barracuda --nvidia-ptx tests/vector_add.cu -o va.ptx
 *   ./tnv_rt va.ptx
 */

#include "booth/nv_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N_ELEMS  (1 << 16)   /* 64K elements — enough to need many blocks */
#define BLK_SZ   256

static float h_a[N_ELEMS];
static float h_b[N_ELEMS];
static float h_c[N_ELEMS];

int main(int argc, char **argv)
{
    int errs = 0;
    const char *ptx = "tests/vector_add.ptx";
    if (argc > 1) ptx = argv[1];

    /* ---- Init CUDA ---- */
    nv_dev_t dev;
    int rc = nv_rt_init(&dev);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "tnv_rt: init failed (%d)\n", rc);
        return 1;
    }

    /* ---- Load PTX ---- */
    nv_kern_t kern;
    rc = nv_rt_load(&dev, ptx, "vectorAdd", &kern);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "tnv_rt: load failed (%d)\n", rc);
        nv_rt_shut(&dev);
        return 1;
    }

    /* ---- Prepare host data ---- */
    for (int i = 0; i < N_ELEMS; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
        h_c[i] = -1.0f;
    }

    /* ---- Allocate device memory ---- */
    size_t bytes = (size_t)N_ELEMS * sizeof(float);
    CUdevptr d_a = nv_rt_alloc(&dev, bytes);
    CUdevptr d_b = nv_rt_alloc(&dev, bytes);
    CUdevptr d_c = nv_rt_alloc(&dev, bytes);
    if (!d_a || !d_b || !d_c) {
        fprintf(stderr, "tnv_rt: alloc failed\n");
        nv_rt_unload(&dev, &kern);
        nv_rt_shut(&dev);
        return 1;
    }

    /* ---- H2D copies ---- */
    rc  = nv_rt_h2d(&dev, d_a, h_a, bytes);
    rc |= nv_rt_h2d(&dev, d_b, h_b, bytes);
    if (rc) {
        fprintf(stderr, "tnv_rt: H2D failed\n");
        goto cleanup;
    }

    /* ---- Launch ---- */
    int n = N_ELEMS;
    unsigned grid = ((unsigned)N_ELEMS + BLK_SZ - 1) / BLK_SZ;

    /* CUDA Driver API: kernel params are pointers-to-values.
     * For device pointers, that means pointer-to-CUdevptr.
     * For scalars, pointer-to-scalar. Simple as. */
    void *args[] = { &d_a, &d_b, &d_c, &n };

    printf("tnv_rt: launching vectorAdd <<< %u, %d >>> (%d elements)\n",
           grid, BLK_SZ, N_ELEMS);

    rc = nv_rt_launch(&dev, &kern, grid, 1, 1, BLK_SZ, 1, 1, 0, args);
    if (rc) {
        fprintf(stderr, "tnv_rt: launch failed\n");
        goto cleanup;
    }

    /* ---- Sync ---- */
    rc = nv_rt_sync(&dev);
    if (rc) {
        fprintf(stderr, "tnv_rt: sync failed\n");
        goto cleanup;
    }

    /* ---- D2H copy ---- */
    rc = nv_rt_d2h(&dev, h_c, d_c, bytes);
    if (rc) {
        fprintf(stderr, "tnv_rt: D2H failed\n");
        goto cleanup;
    }

    /* ---- Verify ---- */
    errs = 0;
    for (int i = 0; i < N_ELEMS; i++) {
        float expect = (float)i + (float)(i * 2);
        if (fabsf(h_c[i] - expect) > 1e-5f) {
            if (errs < 10)
                fprintf(stderr, "  MISMATCH [%d]: got %f, expected %f\n",
                        i, (double)h_c[i], (double)expect);
            errs++;
        }
    }

    if (errs == 0) {
        printf("tnv_rt: PASS — %d elements verified\n", N_ELEMS);
    } else {
        printf("tnv_rt: FAIL — %d / %d mismatches\n", errs, N_ELEMS);
    }

cleanup:
    nv_rt_free(&dev, d_a);
    nv_rt_free(&dev, d_b);
    nv_rt_free(&dev, d_c);
    nv_rt_unload(&dev, &kern);
    nv_rt_shut(&dev);
    return errs ? 1 : 0;
}
