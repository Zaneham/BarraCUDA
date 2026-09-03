/* gpu_mma.c -- run the 16x16x16 f16 matrix multiply on a real NVIDIA GPU.
 * PTX is JITed by the driver, so no CUDA SDK is needed.
 *
 *   kath --nvidia-ptx tests/mma16.cu -o mma16.ptx
 *   gcc tests/gpu_mma.c runtime/host/cuda/nv_rt.c -Iruntime/include -o gpu_mma
 *   ./gpu_mma mma16.ptx
 */
#include "booth/nv_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MM 16
#define NN 16
#define KK 16
#define WARP 32

/* Padded row strides, so a hard-coded 16 in the address maths shows up. */
#define LDA (KK + 3)
#define LDB (NN + 2)
#define LDD (NN + 5)

/* Asymmetric in both indices: a layout that transposed a fragment, or
 * swapped the two N halves, would not survive this. */
#define AV(i, k) ((float)((((i) * 3 + (k) * 5) % 7) - 3))
#define BV(k, j) ((float)((((k) * 2 + (j) * 3) % 5) - 2))

/* bf16 is just the top half of the f32; exact for the small integers used
 * here, so there is nothing to round. */
static unsigned short f2b(float f)
{
    union { float f; unsigned int u; } p;
    p.f = f;
    return (unsigned short)(p.u >> 16);
}

/* Only small integers go through here, so the exponent path is the whole
 * story and there is nothing to round. */
static unsigned short f2h(float f)
{
    union { float f; unsigned int u; } p;
    p.f = f;
    unsigned int s = (p.u >> 16) & 0x8000u;
    int e = (int)((p.u >> 23) & 0xFFu) - 127 + 15;
    unsigned int m = p.u & 0x7FFFFFu;
    if (p.f == 0.0f) return (unsigned short)s;
    if (e <= 0 || e >= 31) return (unsigned short)(s | 0x7C00u);
    return (unsigned short)(s | ((unsigned int)e << 10) | (m >> 13));
}

struct shape { const char *kern; int kk; int bf; };

static int oneshape(nv_dev_t *dev, const char *ptx, const struct shape *sp)
{
    nv_kern_t k;
    if (nv_rt_load(dev, ptx, sp->kern, &k) != NV_RT_OK) {
        fprintf(stderr, "gpu_mma: load %s failed\n", sp->kern);
        return 1;
    }
    static unsigned short ha[MM * LDA], hb[KK * LDB];
    static float hd[MM * LDD], ref[MM * LDD];
    memset(ha, 0, sizeof ha);
    memset(hb, 0, sizeof hb);
    for (int i = 0; i < MM; i++)
        for (int q = 0; q < sp->kk; q++)
            ha[i * LDA + q] = sp->bf ? f2b(AV(i, q)) : f2h(AV(i, q));
    for (int q = 0; q < sp->kk; q++)
        for (int j = 0; j < NN; j++)
            hb[q * LDB + j] = sp->bf ? f2b(BV(q, j)) : f2h(BV(q, j));
    for (int i = 0; i < MM * LDD; i++)
        hd[i] = (float)(i * 2 + 1);
    memcpy(ref, hd, sizeof ref);
    for (int i = 0; i < MM; i++)
        for (int j = 0; j < NN; j++) {
            float acc = hd[i * LDD + j];
            for (int q = 0; q < sp->kk; q++)
                acc += AV(i, q) * BV(q, j);
            ref[i * LDD + j] = acc;
        }

    CUdevptr da = nv_rt_alloc(dev, sizeof ha);
    CUdevptr db = nv_rt_alloc(dev, sizeof hb);
    CUdevptr dd = nv_rt_alloc(dev, sizeof hd);
    nv_rt_h2d(dev, da, ha, sizeof ha);
    nv_rt_h2d(dev, db, hb, sizeof hb);
    nv_rt_h2d(dev, dd, hd, sizeof hd);
    int lda = LDA, ldb = LDB, ldd = LDD;
    void *args[6] = { &da, &db, &dd, &lda, &ldb, &ldd };
    int rc = nv_rt_launch(dev, &k, 1, 1, 1, WARP, 1, 1, 0, args);
    nv_rt_sync(dev);
    nv_rt_d2h(dev, hd, dd, sizeof hd);
    nv_rt_free(dev, da); nv_rt_free(dev, db); nv_rt_free(dev, dd);
    nv_rt_unload(dev, &k);
    if (rc != NV_RT_OK) { fprintf(stderr, "gpu_mma: launch %s failed\n", sp->kern); return 1; }

    int bad = 0;
    for (int i = 0; i < MM * LDD; i++) {
        if (hd[i] != ref[i]) {
            if (bad < 4)
                fprintf(stderr, "  %s [%d,%d] got %.1f want %.1f\n", sp->kern,
                        i / LDD, i % LDD, (double)hd[i], (double)ref[i]);
            bad++;
        }
    }
    printf("  %-7s k=%2d %-4s  %s (%d elements)\n", sp->kern, sp->kk,
           sp->bf ? "bf16" : "f16", bad ? "FAIL" : "PASS", MM * LDD);
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "mma16.ptx";
    static const struct shape shapes[] = {
        { "mma16",  16, 0 }, { "mmab16", 16, 1 },
        { "mma8",    8, 0 }, { "mmab8",   8, 1 },
    };
    nv_dev_t dev;
    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "gpu_mma: no CUDA driver, skipping\n");
        return 77;
    }
    int bad = 0;
    for (unsigned i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        bad += oneshape(&dev, ptx, &shapes[i]);
    nv_rt_shut(&dev);
    printf("gpu_mma: %s\n", bad ? "FAIL" : "PASS - every shape matches the host reference");
    return bad ? 1 : 0;
}
