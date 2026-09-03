/* tnv_bstr.c — does a byte array survive a round trip through the card
 *
 * The GEP stride for a one-byte element is one byte, and the access has to be
 * one byte with it. On master PTX strode by one and loaded four, so element i
 * read elements i..i+3 and a store scribbled over the three that followed.
 * Sixteen distinct values in sixteen adjacent bytes is the smallest thing that
 * catches it: any over-wide access loses the neighbours.
 *
 * Wants a real card. Not a trunner test.
 *
 *   ./kath --nvidia-ptx tests/bstr.cu -o bstr.ptx
 *   ./tnv_bstr bstr.ptx
 */

#include "booth/nv_rt.h"
#include <stdio.h>

#define N 16

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "bstr.ptx";
    nv_dev_t dev;
    nv_kern_t kern;
    char src[N], dst[N];
    int errs = 0;

    for (int i = 0; i < N; i++) { src[i] = (char)(i * 7 + 1); dst[i] = 0; }

    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "tnv_bstr: no device\n");
        return 2;
    }
    printf("tnv_bstr: %s, sm_%d%d\n", dev.dev_name, dev.sm_major, dev.sm_minor);

    if (nv_rt_load(&dev, ptx, "ccopy", &kern) != NV_RT_OK) {
        fprintf(stderr, "tnv_bstr: JIT refused %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }

    CUdevptr d_in = nv_rt_alloc(&dev, sizeof src);
    CUdevptr d_out = nv_rt_alloc(&dev, sizeof dst);
    if (!d_in || !d_out) {
        fprintf(stderr, "tnv_bstr: alloc failed\n");
        nv_rt_shut(&dev);
        return 1;
    }

    void *args[2];
    args[0] = &d_out;
    args[1] = &d_in;

    if (nv_rt_h2d(&dev, d_in, src, sizeof src) != NV_RT_OK
     || nv_rt_h2d(&dev, d_out, dst, sizeof dst) != NV_RT_OK
     || nv_rt_launch(&dev, &kern, 1, 1, 1, N, 1, 1, 0, args) != NV_RT_OK
     || nv_rt_sync(&dev) != NV_RT_OK
     || nv_rt_d2h(&dev, dst, d_out, sizeof dst) != NV_RT_OK) {
        fprintf(stderr, "tnv_bstr: launch or readback failed\n");
        nv_rt_free(&dev, d_in); nv_rt_free(&dev, d_out);
        nv_rt_shut(&dev);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        printf("  [%2d] got %4d want %4d%s\n", i, dst[i], src[i],
               dst[i] == src[i] ? "" : "   <-- ");
        if (dst[i] != src[i]) errs++;
    }
    printf("tnv_bstr: %s (%d mismatches)\n", errs ? "FAIL" : "PASS", errs);

    nv_rt_free(&dev, d_in);
    nv_rt_free(&dev, d_out);
    nv_rt_unload(&dev, &kern);
    nv_rt_shut(&dev);
    return errs ? 1 : 0;
}
