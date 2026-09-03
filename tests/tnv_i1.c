/* tnv_i1.c — does the i1 escape kernel load and give the right numbers
 *
 * Before the predicate materialisation fix the PTX for tests/i1esc.cu carried
 * `st.global.u32 [%rd], %p` and its relatives, so the driver JIT refused the
 * module and nv_rt_load failed outright. That makes this unambiguous: the load
 * either happens or it does not. The values then check the other half, because
 * C promises `(a==0)` is exactly 0 or 1 and `(a==0)+5` is 5 or 6, which a fix
 * that clamps everything to a predicate would get wrong.
 *
 * Wants a real card. Not a trunner test.
 *
 *   ./kath --nvidia-ptx tests/i1esc.cu -o i1esc.ptx
 *   ./tnv_i1 i1esc.ptx
 */

#include "booth/nv_rt.h"
#include <stdio.h>

#define NSLOT 9
#define ATBAS 100

static int host[NSLOT];

static int check(nv_dev_t *dev, nv_kern_t *kern, int a, int b)
{
    int want[NSLOT];
    int errs = 0;

    want[0] = (a == 0) || (b == 0);
    want[1] = (a == 0) && (b == 0);
    want[2] = (a == 0) + 5;
    want[3] = (b == 0) * 3;
    want[4] = (a == 0);
    want[5] = !a;
    want[6] = (int)((float)(a == 0) * 4.0f);
    want[7] = (a == b);
    want[8] = ATBAS + (b == 0);

    for (int i = 0; i < NSLOT; i++) host[i] = 0;
    host[8] = ATBAS;

    CUdevptr d_out = nv_rt_alloc(dev, sizeof host);
    if (!d_out) { fprintf(stderr, "  alloc failed\n"); return 1; }

    if (nv_rt_h2d(dev, d_out, host, sizeof host) != NV_RT_OK) {
        fprintf(stderr, "  H2D failed\n");
        nv_rt_free(dev, d_out);
        return 1;
    }

    void *args[3];
    args[0] = &d_out;
    args[1] = &a;
    args[2] = &b;

    if (nv_rt_launch(dev, kern, 1, 1, 1, 1, 1, 1, 0, args) != NV_RT_OK
     || nv_rt_sync(dev) != NV_RT_OK
     || nv_rt_d2h(dev, host, d_out, sizeof host) != NV_RT_OK) {
        fprintf(stderr, "  launch or readback failed\n");
        nv_rt_free(dev, d_out);
        return 1;
    }
    nv_rt_free(dev, d_out);

    printf("  a=%d b=%d ->", a, b);
    for (int i = 0; i < NSLOT; i++) printf(" %d", host[i]);
    printf("\n");

    for (int i = 0; i < NSLOT; i++) {
        if (host[i] != want[i]) {
            fprintf(stderr, "    slot %d: got %d, wanted %d\n",
                    i, host[i], want[i]);
            errs++;
        }
    }
    return errs;
}

int main(int argc, char **argv)
{
    const char *ptx = (argc > 1) ? argv[1] : "i1esc.ptx";
    nv_dev_t dev;
    nv_kern_t kern;
    int errs = 0;

    if (nv_rt_init(&dev) != NV_RT_OK) {
        fprintf(stderr, "tnv_i1: no device\n");
        return 2;
    }
    printf("tnv_i1: %s, sm_%d%d\n", dev.dev_name, dev.sm_major, dev.sm_minor);

    if (nv_rt_load(&dev, ptx, "i1esc", &kern) != NV_RT_OK) {
        fprintf(stderr, "tnv_i1: JIT refused %s\n", ptx);
        nv_rt_shut(&dev);
        return 1;
    }
    printf("tnv_i1: %s loaded\n", ptx);

    errs += check(&dev, &kern, 0, 0);
    errs += check(&dev, &kern, 0, 1);
    errs += check(&dev, &kern, 1, 0);
    errs += check(&dev, &kern, 1, 1);

    printf("tnv_i1: %s (%d mismatches)\n", errs ? "FAIL" : "PASS", errs);

    nv_rt_unload(&dev, &kern);
    nv_rt_shut(&dev);
    return errs ? 1 : 0;
}
