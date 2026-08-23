/* lf_gpu_hsa.c — LFortran GPU offload ABI over the Booth HSA launcher.
 *
 * The AMD sibling of lf_gpu.c. Same eight functions, same contract, but the
 * kernel is a .hsaco run through bc_runtime instead of PTX through nv_rt, so an
 * LFortran-compiled program can offload to an AMD GPU. The build links one or
 * the other into the Fortran executable depending on the target; they can't
 * coexist, they define the same symbols.
 *
 * HSA takes a single packed kernarg blob rather than an array of arg pointers,
 * so launch assembles it here: each parameter in an 8-byte slot, then the
 * hidden block_count/group_size the AMD isel reads for blockDim/gridDim. The
 * layout matches the one the RDNA emulator checks in tests/numeric.
 *
 * Linux/ROCm only. Smoke-tested on an MI300X (CDNA): a do-concurrent kernel
 * offloads through here and comes back with the right answer.
 */

#include "booth/lf_gpu.h"
#include "booth/bc_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Limits ---- */

#define LF_MAX_ARGS  32u
#define LF_MAX_KERN   8u
#define LF_SCAL_MAX  16u
#define LF_PATH_MAX 512u
#define LF_KARG_MAX (LF_MAX_ARGS * 8u + 32u)   /* params + hidden args */

/* ---- State ---- */

struct lfortran_gpu_ctx {
    bc_device_t dev;
    int         live;
};

typedef struct {
    void   *hptr;                 /* host array, for the copy back */
    void   *dptr;                 /* device allocation */
    size_t  size;
    uint8_t scal[LF_SCAL_MAX];
    size_t  scal_sz;
    int     is_buf;
    int     used;
} lf_arg_t;

struct lfortran_gpu_kernel {
    bc_kernel_t k;
    lf_arg_t    args[LF_MAX_ARGS];
    int         nargs;
    int         used;
    lfortran_gpu_ctx *ctx;
};

static lfortran_gpu_ctx    g_ctx;
static lfortran_gpu_kernel g_kern[LF_MAX_KERN];

/* ---- Helpers ---- */

static void lf_die(const char *what, int rc)
{
    fprintf(stderr, "lf_gpu_hsa: %s failed (rc=%d)\n", what, rc);
    exit(1);
}

/* Where to find the hsaco: explicit source, then $BOOTH_HSACO, then <name>. */
static void lf_hsaco(const char *source, const char *entry,
                     char *out, size_t cap)
{
    const char *env;

    if (source != NULL && source[0] != '\0') {
        (void)snprintf(out, cap, "%s", source);
        return;
    }
    env = getenv("BOOTH_HSACO");
    if (env != NULL && env[0] != '\0') {
        (void)snprintf(out, cap, "%s", env);
        return;
    }
    (void)snprintf(out, cap, "%s.hsaco", entry);
}

/* ---- Lifecycle ---- */

lfortran_gpu_ctx *lfortran_gpu_init(void)
{
    int rc;

    if (g_ctx.live) return &g_ctx;

    rc = bc_device_init(&g_ctx.dev);
    if (rc != BC_RT_OK) lf_die("bc_device_init", rc);

    g_ctx.live = 1;
    return &g_ctx;
}

void lfortran_gpu_shutdown(lfortran_gpu_ctx *ctx)
{
    uint32_t i;

    if (ctx == NULL || !ctx->live) return;

    for (i = 0u; i < LF_MAX_KERN; i++) {
        if (g_kern[i].used) lfortran_gpu_release_kernel(&g_kern[i]);
    }
    bc_device_shutdown(&ctx->dev);
    ctx->live = 0;
}

/* ---- Kernels ---- */

lfortran_gpu_kernel *lfortran_gpu_load_kernel(
    lfortran_gpu_ctx *ctx, const char *source, const char *entry_point)
{
    char     path[LF_PATH_MAX];
    uint32_t i;
    int      rc;

    if (ctx == NULL || entry_point == NULL) lf_die("load_kernel args", 0);

    for (i = 0u; i < LF_MAX_KERN; i++) {
        if (!g_kern[i].used) break;
    }
    if (i == LF_MAX_KERN) lf_die("kernel pool exhausted", (int)LF_MAX_KERN);

    lf_hsaco(source, entry_point, path, sizeof path);

    memset(&g_kern[i], 0, sizeof g_kern[i]);
    rc = bc_load_kernel(&ctx->dev, path, entry_point, &g_kern[i].k);
    if (rc != BC_RT_OK) {
        fprintf(stderr, "lf_gpu_hsa: could not load '%s' from %s\n",
                entry_point, path);
        lf_die("bc_load_kernel", rc);
    }

    g_kern[i].used = 1;
    g_kern[i].ctx  = ctx;
    return &g_kern[i];
}

void lfortran_gpu_release_kernel(lfortran_gpu_kernel *k)
{
    int j;

    if (k == NULL || !k->used) return;

    for (j = 0; j < k->nargs; j++) {
        if (k->args[j].is_buf && k->args[j].dptr != NULL) {
            bc_free(&k->ctx->dev, k->args[j].dptr);
            k->args[j].dptr = NULL;
        }
    }
    bc_unload_kernel(&k->ctx->dev, &k->k);
    k->used  = 0;
    k->nargs = 0;
}

/* ---- Arguments ---- */

void lfortran_gpu_set_buffer_arg(lfortran_gpu_kernel *k, int idx,
    void *ptr, size_t size)
{
    lf_arg_t *a;
    int       rc;

    if (k == NULL || idx < 0 || (uint32_t)idx >= LF_MAX_ARGS) return;

    a = &k->args[idx];
    if (a->is_buf && a->dptr != NULL) bc_free(&k->ctx->dev, a->dptr);

    a->dptr = bc_alloc(&k->ctx->dev, size);
    if (a->dptr == NULL) lf_die("bc_alloc", (int)size);

    rc = bc_copy_h2d(&k->ctx->dev, a->dptr, ptr, size);
    if (rc != BC_RT_OK) lf_die("bc_copy_h2d", rc);

    a->hptr    = ptr;
    a->size    = size;
    a->is_buf  = 1;
    a->used    = 1;
    if (idx >= k->nargs) k->nargs = idx + 1;
}

void lfortran_gpu_set_scalar_arg(lfortran_gpu_kernel *k, int idx,
    const void *val, size_t size)
{
    lf_arg_t *a;

    if (k == NULL || idx < 0 || (uint32_t)idx >= LF_MAX_ARGS) return;
    if (val == NULL || size == 0u || size > LF_SCAL_MAX) {
        lf_die("scalar arg too wide", (int)size);
    }

    a = &k->args[idx];
    memcpy(a->scal, val, size);
    a->scal_sz = size;
    a->is_buf  = 0;
    a->used    = 1;
    if (idx >= k->nargs) k->nargs = idx + 1;
}

/* ---- Launch ---- */

void lfortran_gpu_launch(lfortran_gpu_ctx *ctx, lfortran_gpu_kernel *k,
    int grid[3], int block[3])
{
    uint8_t  karg[LF_KARG_MAX];
    uint32_t slot = 0u, hk;
    int      j, rc;

    if (ctx == NULL || k == NULL || !k->used) return;
    if (grid == NULL || block == NULL) lf_die("launch dims", 0);
    if (grid[0] < 1 || grid[1] < 1 || grid[2] < 1 ||
        block[0] < 1 || block[1] < 1 || block[2] < 1) {
        lf_die("launch dims non-positive", grid[0]);
    }

    memset(karg, 0, sizeof karg);

    /* One 8-byte slot per parameter: a buffer contributes its device pointer,
       a scalar its value in the low bytes. */
    for (j = 0; j < k->nargs; j++) {
        if (!k->args[j].used) lf_die("argument gap", j);
        if (slot + 8u > LF_KARG_MAX) lf_die("kernarg overflow", j);
        if (k->args[j].is_buf) {
            void *dp = k->args[j].dptr;
            memcpy(karg + slot, &dp, sizeof dp);
        } else {
            memcpy(karg + slot, k->args[j].scal, k->args[j].scal_sz);
        }
        slot += 8u;
    }

    /* Hidden kernarg the AMD isel reads for gridDim/blockDim: block_count as
       three u32, then group_size as three u16. grid = workgroups, block =
       work-items per group. */
    hk = slot;
    if (hk + 18u > LF_KARG_MAX) lf_die("hidden kernarg overflow", (int)hk);
    {
        uint32_t bc3[3] = { (uint32_t)grid[0], (uint32_t)grid[1], (uint32_t)grid[2] };
        uint16_t gs3[3] = { (uint16_t)block[0], (uint16_t)block[1], (uint16_t)block[2] };
        memcpy(karg + hk,       bc3, sizeof bc3);
        memcpy(karg + hk + 12u, gs3, sizeof gs3);
    }

    rc = bc_dispatch(&ctx->dev, &k->k,
                     (uint32_t)grid[0],  (uint32_t)grid[1],  (uint32_t)grid[2],
                     (uint32_t)block[0], (uint32_t)block[1], (uint32_t)block[2],
                     karg, hk + 18u);
    if (rc != BC_RT_OK) lf_die("bc_dispatch", rc);

    /* bc_dispatch is synchronous, so results are ready. Copy each output
       buffer back into the host array Fortran handed us. */
    for (j = 0; j < k->nargs; j++) {
        if (!k->args[j].is_buf || k->args[j].dptr == NULL) continue;
        rc = bc_copy_d2h(&ctx->dev, k->args[j].hptr,
                         k->args[j].dptr, k->args[j].size);
        if (rc != BC_RT_OK) lf_die("bc_copy_d2h", rc);
    }
}

void lfortran_gpu_sync(lfortran_gpu_ctx *ctx)
{
    /* bc_dispatch already blocks, so there is nothing outstanding to wait on. */
    (void)ctx;
}
