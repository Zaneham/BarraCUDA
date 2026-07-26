/* lf_gpu.c — LFortran GPU offload ABI over the Booth NVIDIA launcher.
 *
 * The ABI is void-returning with no error channel, so failure is fatal and
 * loud: a silent bad copy hands Fortran an array of garbage, which is worse.
 * Kernels are PTX on disk, so we resolve the path at load time rather than
 * lean on the static registrar the CUDA runtime links in. */

#include "lf_gpu.h"
#include "nv_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Limits ---- */

#define LF_MAX_ARGS  32u    /* matches the isel parameter cap */
#define LF_MAX_KERN   8u
#define LF_SCAL_MAX  16u    /* widest scalar we pass by value */
#define LF_PATH_MAX 512u

/* ---- State ---- */

/* One device, fixed kernel pool. A do-concurrent program launches a handful
 * of kernels, so there is nothing here worth a malloc. */

struct lfortran_gpu_ctx {
    nv_dev_t dev;
    int      live;
};

typedef struct {
    void    *hptr;                    /* host side, for the copy back */
    CUdevptr dptr;
    size_t   size;
    uint8_t  scal[LF_SCAL_MAX];
    int      is_buf;
    int      used;
} lf_arg_t;

struct lfortran_gpu_kernel {
    nv_kern_t k;
    lf_arg_t  args[LF_MAX_ARGS];
    int       nargs;
    int       used;
    lfortran_gpu_ctx *ctx;
};

static lfortran_gpu_ctx    g_ctx;
static lfortran_gpu_kernel g_kern[LF_MAX_KERN];

/* ---- Helpers ---- */

static void lf_die(const char *what, int rc)
{
    fprintf(stderr, "lf_gpu: %s failed (rc=%d)\n", what, rc);
    exit(1);
}

/* Find the PTX: explicit source, then $BOOTH_PTX, then <kernel>.ptx. */
static void lf_ptx(const char *source, const char *entry,
                   char *out, size_t cap)
{
    const char *env;

    if (source != NULL && source[0] != '\0') {
        (void)snprintf(out, cap, "%s", source);
        return;
    }
    env = getenv("BOOTH_PTX");
    if (env != NULL && env[0] != '\0') {
        (void)snprintf(out, cap, "%s", env);
        return;
    }
    (void)snprintf(out, cap, "%s.ptx", entry);
}

/* ---- Lifecycle ---- */

lfortran_gpu_ctx *lfortran_gpu_init(void)
{
    int rc;

    if (g_ctx.live) return &g_ctx;

    rc = nv_rt_init(&g_ctx.dev);
    if (rc != NV_RT_OK) lf_die("nv_rt_init", rc);

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
    nv_rt_shut(&ctx->dev);
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

    lf_ptx(source, entry_point, path, sizeof path);

    memset(&g_kern[i], 0, sizeof g_kern[i]);
    rc = nv_rt_load(&ctx->dev, path, entry_point, &g_kern[i].k);
    if (rc != NV_RT_OK) {
        fprintf(stderr, "lf_gpu: could not load '%s' from %s\n",
                entry_point, path);
        lf_die("nv_rt_load", rc);
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
        if (k->args[j].is_buf && k->args[j].dptr != 0u) {
            nv_rt_free(&k->ctx->dev, k->args[j].dptr);
            k->args[j].dptr = 0u;
        }
    }
    nv_rt_unload(&k->ctx->dev, &k->k);
    k->used  = 0;
    k->nargs = 0;
}

/* ---- Arguments ---- */

/* Copy in now, out after the launch: Fortran reads results from the same
 * host array it handed us. */
void lfortran_gpu_set_buffer_arg(lfortran_gpu_kernel *k, int idx,
    void *ptr, size_t size)
{
    lf_arg_t *a;
    int       rc;

    if (k == NULL || idx < 0 || (uint32_t)idx >= LF_MAX_ARGS) return;

    a = &k->args[idx];
    if (a->is_buf && a->dptr != 0u) nv_rt_free(&k->ctx->dev, a->dptr);

    a->dptr = nv_rt_alloc(&k->ctx->dev, size);
    if (a->dptr == 0u) lf_die("nv_rt_alloc", (int)size);

    rc = nv_rt_h2d(&k->ctx->dev, a->dptr, ptr, size);
    if (rc != NV_RT_OK) lf_die("nv_rt_h2d", rc);

    a->hptr   = ptr;
    a->size   = size;
    a->is_buf = 1;
    a->used   = 1;
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
    a->size   = size;
    a->is_buf = 0;
    a->used   = 1;
    if (idx >= k->nargs) k->nargs = idx + 1;
}

/* ---- Launch ---- */

void lfortran_gpu_launch(lfortran_gpu_ctx *ctx, lfortran_gpu_kernel *k,
    int grid[3], int block[3])
{
    void *argv[LF_MAX_ARGS];
    int   j, n = 0;
    int   rc;

    if (ctx == NULL || k == NULL || !k->used) return;
    if (grid == NULL || block == NULL) lf_die("launch dims", 0);
    if (grid[0] < 1 || grid[1] < 1 || grid[2] < 1 ||
        block[0] < 1 || block[1] < 1 || block[2] < 1) {
        lf_die("launch dims non-positive", grid[0]);
    }

    /* Driver API wants a pointer to each arg, so buffers pass &dptr. */
    for (j = 0; j < k->nargs; j++) {
        if (!k->args[j].used) lf_die("argument gap", j);
        argv[n++] = k->args[j].is_buf ? (void *)&k->args[j].dptr
                                      : (void *)k->args[j].scal;
    }

    rc = nv_rt_launch(&ctx->dev, &k->k,
                      (uint32_t)grid[0],  (uint32_t)grid[1],  (uint32_t)grid[2],
                      (uint32_t)block[0], (uint32_t)block[1], (uint32_t)block[2],
                      0u, argv);
    if (rc != NV_RT_OK) lf_die("nv_rt_launch", rc);

    rc = nv_rt_sync(&ctx->dev);
    if (rc != NV_RT_OK) lf_die("nv_rt_sync", rc);

    for (j = 0; j < k->nargs; j++) {
        if (!k->args[j].is_buf || k->args[j].dptr == 0u) continue;
        rc = nv_rt_d2h(&ctx->dev, k->args[j].hptr,
                       k->args[j].dptr, k->args[j].size);
        if (rc != NV_RT_OK) lf_die("nv_rt_d2h", rc);
    }
}

void lfortran_gpu_sync(lfortran_gpu_ctx *ctx)
{
    if (ctx == NULL || !ctx->live) return;
    (void)nv_rt_sync(&ctx->dev);
}
