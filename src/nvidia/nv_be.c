/* nv_be.c -- NVIDIA PTX as a be_desc_t. */

#include "backend.h"
#include "backend_cfg.h"
#include "nvidia.h"
#include "barracuda.h"
#include <stdlib.h>

static int nv_on(const be_cfg_t *cfg)
{
    return cfg->mode_nvidia;
}

static int nv_isel(const struct bir_module *M,
                   const be_cfg_t *cfg,
                   void **out_mmod)
{
    nv_module_t *nvm = (nv_module_t *)calloc(1, sizeof(nv_module_t));
    if (nvm == NULL) return BE_ENOMEM;
    int rc = nv_compile((const bir_module_t *)M, nvm);
    if (rc != BC_OK) { free(nvm); return BE_EISEL; }
    nvm->bkhit = (uint8_t)cfg->nv_bkhit;
    *out_mmod = nvm;
    return BE_OK;
}

static int nv_emit_op(const void *mmod, const be_cfg_t *cfg,
                      const char *out_path)
{
    const char *p = out_path ? out_path : "a.ptx";
    (void)cfg;
    int rc = nv_emit_ptx((nv_module_t *)mmod, p);
    return (rc == BC_OK) ? BE_OK : BE_EEMIT;
}

static void nv_free(void *mmod) { free(mmod); }

const be_desc_t be_ptx = {
    .name    = "nvptx",
    .triple  = "nvptx64--",
    .feats   = BE_F_SIMT | BE_F_ATOMIC | BE_F_SHARED | BE_F_WARP
             | BE_F_BARRIER | BE_F_DIV | BE_F_SCRATCH | BE_F_TRANSC
             | BE_F_F16 | BE_F_F64 | BE_F_BF16,
    .is_on   = nv_on,
    .isel    = nv_isel,
    .sched   = NULL,
    .regalc  = NULL,
    .verify  = NULL,
    .emit    = nv_emit_op,
    .mfree   = nv_free
};
