/* metal_be.c -- Apple Metal as a be_desc_t. Stub-level today. */

#include "backend.h"
#include "backend_cfg.h"
#include "metal.h"
#include "barracuda.h"
#include <stdlib.h>

static int metal_on(const be_cfg_t *cfg)
{
    return cfg->mode_metal;
}

static int metal_isel(const struct bir_module *M,
                      const be_cfg_t *cfg,
                      void **out_mmod)
{
    (void)cfg;
    metal_module_t *mm = (metal_module_t *)calloc(1, sizeof(metal_module_t));
    if (mm == NULL) return BE_ENOMEM;
    int rc = metal_compile((const bir_module_t *)M, mm);
    if (rc != BC_OK) { free(mm); return BE_EISEL; }
    *out_mmod = mm;
    return BE_OK;
}

static int metal_emit_op(const void *mmod, const be_cfg_t *cfg,
                         const char *out_path)
{
    (void)cfg;
    const char *p = out_path ? out_path : "a.metal";
    int rc = metal_emit_msl((metal_module_t *)mmod, p);
    return (rc == BC_OK) ? BE_OK : BE_EEMIT;
}

static void metal_free(void *mmod) { free(mmod); }

const be_desc_t be_metal = {
    .name    = "metal",
    .triple  = NULL,
    .feats   = BE_F_SIMT | BE_F_SHARED | BE_F_BARRIER | BE_F_TRANSC
             | BE_F_F16 | BE_F_F64,
    .is_on   = metal_on,
    .isel    = metal_isel,
    .sched   = NULL,
    .regalc  = NULL,
    .verify  = NULL,
    .emit    = metal_emit_op,
    .mfree   = metal_free
};
