/* intel_be.c -- Intel Arc SPIR-V as a be_desc_t. Stub-level today. */

#include "backend.h"
#include "backend_cfg.h"
#include "intel.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>

static int intel_on(const be_cfg_t *cfg)
{
    return cfg->mode_intel;
}

static int intel_isel(const struct bir_module *M,
                      const be_cfg_t *cfg,
                      void **out_mmod)
{
    intel_module_t *im = (intel_module_t *)calloc(1, sizeof(intel_module_t));
    if (im == NULL) return BE_ENOMEM;
    int rc = intel_compile((const bir_module_t *)M, im, cfg->intel_target);
    if (rc != BC_OK) {
        fprintf(stderr,
            "error: Intel SPIR-V backend not yet a working compiler\n");
        free(im);
        return BE_EISEL;
    }
    *out_mmod = im;
    return BE_OK;
}

static int intel_emit_op(const void *mmod, const be_cfg_t *cfg,
                         const char *out_path)
{
    (void)cfg;
    const char *p = out_path ? out_path : "a.spv";
    int rc = intel_emit_spirv((const intel_module_t *)mmod, p);
    return (rc == BC_OK) ? BE_OK : BE_EEMIT;
}

static void intel_free(void *mmod) { free(mmod); }

const be_desc_t be_intel = {
    .name    = "intel-spirv",
    .triple  = NULL,
    .feats   = BE_F_SIMT | BE_F_SHARED | BE_F_BARRIER,
    .is_on   = intel_on,
    .isel    = intel_isel,
    .sched   = NULL,
    .regalc  = NULL,
    .verify  = NULL,
    .emit    = intel_emit_op,
    .mfree   = intel_free
};
