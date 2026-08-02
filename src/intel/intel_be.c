/* intel_be.c -- Intel Arc SPIR-V as a be_desc_t.
 * Even more stub than Metal; intel_compile prints an honest "not yet
 * a working compiler" so nobody thinks they got a real .spv. */

#include "backend.h"
#include "backend_cfg.h"
#include "intel.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>

static int in_on(const be_cfg_t *cfg) { return cfg->mode_intel; }

static int in_isel(const struct bir_module *M, const be_cfg_t *cfg,
                   void **out_mmod)
{
    intel_module_t *im = calloc(1, sizeof(*im));
    if (im == NULL) return BE_ENOMEM;
    if (intel_compile((const bir_module_t *)M, im, cfg->intel_target) != BC_OK) {
        fprintf(stderr, "error: Intel SPIR-V backend not yet a working compiler\n");
        free(im);
        return BE_EISEL;
    }
    *out_mmod = im;
    return BE_OK;
}

static int in_emit(const void *mmod, const be_cfg_t *cfg, const char *out)
{
    (void)cfg;
    return intel_emit_spirv((const intel_module_t *)mmod,
                            out ? out : "a.spv") == BC_OK
         ? BE_OK : BE_EEMIT;
}

const be_desc_t be_intel = {
    .name    = "intel-spirv",
    .triple  = NULL,
    .feats   = BE_F_SIMT | BE_F_SHARED | BE_F_BARRIER,
    .is_on   = in_on,
    .isel    = in_isel,
    .emit    = in_emit,
    .mfree   = free
};
