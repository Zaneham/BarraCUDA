/* metal_be.c -- Apple Metal as a be_desc_t.
 * Stub-level until someone with an M-series and an afternoon hardware
 * validates it. Who tf makes Fortran run on a Mac? Us, eventually. */

#include "backend.h"
#include "backend_cfg.h"
#include "metal.h"
#include "barracuda.h"
#include <stdlib.h>

static int mt_on(const be_cfg_t *cfg) { return cfg->mode_metal; }

static int mt_isel(const struct bir_module *M, const be_cfg_t *cfg,
                   void **out_mmod)
{
    (void)cfg;
    metal_module_t *mm = calloc(1, sizeof(*mm));
    if (mm == NULL) return BE_ENOMEM;
    if (metal_compile((const bir_module_t *)M, mm) != BC_OK) {
        free(mm);
        return BE_EISEL;
    }
    *out_mmod = mm;
    return BE_OK;
}

static int mt_emit(const void *mmod, const be_cfg_t *cfg, const char *out)
{
    (void)cfg;
    return metal_emit_msl((metal_module_t *)mmod, out ? out : "a.metal") == BC_OK
         ? BE_OK : BE_EEMIT;
}

const be_desc_t be_metal = {
    .name    = "metal",
    .triple  = NULL,
    .feats   = BE_F_SIMT | BE_F_SHARED | BE_F_BARRIER | BE_F_TRANSC
             | BE_F_F16 | BE_F_F64,
    .is_on   = mt_on,
    .isel    = mt_isel,
    .emit    = mt_emit,
    .mfree   = free
};
