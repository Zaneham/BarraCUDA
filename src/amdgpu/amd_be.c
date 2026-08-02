/* amd_be.c -- AMDGPU as a be_desc_t.
 *
 * Thin wrappers around the existing amdgpu_* API. This file is the
 * whole story of "AMDGPU as a Booth backend"; the rest of src/amdgpu/
 * is the implementation. */

#include "backend.h"
#include "backend_cfg.h"
#include "amdgpu.h"
#include "sched.h"
#include "verify.h"
#include "barracuda.h"     /* BC_OK, BC_ERR_* */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int amd_on(const be_cfg_t *cfg)
{
    return cfg->mode_amdgpu || cfg->mode_amdgpu_bin;
}

static int amd_isel(const struct bir_module *M,
                    const be_cfg_t *cfg,
                    void **out_mmod)
{
    amd_module_t *amd = (amd_module_t *)calloc(1, sizeof(amd_module_t));
    if (amd == NULL) return BE_ENOMEM;

    amd->target    = cfg->amd_target;
    amd->elf_mach  = cfg->amd_elfm;
    amd->snap_mode = (uint8_t)cfg->snap_mode;
    snprintf(amd->chip_name, sizeof(amd->chip_name), "%s",
             cfg->amd_chip ? cfg->amd_chip : "");

    int rc = amdgpu_compile((const bir_module_t *)M, amd);
    if (rc != BC_OK) {
        free(amd);
        *out_mmod = NULL;
        return BE_EISEL;
    }
    *out_mmod = amd;
    return BE_OK;
}

static int amd_sched(void *mmod)
{
    amdgpu_sched((amd_module_t *)mmod);
    return BE_OK;
}

static int amd_regalc(void *mmod)
{
    amdgpu_regalloc((amd_module_t *)mmod);
    return BE_OK;
}

static int amd_verify(const void *mmod, int phase)
{
    int p = (phase == BE_VFY_ISEL) ? VFY_ISEL : VFY_RA;
    vfy_res_t v = bc_vfy((const amd_module_t *)mmod, p);
    if (v.errs > 0) {
        fprintf(stderr, "verify: %u error(s) after %s\n",
                v.errs, (p == VFY_ISEL) ? "isel" : "regalloc");
        return BE_EVFY;
    }
    return BE_OK;
}

static int amd_emit(const void *mmod, const be_cfg_t *cfg,
                    const char *out_path)
{
    amd_module_t *amd = (amd_module_t *)mmod;
    if (cfg->mode_amdgpu_bin) {
        const char *p = out_path ? out_path : "a.hsaco";
        int rc = amdgpu_emit_elf(amd, p);
        return (rc == 0) ? BE_OK : BE_EEMIT;
    }
    amdgpu_emit_asm(amd, stdout);
    return BE_OK;
}

static void amd_free(void *mmod)
{
    free(mmod);
}

const be_desc_t be_amd = {
    .name    = "amdgpu",
    .triple  = "amdgcn--",
    .feats   = BE_F_SIMT | BE_F_ATOMIC | BE_F_SHARED | BE_F_WARP
             | BE_F_BARRIER | BE_F_DIV | BE_F_SCRATCH | BE_F_TRANSC
             | BE_F_F16 | BE_F_F64 | BE_F_MFMA,
    .is_on   = amd_on,
    .isel    = amd_isel,
    .sched   = amd_sched,
    .regalc  = amd_regalc,
    .verify  = amd_verify,
    .emit    = amd_emit,
    .mfree   = amd_free
};
