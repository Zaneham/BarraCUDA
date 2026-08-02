/* amd_be.c -- AMDGPU as a be_desc_t. Full pipeline: isel, sched,
 * regalloc, verify twice, emit. What a mature backend looks like at
 * the descriptor layer; the actual work is in the rest of src/amdgpu/. */

#include "backend.h"
#include "backend_cfg.h"
#include "amdgpu.h"
#include "sched.h"
#include "verify.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int amd_on(const be_cfg_t *cfg)
{
    return cfg->mode_amdgpu || cfg->mode_amdgpu_bin;
}

static int amd_isel(const struct bir_module *M, const be_cfg_t *cfg,
                    void **out_mmod)
{
    amd_module_t *amd = calloc(1, sizeof(*amd));
    if (amd == NULL) return BE_ENOMEM;
    amd->target    = cfg->amd_target;
    amd->elf_mach  = cfg->amd_elfm;
    amd->snap_mode = (uint8_t)cfg->snap_mode;
    snprintf(amd->chip_name, sizeof(amd->chip_name), "%s",
             cfg->amd_chip ? cfg->amd_chip : "");
    if (amdgpu_compile((const bir_module_t *)M, amd) != BC_OK) {
        free(amd);
        return BE_EISEL;
    }
    *out_mmod = amd;
    return BE_OK;
}

static int amd_sched(void *mmod)   { amdgpu_sched(mmod);    return BE_OK; }
static int amd_regalc(void *mmod)  { amdgpu_regalloc(mmod); return BE_OK; }

static int amd_verify(const void *mmod, int phase)
{
    int p = (phase == BE_VFY_ISEL) ? VFY_ISEL : VFY_RA;
    vfy_res_t v = bc_vfy(mmod, p);
    if (v.errs == 0) return BE_OK;
    fprintf(stderr, "verify: %u error(s) after %s\n",
            v.errs, p == VFY_ISEL ? "isel" : "regalloc");
    return BE_EVFY;
}

static int amd_emit(const void *mmod, const be_cfg_t *cfg, const char *out)
{
    /* --amdgpu-bin writes a .hsaco; bare --amdgpu dumps asm to stdout. */
    if (cfg->mode_amdgpu_bin) {
        int rc = amdgpu_emit_elf((amd_module_t *)mmod, out ? out : "a.hsaco");
        return rc == 0 ? BE_OK : BE_EEMIT;
    }
    amdgpu_emit_asm(mmod, stdout);
    return BE_OK;
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
    .mfree   = free
};
