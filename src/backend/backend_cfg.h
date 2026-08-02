/* backend_cfg.h -- the field bag every backend rummages through.
 * Yes it pulls in amdgpu.h and intel.h for two enums each; the
 * cleaner extraction can be another day's PR. */

#ifndef BOOTH_BE_CFG_H
#define BOOTH_BE_CFG_H

#include "amdgpu.h"
#include "intel.h"

typedef struct be_cfg {
    int             no_mem2reg, no_cfold, no_dce, no_sched, no_sroa;
    int             mode_ir, mode_tdf, mode_tdf_fission;
    int             mode_amdgpu, mode_amdgpu_bin;
    int             mode_tensix, mode_nvidia, nv_bkhit;
    int             mode_metal, mode_intel, mode_rv_elf, mode_cpu, mode_rv64;
    amd_target_t    amd_target;
    uint32_t        amd_elfm;
    const char     *amd_chip;
    int             snap_mode;
    intel_target_t  intel_target;
    const char     *output_file;
} be_cfg_t;

#endif
