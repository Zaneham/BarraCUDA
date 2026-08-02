/* backend_cfg.h -- driver config shared with every backend descriptor.
 *
 * Backends' is_on/isel/emit callbacks receive a const be_cfg_t* so
 * they can read the mode flags, target selectors, and output paths
 * they care about. Kept in one place so main.c and every backend
 * agree on the field layout.
 *
 * If your backend needs a new knob, add the field here, populate it
 * in main.c's CLI parse, read it from your backend descriptor. */

#ifndef BOOTH_BE_CFG_H
#define BOOTH_BE_CFG_H

#include "amdgpu.h"   /* amd_target_t */
#include "intel.h"    /* intel_target_t */

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

#endif /* BOOTH_BE_CFG_H */
