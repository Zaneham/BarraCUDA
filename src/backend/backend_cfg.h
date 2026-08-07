/* backend_cfg.h -- the settings every backend shares, and nothing else.
 * Target selection and per-target knobs live in the backend that owns
 * them, reached through be_desc_t.parse, so this no longer needs to
 * include a single backend header. */

#ifndef BOOTH_BE_CFG_H
#define BOOTH_BE_CFG_H

typedef struct be_cfg {
    int         no_mem2reg, no_cfold, no_dce, no_sched, no_sroa;
    int         mode_ir, mode_tdf, mode_tdf_fission;
    const char *output_file;
} be_cfg_t;

#endif
