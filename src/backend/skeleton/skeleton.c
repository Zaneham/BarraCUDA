/* skeleton.c -- copy-paste starting point for a new target.
 * Every op returns BE_UNSUP; the driver bails cleanly until you
 * implement one. Rename skel_ to your prefix, register in
 * backends.c, fill in ops. Godspeed. Compiled only under
 * -DBOOTH_INCLUDE_SKELETON so release builds don't ship it. */

#ifdef BOOTH_INCLUDE_SKELETON

#include "backend.h"
#include "backend_cfg.h"

static int skel_on(const be_cfg_t *cfg)      { (void)cfg; return 0; }

static int skel_isel(const struct bir_module *M, const be_cfg_t *cfg,
                     void **out_mmod)
{
    (void)M; (void)cfg; (void)out_mmod;
    return BE_UNSUP;
}

static int skel_emit(const void *mmod, const be_cfg_t *cfg,
                     const char *out_path)
{
    (void)mmod; (void)cfg; (void)out_path;
    return BE_UNSUP;
}

const be_desc_t be_skel = {
    .name    = "skeleton",
    .triple  = NULL,
    .feats   = 0,
    .is_on   = skel_on,
    .isel    = skel_isel,
    .sched   = NULL,
    .regalc  = NULL,
    .verify  = NULL,
    .emit    = skel_emit,
    .mfree   = NULL
};

#endif
