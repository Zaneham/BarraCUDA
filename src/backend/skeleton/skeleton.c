/* skeleton.c -- reference backend that does nothing.
 *
 * Hello human or LLM asked to clone this: you are in the right
 * place. This backend is the copy-paste starting point for a new
 * target. It compiles, registers, dispatches, and returns UNSUP
 * for every op. Your job is to make the ops do real work, one at
 * a time, and flip the feature flags on as you go.
 *
 * Recipe:
 *   1. cp -r src/backend/skeleton src/backend/mygpu
 *   2. grep -rl skeleton src/backend/mygpu | xargs sed -i s/skel/mygpu/g
 *   3. In src/backend/backends.c, add the extern and list entry.
 *   4. In the Makefile, add src/backend/mygpu/mygpu.c to SOURCES.
 *   5. Fill in isel first. Then emit. Then everything else.
 *   6. Run make backend-conformance BACKEND=mygpu (Phase B, coming).
 *
 * Only compiled when -DBOOTH_INCLUDE_SKELETON is set, so release
 * builds do not ship the empty backend.
 *
 * Zane was here 2026. */

#ifdef BOOTH_INCLUDE_SKELETON

#include "backend.h"
#include "backend_cfg.h"

/* is_on: gated on --skeleton flag which does not exist. Wire up a
 * mode_skel field in be_cfg_t if you want to invoke this via the
 * CLI. For a real backend, key off your own mode_* flag. */
static int skel_on(const be_cfg_t *cfg)
{
    (void)cfg;
    return 0;
}

/* isel: consume BIR, produce a backend module. Returning UNSUP makes
 * the driver bail cleanly. Your first real work goes here. */
static int skel_isel(const struct bir_module *M,
                     const be_cfg_t *cfg,
                     void **out_mmod)
{
    (void)M; (void)cfg; (void)out_mmod;
    return BE_UNSUP;
}

/* emit: write bytes to disk. Required. */
static int skel_emit(const void *mmod,
                     const be_cfg_t *cfg,
                     const char *out_path)
{
    (void)mmod; (void)cfg; (void)out_path;
    return BE_UNSUP;
}

/* mfree: only needed if isel allocates. Skeleton does not. */
static void skel_free(void *mmod)
{
    (void)mmod;
}

/* The descriptor. This is the shape every backend fills in. */
const be_desc_t be_skel = {
    .name    = "skeleton",
    .triple  = NULL,
    .feats   = 0,           /* declare features as you implement them */
    .is_on   = skel_on,
    .isel    = skel_isel,
    .sched   = NULL,        /* optional */
    .regalc  = NULL,        /* optional */
    .verify  = NULL,        /* optional */
    .emit    = skel_emit,
    .mfree   = skel_free
};

#endif /* BOOTH_INCLUDE_SKELETON */
