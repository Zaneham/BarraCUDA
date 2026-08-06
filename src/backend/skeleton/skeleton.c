/* skeleton.c -- copy-paste starting point for a new target.
 * Every op returns BE_UNSUP; the driver bails cleanly until you
 * implement one. Rename skel_ to your prefix, register in
 * backends.c, fill in ops. Godspeed. Compiled only under
 * -DBOOTH_INCLUDE_SKELETON so release builds don't ship it. */

#ifdef BOOTH_INCLUDE_SKELETON

#include "backend.h"
#include "backend_cfg.h"
#include <string.h>

/* Whatever your flags set. The driver keeps one of these per backend
 * and hands it back to every op, so nothing target-specific has to go
 * anywhere near be_cfg_t. */
typedef struct {
    int on;
} skel_opts_t;

/* The flags you answer for. The suite checks no two backends claim the
 * same one, and that each appears in usage(). */
static const char *const skel_flags[] = { "--skeleton", NULL };

static int skel_parse(const char *arg, const char *next, void *o)
{
    (void)next;   /* return 1 instead if you consume the following entry */
    if (strcmp(arg, "--skeleton") == 0) { ((skel_opts_t *)o)->on = 1; }
    return 0;
}

static int skel_on(const void *o) { return ((const skel_opts_t *)o)->on; }

static int skel_isel(const struct bir_module *M, const be_cfg_t *cfg,
                     const void *o, void **out_mmod)
{
    (void)M; (void)cfg; (void)o; (void)out_mmod;
    return BE_UNSUP;
}

static int skel_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                     const char *out_path)
{
    (void)mmod; (void)cfg; (void)o; (void)out_path;
    return BE_UNSUP;
}

const be_desc_t be_skel = {
    .name      = "skeleton",
    .triple    = NULL,
    .feats     = 0,
    .opts_size = sizeof(skel_opts_t),
    .flags     = skel_flags,
    .parse     = skel_parse,
    .is_on     = skel_on,
    .isel      = skel_isel,
    .sched     = NULL,
    .regalc    = NULL,
    .verify    = NULL,
    .emit      = skel_emit,
    .mfree     = NULL
};

#endif
