/* intel_be.c -- Intel Arc SPIR-V as a be_desc_t.
 * Even more stub than Metal; intel_compile prints an honest "not yet
 * a working compiler" so nobody thinks they got a real .spv. */

#include "backend.h"
#include "backend_cfg.h"
#include "intel.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int            on;
    intel_target_t target;
} in_opts_t;

static const char *const in_flags[] = {
    "--intel-spirv", "--xe-lpg", "--xe-hpg", "--xe-hpc", "--xe2", NULL
};

static int in_parse(const char *arg, const char *next, void *o)
{
    in_opts_t *p = (in_opts_t *)o;
    (void)next;
    if (strcmp(arg, "--intel-spirv") == 0) { p->on = 1; return 0; }
    if (strcmp(arg, "--xe-lpg") == 0) { p->target = INTEL_TARGET_XE_LPG; return 0; }
    if (strcmp(arg, "--xe-hpg") == 0) { p->target = INTEL_TARGET_XE_HPG; return 0; }
    if (strcmp(arg, "--xe-hpc") == 0) { p->target = INTEL_TARGET_XE_HPC; return 0; }
    if (strcmp(arg, "--xe2")    == 0) { p->target = INTEL_TARGET_XE2;    return 0; }
    return 0;
}

static int in_on(const void *o) { return ((const in_opts_t *)o)->on; }

static int in_isel(const struct bir_module *M, const be_cfg_t *cfg,
                   const void *o, void **out_mmod)
{
    const in_opts_t *p = (const in_opts_t *)o;
    (void)cfg;
    intel_module_t *im = calloc(1, sizeof(*im));
    if (im == NULL) return BE_ENOMEM;
    if (intel_compile((const bir_module_t *)M, im, p->target) != BC_OK) {
        fprintf(stderr, "error: Intel SPIR-V backend not yet a working compiler\n");
        free(im);
        return BE_EISEL;
    }
    *out_mmod = im;
    return BE_OK;
}

static int in_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                   const char *out)
{
    (void)cfg; (void)o;
    return intel_emit_spirv((const intel_module_t *)mmod,
                            out ? out : "a.spv") == BC_OK
         ? BE_OK : BE_EEMIT;
}

const be_desc_t be_intel = {
    .name    = "intel-spirv",
    .triple  = NULL,
    .feats   = BE_F_SIMT | BE_F_SHARED | BE_F_BARRIER,
    .opts_size = sizeof(in_opts_t),
    .flags   = in_flags,
    .parse   = in_parse,
    .is_on   = in_on,
    .isel    = in_isel,
    .emit    = in_emit,
    .mfree   = free
};
