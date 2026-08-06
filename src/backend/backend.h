/* backend.h -- the shape every Booth backend implements.
 * Seven function pointers, one struct, no dlopen, no frameworks.
 * See docs/backends.md for the walkthrough. */

#ifndef BOOTH_BE_H
#define BOOTH_BE_H

#include <stdint.h>

struct bir_module;
struct be_cfg;

/* ---- Return codes ---- */

typedef enum {
    BE_OK      =  0,
    BE_UNSUP   = -1,  /* op deliberately not implemented */
    BE_EISEL   = -2,
    BE_ESCHED  = -3,
    BE_ERA     = -4,
    BE_EVFY    = -5,
    BE_EEMIT   = -6,
    BE_EIO     = -7,
    BE_EINPUT  = -8,
    BE_ENOMEM  = -9
} be_ret_t;

/* ---- Verify phase ----
 * Only the two the driver actually calls. Phases for post-sched and
 * post-emit were declared here first, but nothing invoked them and the
 * AMD adapter folds every non-ISEL phase onto its post-regalloc check,
 * so a backend implementing them would have been checking physical
 * registers on a module that still had virtual ones. Add them back with
 * the call sites, not before. */

typedef enum {
    BE_VFY_ISEL,
    BE_VFY_RA
} be_vfy_t;

/* ---- Feature flags ----
 * A backend must not claim a feature it does not implement. The
 * conformance suite (not yet landed) keys tests off these bits. */

#define BE_F_SIMT       (1u <<  0)
#define BE_F_SCALAR     (1u <<  1)
#define BE_F_ATOMIC     (1u <<  2)
#define BE_F_SHARED     (1u <<  3)
#define BE_F_WARP       (1u <<  4)
#define BE_F_MFMA       (1u <<  5)
#define BE_F_BARRIER    (1u <<  6)
#define BE_F_DIV        (1u <<  7)
#define BE_F_SCRATCH    (1u <<  8)
#define BE_F_TRANSC     (1u <<  9)
#define BE_F_F16        (1u << 10)
#define BE_F_F64        (1u << 11)
#define BE_F_BF16       (1u << 12)
#define BE_F_MULTIOUT   (1u << 13)  /* emits more than one file (Tensix) */

/* ---- Backend descriptor ----
 * NULL op means "not applicable, driver skips". isel and emit are
 * required; a registered backend without them is a bug. cfg is
 * opaque here so the contract does not couple to driver internals;
 * implementations cast to be_cfg_t. */

typedef struct be_desc {
    const char *name;
    const char *triple;
    uint32_t    feats;

    int  (*is_on)  (const struct be_cfg *cfg);
    int  (*isel)   (const struct bir_module *M, const struct be_cfg *cfg,
                    void **out_mmod);
    int  (*sched)  (void *mmod);
    int  (*regalc) (void *mmod);
    int  (*verify) (const void *mmod, int phase);
    int  (*emit)   (const void *mmod, const struct be_cfg *cfg,
                    const char *out_path);
    void (*mfree)  (void *mmod);
} be_desc_t;

/* ---- Registration ---- */

extern const be_desc_t * const be_list[];   /* NULL-terminated */

const be_desc_t *be_find(const char *name);

int be_run(const struct bir_module *M, const struct be_cfg *cfg);

#endif
