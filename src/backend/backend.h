/* backend.h -- the contract every Booth backend implements.
 *
 * Hello human or LLM asked to clone this: welcome. If you are here
 * because you want to add a new backend, the whole show is in this
 * file plus src/backend/skeleton/. Copy skeleton, edit copy, register
 * in backends.c, done. See docs/backends.md for the walkthrough.
 *
 * The contract is a struct of function pointers. Backends fill in
 * what they support and leave the rest NULL. The driver iterates a
 * static registration list, calls each active backend, moves on.
 *
 * JPL discipline: no dynamic dispatch by name lookup in hot paths,
 * bounded static list, every op has a defined return convention,
 * NULL means "not this backend" not "silently succeed". */

#ifndef BOOTH_BE_H
#define BOOTH_BE_H

#include <stdint.h>

/* Forward declarations. Backends see the full types via their own
 * headers; the contract itself does not need to know the shapes. */
struct bir_module;
struct be_cfg;

/* ---- Return codes ----
 * Zero is success. Negative is a defined failure. Positive currently
 * unused, reserved for warnings if we ever want them. */
typedef enum {
    BE_OK           =  0,
    BE_UNSUP        = -1,  /* op deliberately not implemented */
    BE_EISEL        = -2,
    BE_ESCHED       = -3,
    BE_ERA          = -4,
    BE_EVFY         = -5,
    BE_EEMIT        = -6,
    BE_EIO          = -7,
    BE_EINPUT       = -8,
    BE_ENOMEM       = -9
} be_ret_t;

/* ---- Verify phase ----
 * Backends with a verifier get called at each phase; they decide
 * which ones actually run checks. */
typedef enum {
    BE_VFY_ISEL,
    BE_VFY_SCHED,
    BE_VFY_RA,
    BE_VFY_EMIT
} be_vfy_t;

/* ---- Feature flags ----
 * Backends declare what they honestly implement. Conformance suite
 * (Phase B) runs the tests matching declared features. If your bit
 * is set the tests will call you on it, so don't oversell. */
#define BE_F_SIMT       (1u <<  0)  /* SIMT execution model */
#define BE_F_SCALAR     (1u <<  1)  /* scalar / stack-everything */
#define BE_F_ATOMIC     (1u <<  2)  /* atomic RMW family */
#define BE_F_SHARED     (1u <<  3)  /* per-block shared memory */
#define BE_F_WARP       (1u <<  4)  /* shfl, ballot, vote */
#define BE_F_MFMA       (1u <<  5)  /* matrix multiply-accumulate */
#define BE_F_BARRIER    (1u <<  6)  /* __syncthreads equivalent */
#define BE_F_DIV        (1u <<  7)  /* per-lane divergence handling */
#define BE_F_SCRATCH    (1u <<  8)  /* per-thread private */
#define BE_F_TRANSC     (1u <<  9)  /* sin, cos, exp2, log2 */
#define BE_F_F16        (1u << 10)
#define BE_F_F64        (1u << 11)
#define BE_F_BF16       (1u << 12)
#define BE_F_MULTIOUT   (1u << 13)  /* emits multiple files (Tensix) */

/* ---- Backend descriptor ----
 * Every backend defines one of these. Static const, lives in the
 * backend's own .c file, registered in src/backend/backends.c.
 *
 * Ops in pipeline order. NULL means "not applicable, driver skips":
 *   isel     - required. Consumes BIR, produces backend module.
 *   sched    - optional. Reorders instructions.
 *   regalloc - optional. Assigns physical regs.
 *   verify   - optional. Called at each defined phase.
 *   emit     - required. Writes final output(s) to disk.
 *   free_mmod- required if isel allocates.
 *
 * cfg is Booth's driver config; backends cast to be_cfg_t and pull
 * the knobs they care about. Opaque here on purpose so the contract
 * does not couple to driver internals. */
typedef struct be_desc {
    const char *name;              /* stable id: "amdgpu", "nvptx", ... */
    const char *triple;            /* "amdgcn--", "nvptx64--", or NULL */
    uint32_t    feats;             /* BE_F_* bitmask */

    /* True when this backend should run for this invocation. Cheap. */
    int  (*is_on)(const struct be_cfg *cfg);

    /* Pipeline. */
    int  (*isel)   (const struct bir_module *M,
                    const struct be_cfg *cfg,
                    void **out_mmod);
    int  (*sched)  (void *mmod);
    int  (*regalc) (void *mmod);
    int  (*verify) (const void *mmod, int phase);
    int  (*emit)   (const void *mmod,
                    const struct be_cfg *cfg,
                    const char *out_path);
    void (*mfree)  (void *mmod);
} be_desc_t;

/* ---- Registration ----
 * Static NULL-terminated list, populated in backends.c. Adding a
 * backend means one extern decl + one entry in the array. No init
 * order surprises. */
extern const be_desc_t * const be_list[];

/* Find by name. Returns NULL if unknown. O(n) over ~10 entries. */
const be_desc_t *be_find(const char *name);

/* ---- Driver-side glue ----
 * The loop that iterates registered backends, calls is_on on each,
 * runs the pipeline for the actives. Returns first non-OK code, or
 * OK if all actives succeeded. */
int be_run(const struct bir_module *M, const struct be_cfg *cfg);

#endif /* BOOTH_BE_H */
