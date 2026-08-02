/* backends.c -- registered backend list and dispatch loop.
 *
 * Ooooooh yeah, finally a way to not have to read literally all of
 * the AMD backend to know how to work this thang. If you're adding
 * a new one: extern-declare your descriptor below, add it to be_list,
 * you're wired in.
 *
 * The list is static, NULL-terminated, and iterated in order.
 * be_run walks it, asks each one "you on?", and runs the pipeline
 * for the yeses. No dlopen, no plugin loading, no init-order
 * surprises. JPL discipline: bounded, deterministic, auditable. */

#include "backend.h"
#include "backend_cfg.h"
#include <stdio.h>
#include <string.h>

/* Bound the list walk. If we ever hit 64 registered backends the
 * project has bigger conversations to have. */
#define BE_MAX  64

/* ---- Backend descriptors ----
 * One extern per backend. Descriptor structs live in each backend's
 * own <name>_be.c. Skeleton compiles in only in dev builds. */
extern const be_desc_t be_amd;
extern const be_desc_t be_ptx;
extern const be_desc_t be_tsx;
extern const be_desc_t be_rvcore;
extern const be_desc_t be_x86;
extern const be_desc_t be_rv64;
extern const be_desc_t be_metal;
extern const be_desc_t be_intel;

#ifdef BOOTH_INCLUDE_SKELETON
extern const be_desc_t be_skel;
#endif

/* ---- The list ----
 * Order affects driver output ordering only (which backend message
 * shows first). Work is independent per backend. */
const be_desc_t * const be_list[] = {
    &be_amd,
    &be_ptx,
    &be_tsx,
    &be_rvcore,
    &be_metal,
    &be_intel,
    &be_x86,
    &be_rv64,
#ifdef BOOTH_INCLUDE_SKELETON
    &be_skel,
#endif
    NULL
};

/* ---- Lookup ---- */
const be_desc_t *be_find(const char *name)
{
    if (name == NULL) return NULL;
    for (uint32_t i = 0; be_list[i] != NULL && i < BE_MAX; i++) {
        if (be_list[i]->name != NULL &&
            strcmp(be_list[i]->name, name) == 0) {
            return be_list[i];
        }
    }
    return NULL;
}

/* ---- Dispatch ----
 * Iterate registered backends, run the active ones through the
 * pipeline. Returns first non-OK code seen; keeps going through
 * later actives so a partial invocation still produces the outputs
 * that were fine. */
int be_run(const struct bir_module *M, const be_cfg_t *cfg)
{
    if (M == NULL || cfg == NULL) return BE_EINPUT;

    int first = BE_OK;

    for (uint32_t i = 0; be_list[i] != NULL && i < BE_MAX; i++) {
        const be_desc_t *b = be_list[i];

        /* Skip backends with no is_on or that say no. */
        if (b->is_on == NULL) continue;
        if (!b->is_on(cfg)) continue;

        /* isel is required. If a registered backend has no isel it's
         * a bug in the descriptor, not something we tolerate. */
        if (b->isel == NULL) {
            fprintf(stderr, "backend %s: no isel op registered\n", b->name);
            if (first == BE_OK) first = BE_UNSUP;
            continue;
        }

        void *mmod = NULL;
        int rc = b->isel(M, cfg, &mmod);
        if (rc != BE_OK) {
            if (first == BE_OK) first = rc;
            if (mmod != NULL && b->mfree != NULL) b->mfree(mmod);
            continue;
        }

        /* Optional post-isel verify. */
        if (b->verify != NULL) {
            int v = b->verify(mmod, BE_VFY_ISEL);
            if (v != BE_OK) {
                if (first == BE_OK) first = v;
                if (b->mfree != NULL) b->mfree(mmod);
                continue;
            }
        }

        /* Optional sched. Honour --no-sched. */
        if (b->sched != NULL && !cfg->no_sched) {
            int s = b->sched(mmod);
            if (s != BE_OK) {
                if (first == BE_OK) first = s;
                if (b->mfree != NULL) b->mfree(mmod);
                continue;
            }
        }

        /* Optional regalloc. */
        if (b->regalc != NULL) {
            int r = b->regalc(mmod);
            if (r != BE_OK) {
                if (first == BE_OK) first = r;
                if (b->mfree != NULL) b->mfree(mmod);
                continue;
            }
        }

        /* Post-RA verify. */
        if (b->verify != NULL) {
            int v = b->verify(mmod, BE_VFY_RA);
            if (v != BE_OK) {
                if (first == BE_OK) first = v;
                if (b->mfree != NULL) b->mfree(mmod);
                continue;
            }
        }

        /* Emit. Required. */
        if (b->emit == NULL) {
            fprintf(stderr, "backend %s: no emit op registered\n", b->name);
            if (first == BE_OK) first = BE_UNSUP;
            if (b->mfree != NULL) b->mfree(mmod);
            continue;
        }

        int e = b->emit(mmod, cfg, cfg->output_file);
        if (e != BE_OK) {
            if (first == BE_OK) first = e;
        }

        if (b->mfree != NULL) b->mfree(mmod);
    }

    return first;
}
