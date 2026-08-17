/* cpu_be.c -- x86-64 and RV64 as be_desc_t descriptors.
 * The reason a Triton kernel runs on a laptop with no GPU in it. */

#include "backend.h"
#include "backend_cfg.h"
#include "cpu.h"
#include "rv64.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- x86-64 ---- */

typedef struct { int on; } cpu_opts_t;

static const char *const x86_flags[] = { "--cpu", NULL };

static int x86_parse(const char *arg, const char *next, void *o)
{
    (void)next;
    if (strcmp(arg, "--cpu") == 0) { ((cpu_opts_t *)o)->on = 1; }
    return 0;
}

static int x86_on(const void *o) { return ((const cpu_opts_t *)o)->on; }

static int x86_isel(const struct bir_module *M, const be_cfg_t *cfg,
                    const void *o, void **out_mmod)
{
    (void)cfg; (void)o;
    const bir_module_t *bir = (const bir_module_t *)M;
    if (bir->num_funcs == 0u) {
        fprintf(stderr, "error: no functions\n");
        return BE_EINPUT;
    }
    cpu_mod_t *cm = calloc(1, sizeof(*cm));
    if (cm == NULL) return BE_ENOMEM;
    cpu_init(cm, bir);
    if (cpu_emit(cm) != 0) { free(cm); return BE_EISEL; }
    /* A refused op leaves a hole, so don't write the object */
    if (cm->n_errs != 0) { free(cm); return BE_EISEL; }
    *out_mmod = cm;
    return BE_OK;
}

static int x86_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                   const char *out)
{
    (void)cfg; (void)o;
    const cpu_mod_t *cm = mmod;
    const char *p = out ? out : "a.o";
    if (cpu_elf(cm, p) != 0) {
        fprintf(stderr, "cpu: elf write failed\n");
        return BE_EIO;
    }
    fprintf(stderr, "wrote %s (%u bytes x86-64)\n", p, cm->codelen);
    return BE_OK;
}

const be_desc_t be_x86 = {
    .name    = "cpu-x86-64",
    .triple  = "x86_64-unknown-none",
    .feats   = BE_F_SCALAR | BE_F_TRANSC | BE_F_F64,
    .opts_size = sizeof(cpu_opts_t),
    .flags   = x86_flags,
    .parse   = x86_parse,
    .is_on   = x86_on,
    .isel    = x86_isel,
    .emit    = x86_emit,
    .mfree   = free
};

/* ---- RV64 ---- */

static const char *const rv_flags[] = { "--rv64", NULL };

static int rv_parse(const char *arg, const char *next, void *o)
{
    (void)next;
    if (strcmp(arg, "--rv64") == 0) { ((cpu_opts_t *)o)->on = 1; }
    return 0;
}

static int rv_on(const void *o) { return ((const cpu_opts_t *)o)->on; }

static int rv_isel(const struct bir_module *M, const be_cfg_t *cfg,
                   const void *o, void **out_mmod)
{
    (void)cfg; (void)o;
    const bir_module_t *bir = (const bir_module_t *)M;
    if (bir->num_funcs == 0u) {
        fprintf(stderr, "error: no functions\n");
        return BE_EINPUT;
    }
    rv64_mod_t *vm = calloc(1, sizeof(*vm));
    if (vm == NULL) return BE_ENOMEM;
    rv64_init(vm, bir);
    if (rv64_emit(vm) != 0) { free(vm); return BE_EISEL; }
    if (vm->n_errs != 0) { free(vm); return BE_EISEL; }
    *out_mmod = vm;
    return BE_OK;
}

static int rv_emit(const void *mmod, const be_cfg_t *cfg, const void *o,
                   const char *out)
{
    (void)cfg; (void)o;
    const rv64_mod_t *vm = mmod;
    const char *p = out ? out : "a.o";
    if (rv64_elf(vm, p) != 0) {
        fprintf(stderr, "rv64: elf write failed\n");
        return BE_EIO;
    }
    fprintf(stderr, "wrote %s (%u bytes RV64)\n", p, vm->codelen);
    return BE_OK;
}

const be_desc_t be_rv64 = {
    .name    = "cpu-rv64",
    .triple  = "riscv64-unknown-none",
    .feats   = BE_F_SCALAR | BE_F_TRANSC | BE_F_F64,
    .opts_size = sizeof(cpu_opts_t),
    .flags   = rv_flags,
    .parse   = rv_parse,
    .is_on   = rv_on,
    .isel    = rv_isel,
    .emit    = rv_emit,
    .mfree   = free
};
