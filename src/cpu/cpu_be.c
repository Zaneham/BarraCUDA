/* cpu_be.c -- x86-64 and RV64 as be_desc_t descriptors.
 * The reason a Triton kernel runs on a laptop with no GPU in it. */

#include "backend.h"
#include "backend_cfg.h"
#include "cpu.h"
#include "rv64.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>

/* ---- x86-64 ---- */

static int x86_on(const be_cfg_t *cfg) { return cfg->mode_cpu; }

static int x86_isel(const struct bir_module *M, const be_cfg_t *cfg,
                    void **out_mmod)
{
    (void)cfg;
    const bir_module_t *bir = (const bir_module_t *)M;
    if (bir->num_funcs == 0u) {
        fprintf(stderr, "error: no functions\n");
        return BE_EINPUT;
    }
    cpu_mod_t *cm = calloc(1, sizeof(*cm));
    if (cm == NULL) return BE_ENOMEM;
    cpu_init(cm, bir);
    if (cpu_emit(cm) != 0) { free(cm); return BE_EISEL; }
    *out_mmod = cm;
    return BE_OK;
}

static int x86_emit(const void *mmod, const be_cfg_t *cfg, const char *out)
{
    (void)cfg;
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
    .is_on   = x86_on,
    .isel    = x86_isel,
    .emit    = x86_emit,
    .mfree   = free
};

/* ---- RV64 ---- */

static int rv_on(const be_cfg_t *cfg) { return cfg->mode_rv64; }

static int rv_isel(const struct bir_module *M, const be_cfg_t *cfg,
                   void **out_mmod)
{
    (void)cfg;
    const bir_module_t *bir = (const bir_module_t *)M;
    if (bir->num_funcs == 0u) {
        fprintf(stderr, "error: no functions\n");
        return BE_EINPUT;
    }
    rv64_mod_t *vm = calloc(1, sizeof(*vm));
    if (vm == NULL) return BE_ENOMEM;
    rv64_init(vm, bir);
    if (rv64_emit(vm) != 0) { free(vm); return BE_EISEL; }
    *out_mmod = vm;
    return BE_OK;
}

static int rv_emit(const void *mmod, const be_cfg_t *cfg, const char *out)
{
    (void)cfg;
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
    .is_on   = rv_on,
    .isel    = rv_isel,
    .emit    = rv_emit,
    .mfree   = free
};
