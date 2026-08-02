/* tensix_be.c -- Tensix Metalium and RV32 baby-core as be_desc_t.
 *
 * Two descriptors for the two Tensix personalities:
 *   be_tsx    = --tensix, Metalium C++ compute + reader + writer + host
 *   be_rvcore = --rv-elf, native RV32IM machine code for baby cores
 *
 * The Metalium path emits a small forest of files; the multi-output
 * dance stays inside tt_emit, driver only asks for one op. */

#include "backend.h"
#include "backend_cfg.h"
#include "tensix.h"
#include "rv_isel.h"
#include "rv_elf.h"
#include "rv_buf.h"
#include "barracuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Metalium / Tensix Tensix ----
 * Stashes the BIR pointer alongside tt_module_t so emit doesn't
 * need it re-plumbed through the driver. datamov analysis happens
 * inside isel while the BIR is still in hand. */

typedef struct {
    tt_module_t          *tt;
    const bir_module_t   *bir;
} tt_wrap_t;

static int tt_on(const be_cfg_t *cfg)
{
    return cfg->mode_tensix;
}

static int tt_isel(const struct bir_module *M,
                   const be_cfg_t *cfg,
                   void **out_mmod)
{
    (void)cfg;
    tt_wrap_t *w = (tt_wrap_t *)calloc(1, sizeof(tt_wrap_t));
    if (w == NULL) return BE_ENOMEM;
    w->tt = (tt_module_t *)calloc(1, sizeof(tt_module_t));
    if (w->tt == NULL) { free(w); return BE_ENOMEM; }
    w->bir = (const bir_module_t *)M;

    int rc = tensix_compile(w->bir, w->tt);
    if (rc != BC_OK) {
        fprintf(stderr, "error: Tensix compilation failed\n");
        free(w->tt); free(w);
        return BE_EISEL;
    }
    tensix_coarsen(w->tt);
    tensix_regalloc(w->tt);
    tensix_analyze_datamov(w->bir, w->tt, &w->tt->dmov);
    *out_mmod = w;
    return BE_OK;
}

static int tt_emit(const void *mmod, const be_cfg_t *cfg,
                   const char *out_path)
{
    (void)cfg;
    const tt_wrap_t *w = (const tt_wrap_t *)mmod;
    tt_module_t *ttm = w->tt;
    const char *compute_path = out_path ? out_path : "a_compute.cpp";

    tensix_emit_metalium(ttm, compute_path);

    /* Sibling paths derived from compute stem (strip _compute or .ext). */
    char bin_path[BC_MAX_PATH];
    const char *st2 = strstr(compute_path, "_compute");
    int bp = st2 ? (int)(st2 - compute_path) : (int)strlen(compute_path);

    snprintf(bin_path, sizeof(bin_path), "%.*s_compute.bin",
             bp, compute_path);
    tensix_emit_binary(ttm, bin_path);
    snprintf(bin_path, sizeof(bin_path), "%.*s_compute.ttinsn",
             bp, compute_path);
    tensix_emit_ttinsn(ttm, bin_path);

    char host_path[BC_MAX_PATH];
    char reader_path[BC_MAX_PATH];
    char writer_path[BC_MAX_PATH];
    const char *stem = strstr(compute_path, "_compute");
    int pfx;
    if (stem) pfx = (int)(stem - compute_path);
    else {
        const char *dot = strrchr(compute_path, '.');
        pfx = dot ? (int)(dot - compute_path)
                  : (int)strlen(compute_path);
    }
    snprintf(host_path,   sizeof(host_path),
             "%.*s_host.cpp",   pfx, compute_path);
    snprintf(reader_path, sizeof(reader_path),
             "%.*s_reader.cpp", pfx, compute_path);
    snprintf(writer_path, sizeof(writer_path),
             "%.*s_writer.cpp", pfx, compute_path);
    tensix_emit_reader(ttm, &ttm->dmov, reader_path);
    tensix_emit_writer(ttm, &ttm->dmov, writer_path);
    tensix_emit_host_full(ttm, &ttm->dmov, host_path,
                          reader_path, compute_path, writer_path);

    /* ELF drop for the toolchain-free path. */
    {
        char elf_stem[BC_MAX_PATH];
        snprintf(elf_stem, sizeof(elf_stem), "%.*s", pfx, compute_path);
        tensix_emit_kernel_elves(ttm, &ttm->dmov, elf_stem);
    }
    return BE_OK;
}

static void tt_free(void *mmod)
{
    tt_wrap_t *w = (tt_wrap_t *)mmod;
    if (w == NULL) return;
    free(w->tt);
    free(w);
}

const be_desc_t be_tsx = {
    .name    = "tensix",
    .triple  = NULL,
    .feats   = BE_F_SIMT | BE_F_SHARED | BE_F_BARRIER | BE_F_MFMA
             | BE_F_MULTIOUT | BE_F_F16 | BE_F_BF16,
    .is_on   = tt_on,
    .isel    = tt_isel,
    .sched   = NULL,
    .regalc  = NULL,
    .verify  = NULL,
    .emit    = tt_emit,
    .mfree   = tt_free
};

/* ---- Tensix baby-core RV32IM ----
 * Different pipeline entirely: BIR to raw RV32IM instructions in an
 * rv_buf_t, then straight to ELF. */

static int rv_on(const be_cfg_t *cfg)
{
    return cfg->mode_rv_elf;
}

static int rv_isel(const struct bir_module *M,
                   const be_cfg_t *cfg,
                   void **out_mmod)
{
    (void)cfg;
    const bir_module_t *bir = (const bir_module_t *)M;
    if (bir->num_funcs == 0u) {
        fprintf(stderr, "error: BIR module has no functions\n");
        return BE_EINPUT;
    }
    rv_buf_t *code = (rv_buf_t *)calloc(1, sizeof(rv_buf_t));
    if (code == NULL) return BE_ENOMEM;
    rv_buf_init(code);
    int rc = rv_isel_module(bir, code);
    if (rc != BC_OK) { free(code); return BE_EISEL; }
    *out_mmod = code;
    return BE_OK;
}

static int rv_emit(const void *mmod, const be_cfg_t *cfg,
                   const char *out_path)
{
    (void)cfg;
    const rv_buf_t *code = (const rv_buf_t *)mmod;
    const char *p = out_path ? out_path : "a.elf";
    int rc = rv_elf_write(code, p);
    if (rc != BC_OK) return BE_EIO;
    fprintf(stderr, "wrote %s (%u bytes code, %u instructions)\n",
            p, rv_buf_nbytes(code), rv_buf_n_words(code));
    return BE_OK;
}

static void rv_free(void *mmod) { free(mmod); }

const be_desc_t be_rvcore = {
    .name    = "tensix-rv32",
    .triple  = "riscv32-unknown-none",
    .feats   = BE_F_SCALAR,
    .is_on   = rv_on,
    .isel    = rv_isel,
    .sched   = NULL,
    .regalc  = NULL,
    .verify  = NULL,
    .emit    = rv_emit,
    .mfree   = rv_free
};
