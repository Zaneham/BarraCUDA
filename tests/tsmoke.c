/* tsmoke.c -- CLI smoke tests
 * Does the binary do anything? Anything at all? Let's find out. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- smoke: help ---- */

static void smk01(void)
{
    int rc = th_run(BC_BIN " --help", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "Usage") != NULL);
    PASS();
}
TH_REG("smk", 1, "--help prints usage", smk01)

/* ---- smoke: no args ---- */

static void smk02(void)
{
    int rc = th_run(BC_BIN, obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smk", 2, "no args is an error", smk02)

/* ---- smoke: version ---- */

static void smk03(void)
{
    SKIP("not implemented");
}
TH_REG("smk", 3, "--version", smk03)

/* ---- smoke: bad flag ---- */

static void smk04(void)
{
    int rc = th_run(BC_BIN " --nonsense", obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smk", 4, "a bad flag is an error", smk04)

/* ---- smoke: backend gates ---- */

/* Exit 0 with nothing on disk is the worst failure we can hand a build
 * system: it believes us. Every emitting mode gets checked both ways. */

static int smk_emit(const char *args, const char *out)
{
    char cmd[512];
    FILE *f;
    int rc, got;

    remove(out);
    snprintf(cmd, sizeof cmd, "%s %s -o %s", BC_BIN, args, out);
    rc = th_run(cmd, obuf, TH_BUFSZ);
    f = fopen(out, "rb");
    got = f != NULL;
    if (f) fclose(f);
    remove(out);
    return rc == 0 && got;
}

static void smk05(void)
{
    CHECK(smk_emit("--triton --cpu tests/tri_vadd.py", "smk_tc.o"));
    PASS();
}
TH_REG("smk", 5, "Triton to CPU object", smk05)

static void smk06(void)
{
    CHECK(smk_emit("--triton --rv64 tests/tri_vadd.py", "smk_tr.o"));
    PASS();
}
TH_REG("smk", 6, "Triton to RISC-V object", smk06)

static void smk07(void)
{
    CHECK(smk_emit("--triton --nvidia-ptx tests/tri_vadd.py", "smk_tp.ptx"));
    PASS();
}
TH_REG("smk", 7, "Triton to PTX", smk07)

static void smk08(void)
{
    CHECK(smk_emit("--cpu tests/canonical.cu", "smk_cc.o"));
    PASS();
}
TH_REG("smk", 8, "CUDA to CPU object", smk08)

/* The Triton fallthrough returned the lexer's status, so a mode it could
 * not honour still exited 0. --pp on Python is nonsense and must say so. */
static void smk09(void)
{
    int rc = th_run(BC_BIN " --triton --pp tests/tri_vadd.py", obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smk", 9, "--pp on Python must not exit 0", smk09)
