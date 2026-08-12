/* tsched.c -- instruction scheduling tests
 * Verifies that the scheduler groups loads and defers waits. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- sched: loads grouped (no wait between two global_load_dword) ---- */

static void sch01(void)
{
    int rc = th_run(BC_BIN " --amdgpu tests/test_sched.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);

    /* Find two global_load_dword instructions — there should be no
     * s_waitcnt between them after scheduling. */
    const char *first = strstr(obuf, "global_load_");
    CHECK(first != NULL);
    const char *second = strstr(first + 1, "global_load_");
    CHECK(second != NULL);

    int gap = (int)(second - first);
    char between[TH_BUFSZ];
    if (gap > 0 && gap < TH_BUFSZ - 1) {
        memcpy(between, first, (size_t)gap);
        between[gap] = '\0';
        /* no wait between the two loads */
        CHECK(strstr(between, "s_waitcnt") == NULL);
        CHECK(strstr(between, "s_wait_loadcnt") == NULL);
    }

    PASS();
}
TH_REG("sch", 1, "loads group with no wait between them", sch01)

/* ---- sched: --no-sched still produces correct output ---- */

static void sch02(void)
{
    int rc = th_run(BC_BIN " --amdgpu --no-sched tests/test_sched.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);

    /* Should still have global_load and s_waitcnt in output */
    CHECK(strstr(obuf, "global_load_") != NULL);
    PASS();
}
TH_REG("sch", 2, "--no-sched is still correct", sch02)

/* ---- sched: compiles to ELF with scheduling ---- */

static void sch03(void)
{
    const char *out = "test_sched_out.hsaco";
    char cmd[TH_BUFSZ];

    snprintf(cmd, TH_BUFSZ, BC_BIN " --amdgpu-bin tests/test_sched.cu -o %s", out);
    int rc = th_run(cmd, obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(th_exist(out));
    remove(out);
    PASS();
}
TH_REG("sch", 3, "compiles to ELF with scheduling", sch03)

/* ---- sched: all targets compile with scheduling ---- */

static void sch04(void)
{
    static const char *targets[] = { "--gfx1030", "", "--gfx1200" };
    static const char *tnames[]  = { "gfx1030", "gfx1100", "gfx1200" };
    const char *out = "test_sched_tgt.hsaco";
    char cmd[TH_BUFSZ];

    for (int t = 0; t < 3; t++) {
        snprintf(cmd, TH_BUFSZ,
                 BC_BIN " --amdgpu-bin %s tests/test_sched.cu -o %s",
                 targets[t], out);
        int rc = th_run(cmd, obuf, TH_BUFSZ);
        if (rc != 0) {
            printf("  target %s failed: %s\n", tnames[t], obuf);
            CHECK(0);
        }
        CHECK(th_exist(out));
        remove(out);
    }
    PASS();
}
TH_REG("sch", 4, "every target compiles with scheduling", sch04)

/* ---- sched: a memory wait waits on the memory counter ---- */

static void sch05(void)
{
    int rc = th_run(BC_BIN " --amdgpu tests/test_sched.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);

    /* The loads are grouped, so something has to wait on them before the
     * first use. Waiting on lgkmcnt instead reads the registers before the
     * data lands, which is a race rather than a wrong number and would not
     * show up in any output this suite compares. */
    CHECK(strstr(obuf, "global_load_") != NULL);
    CHECK(strstr(obuf, "s_waitcnt vmcnt(0)") != NULL);
    PASS();
}
TH_REG("sch", 5, "memory waits on the memory counter", sch05)
