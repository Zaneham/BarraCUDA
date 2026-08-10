/* tguard.c -- regression guard for the divergent early-return lowering.
 * `if (i >= n) return;` must disable the returning lanes with s_andn2 and
 * carry on; the old path masked EXEC down to them and hit s_endpgm, which
 * ends the whole wave and drops a ragged launch's in-range tail. */
#include "tharns.h"

static char buf[1 << 16];

static void grd01(void)
{
    char cmd[TH_BUFSZ];
    int rc;

    snprintf(cmd, TH_BUFSZ, BC_BIN " --amdgpu tests/test_guard.cu");
    rc = th_run(cmd, buf, (int)sizeof buf);
    CHEQ(rc, 0);

    /* The lanes that return are switched off with andn2 ... */
    CHECK(strstr(buf, "s_andn2") != NULL);
    /* ... and a lone guard needs no EXEC save/restore, so its absence is how
       we know we are not back on the s_endpgm-under-mask path. */
    CHECK(strstr(buf, "s_and_saveexec") == NULL);

    PASS();
}
TH_REG("grd", 1, "guarded lanes are masked, not ended", grd01)
