/* trpi.c -- one test per bug that got away
 *
 * z390 files these as RPI1540, RPI2001A and so on, named for the problem report
 * they came from, so years later you can still tell why a test exists at all.
 * Ours are numbered in sequence like every other family and carry the issue in
 * the description instead, which greps just as well and means there is only one
 * naming rule to remember.
 *
 * The bar for landing here is that the bug shipped. If a fix has no test, the
 * fix is a coincidence waiting to be undone. */

#include "tharns.h"

static char obuf[1 << 16];

/* Reads the "; line N" annotation bir_print puts on the instruction matching
 * needle, searching only within the named function. */
static int line_of(const char *fn, const char *needle)
{
    const char *f = strstr(obuf, fn);
    if (!f) return -1;
    const char *end = strstr(f, "\n}");
    if (!end) return -1;

    const char *p = strstr(f, needle);
    if (!p || p > end) return -1;

    const char *tag = strstr(p, "; line ");
    if (!tag || tag > end) return -1;
    return atoi(tag + 7);
}

/* #160: DCE and mem2reg shuffled instructions down over the top of a deletion
 * without moving inst_lines[] along with them, so every instruction past the
 * first thing deleted reported whatever line its old neighbour had. Four sites
 * were fixed and none of them got a test, which is what this is.
 *
 * dce_chain in test_dce.cu is the shape that catches it. Two dead instructions
 * on lines 10 and 11 go away, and the store after them is on line 12. Get the
 * line table wrong and the store starts claiming line 10 or 11. */
static void rpi01(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    /* int live = a + b; */
    CHEQ(line_of("@dce_chain", "= add "), 9);
    /* out[0] = live; sits two deleted instructions later */
    CHEQ(line_of("@dce_chain", "store "), 12);

    PASS();
}
TH_REG("rpi", 1, "#160 line numbers survive DCE", rpi01)

/* Every backend lists its variant flags next to its on-switch, so a variant on
 * its own was accepted, switched nothing on, and fell through to the AST dump
 * the driver uses when no mode is set. kath printed a parse tree and exited 0
 * having compiled nothing, under whatever -o you asked for. */
static void rpi02(void)
{
    static const char *const variants[] = {
        "--bkhit", "--gfx942", "--snap", "--ssa-ra", NULL
    };
    char cmd[512];

    for (int i = 0; variants[i] != NULL; i++) {
        snprintf(cmd, sizeof cmd, "%s %s examples/cmake/vadd.cu -o build/rpi02.out",
                 BC_BIN, variants[i]);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    }

    /* The same flags alongside their target still work. */
    snprintf(cmd, sizeof cmd,
             "%s --nvidia-ptx --bkhit examples/cmake/vadd.cu -o build/rpi02.ptx",
             BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);

    /* And a bare run still prints the tree, which is what the default is for. */
    snprintf(cmd, sizeof cmd, "%s examples/cmake/vadd.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    PASS();
}
TH_REG("rpi", 2, "a variant flag alone is not a target", rpi02)
