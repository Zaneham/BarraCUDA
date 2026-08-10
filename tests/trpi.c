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
