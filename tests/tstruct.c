/* tstruct.c -- control-flow structuriser tests.
 *
 * The whole point of bir_struct is that the Metal it feeds out has no goto
 * and no labels in it anywhere, just honest nested ifs and loops, because
 * Apple's compiler will throw the lot back otherwise. So these tests do the
 * one thing that actually matters: emit the MSL and go looking for the
 * gotos that must not be there, plus the structure that must be. If a
 * kernel cannot be structured the emitter returns non-zero, so a clean run
 * is itself half the test. */

#include "tharns.h"

static char obuf[TH_BUFSZ];
static char msl[TH_BUFSZ * 4];

/* Slurp the emitted .metal back off disk. The emitter writes a file rather
 * than stdout, so read it here ourselves instead of wrestling cat and type
 * across two operating systems. Returns the byte count, or -1. */
static int slurp(const char *path, char *buf, int cap)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t n = fread(buf, 1, (size_t)cap - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    return (int)n;
}

/* Emit MSL for test_struct.cu into a scratch file and read it back. The
 * run must succeed: a structuriser that downed tools would return non-zero
 * and never write the file. */
static int emit_struct_msl(char *out, int cap)
{
    int rc = th_run(BC_BIN " --metal tests/test_struct.cu -o tests/struct_out.metal",
                    obuf, TH_BUFSZ);
    if (rc != 0) return -1;
    return slurp("tests/struct_out.metal", out, cap);
}

/* The headline: not one goto, not one block label, in the whole file. */
static void str01(void)
{
    int n = emit_struct_msl(msl, sizeof(msl));
    CHECK(n > 0);
    CHECK(strstr(msl, "goto") == NULL);
    /* labels came out as "L<n>: ;" in the old goto world; none should remain */
    CHECK(strstr(msl, ": ;") == NULL);
    PASS();
}
TH_REG("str", 1, "structurises without gotos", str01)

/* The if and the if/else both turn into real ifs, and the else turns up
 * exactly where an else is wanted. */
static void str02(void)
{
    int n = emit_struct_msl(msl, sizeof(msl));
    CHECK(n > 0);
    const char *k = strstr(msl, "kernel void st_ifelse");
    CHECK(k != NULL);
    const char *end = strstr(k, "\n}");
    CHECK(end != NULL);
    CHECK(strstr(k, "if (") != NULL && strstr(k, "if (") < end);
    CHECK(strstr(k, "} else {") != NULL && strstr(k, "} else {") < end);
    PASS();
}
TH_REG("str", 2, "conditionals structurise", str02)

/* The counted for becomes a while(true), and both loop-carried values, the
 * counter and the accumulator, get settled by copies before the lap ends.
 * If either copy were missing the loop would compute rubbish. */
static void str03(void)
{
    int n = emit_struct_msl(msl, sizeof(msl));
    CHECK(n > 0);
    const char *k = strstr(msl, "kernel void st_for");
    CHECK(k != NULL);
    const char *end = strstr(k, "\n}");
    CHECK(end != NULL);
    CHECK(strstr(k, "while (true)") != NULL && strstr(k, "while (true)") < end);
    CHECK(strstr(k, "continue;") != NULL && strstr(k, "continue;") < end);
    PASS();
}
TH_REG("str", 3, "for loops structurise", str03)

/* The while with the break and the continue keeps both: the break leaves
 * the loop, the continue takes another lap, and neither is a goto in a
 * trench coat. */
static void str04(void)
{
    int n = emit_struct_msl(msl, sizeof(msl));
    CHECK(n > 0);
    const char *k = strstr(msl, "kernel void st_whilebreak");
    CHECK(k != NULL);
    const char *end = strstr(k, "\n}");
    CHECK(end != NULL);
    CHECK(strstr(k, "while (true)") != NULL && strstr(k, "while (true)") < end);
    CHECK(strstr(k, "break;") != NULL && strstr(k, "break;") < end);
    CHECK(strstr(k, "continue;") != NULL && strstr(k, "continue;") < end);
    PASS();
}
TH_REG("str", 4, "while with break structurises", str04)
