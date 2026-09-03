/* tmtu.c -- several .cu files in one compile
 *
 * The thing worth testing here is not that the driver loops over argv. It is
 * that two files can hold their own `static` helper of the same name and each
 * kernel still calls its own, because getting that wrong builds a program
 * nobody wrote and says nothing about it. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- Two files reach the backend ---- */

static void mtu01(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_sta.cu tests/tu_stb.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @tu_ka") != NULL);
    CHECK(strstr(obuf, "func @tu_kb") != NULL);
    PASS();
}
TH_REG("mtu", 1, "two files lower into one module", mtu01)

/* ---- A static helper belongs to its own file ---- */

static void mtu02(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_sta.cu tests/tu_stb.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @scale__0") != NULL);
    CHECK(strstr(obuf, "func @scale__1") != NULL);
    CHECK(strstr(obuf, "call i32 @scale__0") != NULL);
    CHECK(strstr(obuf, "call i32 @scale__1") != NULL);
    PASS();
}
TH_REG("mtu", 2, "each file calls its own static helper", mtu02)

/* The bodies differ by 1 versus 100, so if the two ever collapsed into one
 * the surviving add would give it away. */
static void mtu03(void)
{
    int rc = th_run(BC_BIN " --ir --no-cfold tests/tu_sta.cu tests/tu_stb.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "add i32 %0, 1") != NULL);
    CHECK(strstr(obuf, "add i32 %0, 100") != NULL);
    PASS();
}
TH_REG("mtu", 3, "the two static bodies both survive", mtu03)

/* ---- Calling across files ---- */

static void mtu04(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_lib.cu tests/tu_use.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @tu_mul7") != NULL);
    CHECK(strstr(obuf, "func @tu_kuse") != NULL);
    PASS();
}
TH_REG("mtu", 4, "a __device__ function crosses files", mtu04)

/* Symbols become visible in command-line order, so the caller listed first
 * finds nothing. Refusing is the point: the alternative is a call to whatever
 * happened to be lying around. */
static void mtu05(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_use.cu tests/tu_lib.cu",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "E105") != NULL);
    PASS();
}
TH_REG("mtu", 5, "a callee listed later is not found", mtu05)

/* ---- One symbol, two definitions ---- */

static void mtu06(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_lib.cu tests/tu_dup.cu",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "E126") != NULL);
    PASS();
}
TH_REG("mtu", 6, "a kernel defined twice is refused", mtu06)

/* ---- A header both files include ---- */

static void mtu07(void)
{
    int rc = th_run(BC_BIN " --ir -Itests tests/tu_h1.cu tests/tu_h2.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @tu_sq") != NULL);
    CHECK(strstr(obuf, "func @tu_sq__0") == NULL);
    CHECK(strstr(obuf, "3 functions, 1 globals") != NULL);
    PASS();
}
TH_REG("mtu", 7, "an inline header gives one copy, not two", mtu07)

/* ---- Both kernels reach an emitter ---- */

static void mtu08(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx tests/tu_sta.cu tests/tu_stb.cu"
                    " -o mtu08.ptx", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "2 kernels") != NULL);
    remove("mtu08.ptx");
    PASS();
}
TH_REG("mtu", 8, "two files give two PTX entry points", mtu08)

static void mtu09(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin tests/tu_sta.cu tests/tu_stb.cu"
                    " -o mtu09.hsaco", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "2 kernels") != NULL);
    remove("mtu09.hsaco");
    PASS();
}
TH_REG("mtu", 9, "two files give two AMD kernels", mtu09)

/* ---- One file is still one file ---- */

/* Nothing is qualified when there is nothing to keep apart, so a single-file
 * compile emits the names it always did. */
static void mtu10(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_sta.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @scale") != NULL);
    CHECK(strstr(obuf, "scale__0") == NULL);
    PASS();
}
TH_REG("mtu", 10, "one file keeps unqualified names", mtu10)

/* ---- The frontends that read one document say so ---- */

static void mtu11(void)
{
    int rc = th_run(BC_BIN " --triton --ir tests/tri_vadd.py tests/tri_vadd.py",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "only one input file") != NULL);
    PASS();
}
TH_REG("mtu", 11, "--triton refuses a second file", mtu11)

/* ---- The same template in both files ---- */

/* A template lives in a header, so both files carry their own copy of the
 * declaration and both ask for the same instantiation. C++ says that is one
 * function, and one is what comes out. */
static void mtu12(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_t1.cu tests/tu_t2.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @tu_tsc") != NULL);
    CHECK(strstr(obuf, "1 functions") != NULL);
    PASS();
}
TH_REG("mtu", 12, "one instantiation, not one per file", mtu12)

/* ---- A global declared in one file and defined in another ---- */

/* Globals are looked up where they are used rather than where the file is
 * read, so unlike a call these do not mind which order the files came in. */
static void mtu13(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_g1.cu tests/tu_g2.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "1 globals") != NULL);
    CHECK(strstr(obuf, "func @tu_kg1") != NULL);
    CHECK(strstr(obuf, "func @tu_kg2") != NULL);
    PASS();
}
TH_REG("mtu", 13, "extern __constant__ resolves across files", mtu13)

static void mtu14(void)
{
    int rc = th_run(BC_BIN " --ir tests/tu_g2.cu tests/tu_g1.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "1 globals") != NULL);
    PASS();
}
TH_REG("mtu", 14, "and resolves in either order", mtu14)
