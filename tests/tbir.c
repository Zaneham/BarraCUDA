/* tbir.c -- the BIR text frontend, which is how anything outside this tree
 * reaches the backends without linking against it. */

#include "tharns.h"

static char abuf[1 << 16];
static char bbuf[1 << 16];

/* Under build/ so a test run does not scatter files through the repo root. */
#define TMP "build/tbir_tmp.bir"

/* --ir adds a stats footer that is not part of the module, and the module
 * always ends at its last brace. */
static void trim(char *s)
{
    char *last = strrchr(s, '}');
    if (last != NULL) last[1] = '\0';
}

static int write_tmp(const char *text)
{
    FILE *f = fopen(TMP, "w");
    if (f == NULL) return 0;
    fputs(text, f);
    fputc('\n', f);
    fclose(f);
    return 1;
}

/* Print a module, read it back, print it again. The two must agree. */
static int roundtrip(const char *src)
{
    char cmd[512];

    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, src);
    if (th_run(cmd, abuf, (int)sizeof abuf) != 0) return -1;
    trim(abuf);
    if (!write_tmp(abuf)) return -2;

    snprintf(cmd, sizeof cmd, "%s --bir-in --ir %s", BC_BIN, TMP);
    if (th_run(cmd, bbuf, (int)sizeof bbuf) != 0) return -3;
    trim(bbuf);

    return strcmp(abuf, bbuf) == 0 ? 0 : 1;
}

static void bir01(void)
{
    CHEQ(roundtrip("examples/cmake/vadd.cu"), 0);
    PASS();
}
TH_REG("bir", 1, "a kernel round-trips through BIR text", bir01)

/* A whole float prints as "1" because %g drops the point, so only the
 * instruction type says it is not an integer. Read back as an integer it still
 * round-trips as text and still compiles, but PTX gets 0f00000001, which is
 * 1.4e-45 rather than 1.0. The emitted constant is the only honest witness. */
static const char *const whole_mod =
    "; Booth IR\n"
    "\n"
    "func @w(ptr<global, f32> %0, i32 %1) __global__ {\n"
    "entry:\n"
    "    %2 = thread_id.x\n"
    "    %3 = gep ptr<global, f32>, %0, %2\n"
    "    %4 = load f32, %3\n"
    "    %5 = fadd f32 %4, 1\n"
    "    store f32 %5, %3\n"
    "    ret void\n"
    "}";

static void bir02(void)
{
    char cmd[512];

    CHNE(write_tmp(whole_mod), 0);
    snprintf(cmd, sizeof cmd, "%s --bir-in --nvidia-ptx %s -o build/tbir_whole.ptx",
             BC_BIN, TMP);
    CHEQ(th_run(cmd, bbuf, (int)sizeof bbuf), 0);

    FILE *f = fopen("build/tbir_whole.ptx", "r");
    CHNE(f, NULL);
    size_t n = fread(abuf, 1, sizeof abuf - 1, f);
    abuf[n] = '\0';
    fclose(f);

    CHNE(strstr(abuf, "0f3F800000"), NULL);   /* 1.0f */
    CHEQ(strstr(abuf, "0f00000001"), NULL);   /* the integer 1, reinterpreted */
    PASS();
}
TH_REG("bir", 2, "a whole float immediate stays a float", bir02)

static const char *const imm_mod =
    "; Booth IR\n"
    "\n"
    "func @s(ptr<global, f32> %0, i32 %1) __global__ {\n"
    "entry:\n"
    "    %2 = thread_id.x\n"
    "    %3 = gep ptr<global, f32>, %0, %2\n"
    "    %4 = load f32, %3\n"
    "    %5 = fmul f32 %4, 2.5\n"
    "    %6 = fadd f32 %5, 0.125\n"
    "    store f32 %6, %3\n"
    "    ret void\n"
    "}";

static void bir03(void)
{
    char cmd[512];

    CHNE(write_tmp(imm_mod), 0);
    snprintf(cmd, sizeof cmd, "%s --bir-in --ir %s", BC_BIN, TMP);
    CHEQ(th_run(cmd, bbuf, (int)sizeof bbuf), 0);

    CHNE(strstr(bbuf, "fmul f32 %4, 2.5"), NULL);
    CHNE(strstr(bbuf, "fadd f32 %5, 0.125"), NULL);
    PASS();
}
TH_REG("bir", 3, "fractional immediates keep their value", bir03)

/* A parse failure must say where and produce nothing. */
static void bir04(void)
{
    char cmd[512];

    CHNE(write_tmp("func @broken( {\nentry:\n  ret void\n}"), 0);
    snprintf(cmd, sizeof cmd, "%s --bir-in --ir %s", BC_BIN, TMP);
    CHNE(th_run(cmd, bbuf, (int)sizeof bbuf), 0);
    CHNE(strstr(bbuf, TMP), NULL);
    PASS();
}
TH_REG("bir", 4, "a bad module is refused with its location", bir04)

/* parse_type recurses per ptr<> level, so nesting is stack depth an input file
 * gets to choose. The pool guard never fired because the stack went first. */
static void bir05(void)
{
    static char deep[16384];
    char cmd[512];
    int n = 0;

    n += snprintf(deep + n, sizeof deep - (size_t)n, "; Booth IR\n\nfunc @d(");
    for (int i = 0; i < 400; i++)
        n += snprintf(deep + n, sizeof deep - (size_t)n, "ptr<global, ");
    n += snprintf(deep + n, sizeof deep - (size_t)n, "f32");
    for (int i = 0; i < 400; i++)
        n += snprintf(deep + n, sizeof deep - (size_t)n, ">");
    snprintf(deep + n, sizeof deep - (size_t)n,
             " %%0) __global__ {\nentry:\n    ret void\n}");

    CHNE(write_tmp(deep), 0);
    snprintf(cmd, sizeof cmd, "%s --bir-in --ir %s", BC_BIN, TMP);
    CHNE(th_run(cmd, bbuf, (int)sizeof bbuf), 0);
    CHNE(strstr(bbuf, "nested too deeply"), NULL);
    PASS();
}
TH_REG("bir", 5, "a deeply nested type is refused, not a crash", bir05)

/* What a loop and an accumulator lower to before mem2reg gets them. */
static const char *const mem_mod =
    "; Booth IR\n"
    "\n"
    "func @m(ptr<global, f32> %0, i32 %1) __global__ {\n"
    "entry:\n"
    "    %2 = alloca ptr<private, f32>\n"
    "    store f32 0.5, %2\n"
    "    %4 = load f32, %2\n"
    "    %5 = fcmp olt f32 %4, 1\n"
    "    br_cond %5, t, e merge e\n"
    "\n"
    "t:\n"
    "    store f32 %4, %0\n"
    "    br e\n"
    "\n"
    "e:\n"
    "    ret void\n"
    "}";

static void bir07(void)
{
    char cmd[512];

    CHNE(write_tmp(mem_mod), 0);
    snprintf(cmd, sizeof cmd,
             "%s --bir-in --ir --no-mem2reg --no-cfold --no-dce --no-sroa %s",
             BC_BIN, TMP);
    CHEQ(th_run(cmd, bbuf, (int)sizeof bbuf), 0);
    trim(bbuf);
    CHNE(strstr(bbuf, "alloca ptr<private, f32>"), NULL);
    CHNE(strstr(bbuf, "fcmp olt f32 %4, 1"), NULL);

    /* And mem2reg still promotes it, which is why the frontend emits no phi. */
    snprintf(cmd, sizeof cmd, "%s --bir-in --ir %s", BC_BIN, TMP);
    CHEQ(th_run(cmd, abuf, (int)sizeof abuf), 0);
    CHEQ(strstr(abuf, "alloca"), NULL);
    PASS();
}
TH_REG("bir", 7, "alloca and fcmp read back, mem2reg promotes", bir07)

/* The path the OCaml emitter actually takes. */
static void bir06(void)
{
    char cmd[512];

    CHNE(write_tmp(imm_mod), 0);
    snprintf(cmd, sizeof cmd, "%s --bir-in --nvidia-ptx %s -o build/tbir_tmp.ptx",
             BC_BIN, TMP);
    CHEQ(th_run(cmd, bbuf, (int)sizeof bbuf), 0);
    CHNE(th_exist("build/tbir_tmp.ptx"), 0);
    PASS();
}
TH_REG("bir", 6, "BIR text reaches a backend", bir06)
