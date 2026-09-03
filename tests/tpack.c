/* tpack.c -- template parameter packs
 * The grammar, the standard's placement rules, and what a pack lowers to. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

static int wsrc(const char *path, const char *text)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fputs(text, f);
    fclose(f);
    return 0;
}

static int run_ir(const char *text)
{
    if (wsrc("pack_tmp.cu", text) != 0) return -1;
    int rc = th_run(BC_BIN " --ir pack_tmp.cu", obuf, TH_BUFSZ);
    remove("pack_tmp.cu");
    return rc;
}

static int run_parse(const char *text)
{
    if (wsrc("pack_tmp.cu", text) != 0) return -1;
    int rc = th_run(BC_BIN " --parse pack_tmp.cu", obuf, TH_BUFSZ);
    remove("pack_tmp.cu");
    return rc;
}

/* ---- packs: the fixture compiles ---- */

static void pck01(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "error") == NULL);
    PASS();
}
TH_REG("pck", 1, "the pack fixture reaches BIR", pck01)

/* ---- packs: a pack parameter becomes one BIR parameter per member ---- */

static void pck02(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf,
        "func @pk_sum(ptr<global, f32> %0, f32 %1, f32 %2, f32 %3, f32 %4)")
        != NULL);
    PASS();
}
TH_REG("pck", 2, "a pack widens the parameter list", pck02)

/* ---- packs: two lengths are two functions ---- */

static void pck03(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @pk_sum$1(") != NULL);
    PASS();
}
TH_REG("pck", 3, "two specialisations do not share a symbol", pck03)

/* ---- packs: sizeof... is a constant by lowering ---- */

static void pck04(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__global__ void k(int *o, A... a)\n"
        "{ o[0] = (int)sizeof...(a) + (int)sizeof...(A); }\n"
        "int main(void){ int *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, 1, 2, 3); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "store i32 6,") != NULL);
    PASS();
}
TH_REG("pck", 4, "sizeof... folds to the pack length", pck04)

/* ---- packs: a fold expands to the operator chain ---- */

static void pck05(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__global__ void k(float *o, A... a) { o[0] = (0.0f + ... + (float)a); }\n"
        "int main(void){ float *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, 1.0f, 2.0f, 3.0f); return 0; }\n");
    CHEQ(rc, 0);
    /* seed plus one add per member */
    int n = 0;
    for (const char *p = obuf; (p = strstr(p, "fadd")) != NULL; p++) n++;
    CHEQ(n, 3);
    PASS();
}
TH_REG("pck", 5, "a binary left fold becomes one op per member", pck05)

/* ---- packs: an expansion widens a call ---- */

static void pck06(void)
{
    int rc = th_run(BC_BIN " --ir --no-sroa tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf,
        "func @pk_call(ptr<global, f32> %0, f32 %1, f32 %2, f32 %3)") != NULL);
    PASS();
}
TH_REG("pck", 6, "an expansion in a call supplies every member", pck06)

/* ---- packs: || keeps its short circuit ---- */

static void pck07(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *f = strstr(obuf, "func @pk_any");
    CHECK(f != NULL);
    CHECK(strstr(f, "br_cond") != NULL);
    PASS();
}
TH_REG("pck", 7, "a logical fold branches rather than chains", pck07)

/* ---- packs: an unexpanded pack is ill-formed ---- */

static void pck08(void)
{
    int rc = run_parse(
        "template<typename... A>\n"
        "__device__ int k(A... a) { return f(a); }\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E083]") != NULL);
    PASS();
}
TH_REG("pck", 8, "a pack used without an expansion is rejected", pck08)

/* ---- packs: [temp.param]/14 placement ---- */

static void pck09(void)
{
    int rc = run_parse(
        "template<typename... A, typename B> struct S { B v; };\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E028]") != NULL);
    PASS();
}
TH_REG("pck", 9, "a class template pack must come last", pck09)

/* ---- packs: no default argument on a pack ---- */

static void pck10(void)
{
    int rc = run_parse("template<typename... A = int> struct S { int v; };\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E029]") != NULL);
    PASS();
}
TH_REG("pck", 10, "a pack takes no default argument", pck10)

/* ---- packs: pack indexing refuses by name ---- */

static void pck11(void)
{
    int rc = run_parse(
        "template<typename... A>\n"
        "__device__ int k(A... a) { return (int)a...[0]; }\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E030]") != NULL);
    CHECK(strstr(obuf, "pack indexing") != NULL);
    PASS();
}
TH_REG("pck", 11, "pack indexing refuses by name", pck11)

/* ---- packs: [dcl.fct]/27 keeps the C ellipsis ---- */

static void pck12(void)
{
    int rc = run_ir(
        "__device__ int v(int a, ...) { return a; }\n"
        "__global__ void k(int *o) { o[0] = v(1); }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "error") == NULL);
    PASS();
}
TH_REG("pck", 12, "a C ellipsis is still a C ellipsis", pck12)

/* ---- packs: a forwarding-reference pack parses ---- */

static void pck13(void)
{
    int rc = run_parse(
        "template<typename... A>\n"
        "__host__ __device__ constexpr inline void u(A&&...) noexcept {}\n");
    (void)rc;
    CHECK(strstr(obuf, "error[") == NULL);
    PASS();
}
TH_REG("pck", 13, "an unnamed forwarding pack parses", pck13)

/* ---- packs: the pack need not be last in a function template ---- */

static void pck14(void)
{
    int rc = run_parse(
        "template<int... N, typename T>\n"
        "__device__ T k(T x) { return x; }\n");
    (void)rc;
    CHECK(strstr(obuf, "error[") == NULL);
    PASS();
}
TH_REG("pck", 14, "a function template may declare after a pack", pck14)

/* ---- packs: an overload set is chosen by arity ---- */

static void pck15(void)
{
    /* Both overloads inline, so only the emitted constant tells them apart. */
    CHECK(wsrc("pack_tmp.cu",
        "__device__ int g(int a, int b) { return a * 1000 + b; }\n"
        "__device__ int g(int a) { return a + 7; }\n"
        "__global__ void k(int *o) { o[0] = g(5); }\n"
        "int main(void){ int *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o); return 0; }\n") == 0);
    int rc = th_run(BC_BIN " --nvidia-ptx pack_tmp.cu -o pack_tmp.ptx",
                    obuf, TH_BUFSZ);
    remove("pack_tmp.cu");
    CHEQ(rc, 0);
    FILE *pf = fopen("pack_tmp.ptx", "rb");
    CHECK(pf != NULL);
    size_t got = fread(obuf, 1, TH_BUFSZ - 1, pf);
    obuf[got] = 0;
    fclose(pf);
    remove("pack_tmp.ptx");
    CHECK(strstr(obuf, "mov.u32 %r1, 12;") != NULL);
    PASS();
}
TH_REG("pck", 15, "a call binds to the overload with its arity", pck15)

/* ---- packs: a pack that cannot be deduced refuses ---- */

static void pck16(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__global__ void k(float *o, A... a) { o[0] = (0.0f + ... + (float)a); }\n"
        "int main(void){ float *o; float x = 1.0f; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, x); return 0; }\n");
    (void)rc;
    CHECK(strstr(obuf, "E030") != NULL);
    PASS();
}
TH_REG("pck", 16, "an undeducible pack refuses, never guesses", pck16)
