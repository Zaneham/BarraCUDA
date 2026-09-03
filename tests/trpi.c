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

/* The BIR lexer read integers with strtol, and long is 32 bits on Windows, so
 * any constant above INT32_MAX saturated to 2147483647 without a word. A hash
 * multiplier came back as a different number and the kernel still ran. */
static void rpi03(void)
{
    static const char *const mod =
        "; Booth IR\n"
        "\n"
        "func @big(ptr<global, i32> %0, i32 %1) __global__ {\n"
        "entry:\n"
        "    %2 = thread_id.x\n"
        "    %3 = mul i32 %2, 2222261027\n"
        "    %4 = gep ptr<global, i32>, %0, %2\n"
        "    store i32 %3, %4\n"
        "    ret void\n"
        "}\n";
    char cmd[512];

    FILE *f = fopen("build/rpi03.bir", "w");
    CHNE(f, NULL);
    fputs(mod, f);
    fclose(f);

    snprintf(cmd, sizeof cmd, "%s --bir-in --ir --no-cfold build/rpi03.bir", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "2222261027"), NULL);
    CHEQ(strstr(obuf, "2147483647"), NULL);
    PASS();
}
TH_REG("rpi", 3, "a constant above INT32_MAX is not clamped", rpi03)

/* looks_like_cast read `( ident )` before any prefix operator as a cast to a
 * type called ident, without ever asking whether ident named a type. Only `+`
 * and `-` are also infix, so `(a) + (b)` became a cast applied to `+(b)` and
 * the left operand vanished with no diagnostic. `-` left the tell, `sub 0, b`.
 *
 * The fixture stays on disk after a failure, so build/rpi04.cu names the case
 * that broke. */
static void rpi04(void)
{
    static const struct { const char *ex; const char *ir; } cs[] = {
        { "(a) + (b)",        "add i32 %0, %1"     },
        { "(a) - (b)",        "sub i32 %0, %1"     },
        { "(a) + b",          "add i32 %0, %1"     },
        { "(b) + (a)",        "add i32 %1, %0"     },
        { "(a) + (a * 2)",    "add i32 %0, %2"     },
        { "(a) + a * 2",      "add i32 %0, %2"     },
        { "(a) + (a)",        "add i32 %0, %0"     },
        { "((a)) + (b)",      "add i32 %0, %1"     },
        { "(a) * (b)",        "mul i32 %0, %1"     },
        { "(a) & (b)",        "and i32 %0, %1"     },
        { "(a) / (b)",        "sdiv i32 %0, %1"    },
        { "(a) < (b)",        "icmp slt i32 %0, %1"},
        { "(a * 2) + (a)",    "add i32 %2, %0"     },
        { "(a + 1) + (a * 2)","add i32 %2, %3"     },
    };
    char cmd[512];

    for (size_t i = 0; i < sizeof cs / sizeof cs[0]; i++) {
        FILE *f = fopen("build/rpi04.cu", "w");
        CHNE(f, NULL);
        fprintf(f, "__device__ int f(int a, int b){ return %s; }\n", cs[i].ex);
        fclose(f);

        snprintf(cmd, sizeof cmd, "%s --ir build/rpi04.cu", BC_BIN);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, cs[i].ir), NULL);
    }
    PASS();
}
TH_REG("rpi", 4, "a parenthesised variable is not a cast", rpi04)

/* The other half of the same test. Tightening it must not cost a real cast,
 * so every shape a typedef name reaches the parser in is checked here, and
 * `(pair){...}` is one the loose rule never recognised at all. */
static void rpi05(void)
{
    static const struct { const char *src; const char *ir; } cs[] = {
        { "typedef int myint;\n"
          "__device__ int f(int a, int b){ return (myint)(a) + b; }\n",
          "add i32 %0, %1" },

        { "typedef int myint;\n"
          "__device__ int f(int a, int b){ (void)a; return (myint) + b; }\n",
          "ret i32 %1" },

        { "typedef int myint;\n"
          "__device__ int f(int a, int b){ (void)a; (void)b; return (myint)-1; }\n",
          "ret i32 4294967295" },

        { "typedef int myint;\n"
          "__device__ int f(void *p){ return *(myint*)p; }\n",
          "bitcast ptr<global, void> %0 to ptr<global, i32>" },

        { "typedef struct { int x; int y; } pair;\n"
          "__device__ int f(int a, int b){ pair p = (pair){ a, b }; return p.y; }\n",
          "store i32 %1, %6" },

        { "__device__ int f(int a, int b){ return (int)(uint32_t)-1 + a + b; }\n",
          "add i32 4294967295, %0" },

        { "using myint = int;\n"
          "__device__ int f(int a, int b){ return (myint)(a) + b; }\n",
          "add i32 %0, %1" },
    };
    char cmd[512];

    for (size_t i = 0; i < sizeof cs / sizeof cs[0]; i++) {
        FILE *f = fopen("build/rpi05.cu", "w");
        CHNE(f, NULL);
        fputs(cs[i].src, f);
        fclose(f);

        snprintf(cmd, sizeof cmd, "%s --ir build/rpi05.cu", BC_BIN);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, cs[i].ir), NULL);
    }
    PASS();
}
TH_REG("rpi", 5, "a typedef name still casts", rpi05)

/* A template type parameter is a type name for the body below it. It is not a
 * typedef and never reached the registry, so it survived on the loose rule
 * alone and (T)a + b would have started returning b. */
static void rpi06(void)
{
    static const char *const src =
        "template<typename T> __device__ T g(T a, T b){ return (T)a + b; }\n";
    char cmd[512];

    FILE *f = fopen("build/rpi06.cu", "w");
    CHNE(f, NULL);
    fputs(src, f);
    fclose(f);

    snprintf(cmd, sizeof cmd, "%s --parse build/rpi06.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "(binary +"), NULL);
    CHNE(strstr(obuf, "(cast"), NULL);
    PASS();
}
TH_REG("rpi", 6, "a template type parameter names a type", rpi06)
