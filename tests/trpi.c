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

/* Counts non-overlapping occurrences of needle in obuf. A dropped statement
 * leaves the IR one instruction short rather than visibly wrong, so the count
 * is what catches it. */
static int occurs(const char *needle)
{
    int n = 0;
    for (const char *p = strstr(obuf, needle); p; p = strstr(p + 1, needle))
        n++;
    return n;
}

/* Writes src to build/<name> and returns the path, so a regression carries its
 * input with it instead of adding a fixture nobody can place later. */
static const char *scratch(const char *name, const char *src)
{
    static char path[256];
    snprintf(path, sizeof path, "build/%s", name);
    FILE *f = fopen(path, "w");
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    return path;
}

/* #5 promoted every parameter to an alloca so it could be written to, and
 * shipped without a test. Before it, only struct params were addressable and
 * everything else refused with E108; the promotion is invisible in the IR once
 * mem2reg folds it away, so what is checked here is the behaviour: a parameter
 * is a local initialised from the argument, and reading it after a write sees
 * the write. */
static void rpi04(void)
{
    static const char *const src =
        "__device__ int setc(int x) { x = 100; return x; }\n"
        "__device__ int padd(const int *p, int n) {\n"
        "    p = p + 2; n = n * 3; return p[0] + n;\n"
        "}\n"
        "__device__ int ploop(int a, int b) {\n"
        "    for (int i = 0; i < 3; i++) { a = a + b; b = b * 2; }\n"
        "    return a;\n"
        "}\n"
        "__global__ void kmain(int *out, const int *in) {\n"
        "    int i = threadIdx.x;\n"
        "    i = i + 1;\n"
        "    out[0] = setc(7) + padd(in, 1) + ploop(in[0], 1) + i;\n"
        "}\n";
    char cmd[512];
    const char *path = scratch("rpi04.cu", src);
    CHNE(path, NULL);

    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E108"), NULL);

    /* x = 100 wins over the incoming argument, so the body is the constant. */
    CHNE(strstr(obuf, "ret i32 100"), NULL);
    /* p = p + 2 has to move the pointer, not be dropped. */
    CHNE(strstr(obuf, "gep ptr<global, i32>, %0, 2"), NULL);
    /* Both params are rewritten every trip, so the loop head carries a phi for
     * each of them on top of the counter's. */
    CHEQ(occurs("phi i32"), 3);
    /* A __global__ entry point's parameter is no different. */
    CHNE(strstr(obuf, "__global__"), NULL);
    PASS();
}
TH_REG("rpi", 4, "#5 a parameter can be assigned to", rpi04)

/* The Triton frontend kept a name's value under the node that first bound it
 * and never moved the binding on, so every assignment after the first was
 * lowered and then dropped on the floor. Reading the name afterwards returned
 * the original value and the compiler said nothing. On an RTX 4060 Ti a kernel
 * doing v = v * 2.0 then v = v * 4.0 wrote x back unchanged. */
static void rpi05(void)
{
    static const char *const src =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kloc(x_ptr, out_ptr, n):\n"
        "    offs = tl.program_id(axis=0)\n"
        "    v = tl.load(x_ptr + offs)\n"
        "    v = v * 2.0\n"
        "    v = v * 4.0\n"
        "    tl.store(out_ptr + offs, v)\n";
    char cmd[512];
    const char *path = scratch("rpi05.py", src);
    CHNE(path, NULL);

    snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    /* Both multiplies survive. Dropping the rebind leaves the second one dead
     * and DCE takes it, so the count is one before the fix and two after. */
    CHEQ(occurs("fmul f32"), 2);
    PASS();
}
TH_REG("rpi", 5, "a Triton local keeps its latest value", rpi05)

/* Same root cause on a parameter, where it could not be fixed by moving the
 * binding: the name resolved straight back to the incoming argument, so both
 * n = n + 5 and n += 5 vanished. Writing to a parameter makes the name local
 * from that point, as it does in Python and in C. */
static void rpi06(void)
{
    static const char *const tmpl =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kpar(x_ptr, out_ptr, n):\n"
        "    offs = tl.program_id(axis=0)\n"
        "    %s\n"
        "    v = tl.load(x_ptr + n)\n"
        "    tl.store(out_ptr + offs, v)\n";
    static const char *const forms[] = { "n = n + 5", "n += 5", NULL };
    char src[1024], cmd[512];

    for (int i = 0; forms[i]; i++) {
        snprintf(src, sizeof src, tmpl, forms[i]);
        const char *path = scratch("rpi06.py", src);
        CHNE(path, NULL);
        snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, path);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        /* The add is the whole statement. Dropped, it is dead and DCE takes
         * it, and the load indexes off the raw argument instead. */
        CHNE(strstr(obuf, "add i32"), NULL);
    }
    PASS();
}
TH_REG("rpi", 6, "a Triton parameter can be assigned to", rpi06)

/* Rebinding across a loop back-edge needs a phi at the head, and the lowerer
 * builds one only for the counter. The accumulator every reduction is written
 * with therefore read its pre-loop value on every trip, and the sum came back
 * as whatever the last iteration computed, or as the initialiser. Nothing in
 * the pipeline objected. A refusal is the honest answer until the phi exists. */
static void rpi07(void)
{
    static const char *const tmpl =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kacc(x_ptr, out_ptr, n):\n"
        "    offs = tl.program_id(axis=0)\n"
        "    acc = 0.0\n"
        "    for i in range(0, 4):\n"
        "        %s\n"
        "    tl.store(out_ptr + offs, acc)\n";
    static const char *const forms[] = {
        "acc = acc + tl.load(x_ptr + i)", "acc += tl.load(x_ptr + i)", NULL
    };
    char src[1024], cmd[512];

    for (int i = 0; forms[i]; i++) {
        snprintf(src, sizeof src, tmpl, forms[i]);
        const char *path = scratch("rpi07.py", src);
        CHNE(path, NULL);
        snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, path);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, "E141"), NULL);
    }

    /* A name whose whole life is inside the loop body crosses no back-edge and
     * has to keep working, or the refusal has eaten the ordinary case with it.
     * Both statements survive: the load, then the multiply that rebinds it. */
    static const char *const inner =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kinner(x_ptr, out_ptr, n):\n"
        "    for i in range(0, 4):\n"
        "        t = tl.load(x_ptr + i)\n"
        "        t = t * 2.0\n"
        "        tl.store(out_ptr + i, t)\n";
    const char *ipath = scratch("rpi07b.py", inner);
    CHNE(ipath, NULL);
    snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, ipath);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E141"), NULL);
    CHNE(strstr(obuf, "fmul f32"), NULL);

    /* The rank-2 accumulator in a tl.dot kernel is scratch-backed and unrolled
     * rather than carried in a register, so it must still compile. */
    snprintf(cmd, sizeof cmd, "%s --triton --ir tests/tri_matmul_k.py", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    PASS();
}
TH_REG("rpi", 7, "a loop-carried Triton rebind refuses", rpi07)

/* The NVIDIA backend gave every i1 a %p, including values defined by adds,
 * loads, phis and atomics, none of which can write one. `out[0] = (a==0)||(b==0)`
 * came out as `st.global.u32 [%rd2], %p3`, which ptxas and the driver JIT both
 * reject, so the kernel could not load at all. i1esc.cu walks the ways out:
 * store, arithmetic, call, return, shared, atomic, float conversion and a phi. */
static int pbad(const char *ptx, char *where, int wsz)
{
    const char *p = ptx;
    while ((p = strstr(p, "%p")) != NULL) {
        const char *ls = p;
        while (ls > ptx && ls[-1] != '\n') ls--;
        const char *le = strchr(p, '\n');
        if (!le) le = p + strlen(p);

        while (ls < le && (*ls == ' ' || *ls == '\t')) ls++;

        if (strncmp(ls, ".reg", 4) == 0 || strncmp(ls, "@%p", 3) == 0
         || strncmp(ls, "setp.", 5) == 0 || strncmp(ls, "selp.", 5) == 0
         || strncmp(ls, "vote.", 5) == 0 || strncmp(ls, "mov.pred", 8) == 0) {
            p = le;
            continue;
        }
        int n = (int)(le - ls);
        if (n > wsz - 1) n = wsz - 1;
        memcpy(where, ls, (size_t)n);
        where[n] = '\0';
        return 1;
    }
    return 0;
}

static char pbuf[1 << 16];

static void rpi08(void)
{
    static const char *const modes[] = { "", "--no-mem2reg", NULL };
    char cmd[512], bad[192];

    for (int i = 0; modes[i] != NULL; i++) {
        snprintf(cmd, sizeof cmd,
                 "%s --nvidia-ptx %s tests/i1esc.cu -o build/rpi04.ptx",
                 BC_BIN, modes[i]);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);

        FILE *f = fopen("build/rpi04.ptx", "r");
        CHNE(f, NULL);
        size_t n = fread(pbuf, 1, sizeof pbuf - 1, f);
        pbuf[n] = '\0';
        fclose(f);

        if (pbad(pbuf, bad, (int)sizeof bad))
            printf("  %s: %s\n", modes[i][0] ? modes[i] : "default", bad);
        CHEQ(pbad(pbuf, bad, (int)sizeof bad), 0);
    }
    PASS();
}
TH_REG("rpi", 8, "a predicate never reaches a wider slot", rpi08)

/* A `bool` is one byte, so the stride between elements of a bool array is one
 * byte, and five size functions disagreed about that. width/8 is zero for i1:
 * the alloca sites clamped the zero to a minimum and the GEP sites did not, so
 * AMD multiplied the index by zero and every element of a bool array resolved
 * to the same address. x86-64 and RV64 substituted 4, Tensix refused.
 *
 * bir_bsz is the one answer now. These pin the answer rather than any one
 * backend's arithmetic. */
static void rpi09(void)
{
    int rc = th_run(BC_BIN " --ir tests/bstride.cu", obuf, (int)sizeof obuf);
    CHEQ(rc, 0);
    CHNE(strstr(obuf, "store i64 1,"), NULL);
    CHEQ(strstr(obuf, "store i64 4,"), NULL);
    PASS();
}
TH_REG("rpi", 9, "sizeof(bool) is one byte", rpi09)

/* The AMD case is the serious one because it was silent. A scaled GEP whose
 * stride is zero collapses a whole array onto element zero, and the shape is
 * a v_mul_lo_u32 against an immediate 0. */
static void rpi10(void)
{
    const char *p;
    int rc = th_run(BC_BIN " --amdgpu tests/bstride.cu", obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    for (p = obuf; (p = strstr(p, "v_mul_lo_u32")) != NULL; p++) {
        const char *nl = strchr(p, '\n');
        if (nl == NULL) break;
        CHEQ(nl - p >= 3 && nl[-3] == ',' && nl[-1] == '0', 0);
    }

    /* A bool[64] in LDS reserves 64 bytes, not the 4 the clamp used to give. */
    CHNE(strstr(obuf, "64 LDS bytes"), NULL);
    CHNE(strstr(obuf, "8 scratch bytes"), NULL);
    PASS();
}
TH_REG("rpi", 10, "a bool array does not stride by zero on AMD", rpi10)

/* Storage size, array stride and access width are three questions with one
 * answer for a bool. PTX strides by 1 and must therefore touch one byte;
 * a .u32 access at a one-byte stride writes over the next three elements. */
static void rpi11(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx -o build/rpi11.ptx tests/bstride.cu",
                    obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    FILE *f = fopen("build/rpi11.ptx", "r");
    CHNE(f, NULL);
    if (f == NULL) return;
    obuf[fread(obuf, 1, sizeof obuf - 1, f)] = '\0';
    fclose(f);

    CHNE(strstr(obuf, ", 1, %rd"), NULL);
    CHEQ(strstr(obuf, ", 4, %rd"), NULL);
    CHNE(strstr(obuf, "ld.global.u8"), NULL);
    CHNE(strstr(obuf, "st.global.u8"), NULL);
    CHNE(strstr(obuf, "ld.shared.u8"), NULL);
    CHNE(strstr(obuf, "st.local.u8"), NULL);
    CHEQ(strstr(obuf, "ld.global.u32"), NULL);
    CHEQ(strstr(obuf, "st.shared.u32"), NULL);
    PASS();
}
TH_REG("rpi", 11, "a bool load reads the byte it strides by", rpi11)

/* Sizing a struct by summing its fields is the same mistake in a different
 * hat: an array of { char; int; } strides by 8 on the host and strode by 5
 * here, so element i past the first landed inside its predecessor. One size
 * function that pads the way C does settles both. */
static void rpi12(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx -o build/rpi12.ptx tests/bpad.cu",
                    obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    FILE *f = fopen("build/rpi12.ptx", "r");
    CHNE(f, NULL);
    if (f == NULL) return;
    obuf[fread(obuf, 1, sizeof obuf - 1, f)] = '\0';
    fclose(f);

    CHNE(strstr(obuf, ", 8, %rd"), NULL);
    CHEQ(strstr(obuf, ", 5, %rd"), NULL);
    PASS();
}
TH_REG("rpi", 12, "a struct array strides by its padded size", rpi12)
/* looks_like_cast read `( ident )` before any prefix operator as a cast to a
 * type called ident, without ever asking whether ident named a type. Only `+`
 * and `-` are also infix, so `(a) + (b)` became a cast applied to `+(b)` and
 * the left operand vanished with no diagnostic. `-` left the tell, `sub 0, b`.
 *
 * The fixture stays on disk after a failure, so build/rpi13.cu names the case
 * that broke. */
static void rpi13(void)
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
        FILE *f = fopen("build/rpi13.cu", "w");
        CHNE(f, NULL);
        fprintf(f, "__device__ int f(int a, int b){ return %s; }\n", cs[i].ex);
        fclose(f);

        snprintf(cmd, sizeof cmd, "%s --ir build/rpi13.cu", BC_BIN);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, cs[i].ir), NULL);
    }
    PASS();
}
TH_REG("rpi", 13, "a parenthesised variable is not a cast", rpi13)

/* The other half of the same test. Tightening it must not cost a real cast,
 * so every shape a typedef name reaches the parser in is checked here, and
 * `(pair){...}` is one the loose rule never recognised at all. */
static void rpi14(void)
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
        FILE *f = fopen("build/rpi14.cu", "w");
        CHNE(f, NULL);
        fputs(cs[i].src, f);
        fclose(f);

        snprintf(cmd, sizeof cmd, "%s --ir build/rpi14.cu", BC_BIN);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, cs[i].ir), NULL);
    }
    PASS();
}
TH_REG("rpi", 14, "a typedef name still casts", rpi14)

/* A template type parameter is a type name for the body below it. It is not a
 * typedef and never reached the registry, so it survived on the loose rule
 * alone and (T)a + b would have started returning b. */
static void rpi15(void)
{
    static const char *const src =
        "template<typename T> __device__ T g(T a, T b){ return (T)a + b; }\n";
    char cmd[512];

    FILE *f = fopen("build/rpi15.cu", "w");
    CHNE(f, NULL);
    fputs(src, f);
    fclose(f);

    snprintf(cmd, sizeof cmd, "%s --parse build/rpi15.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "(binary +"), NULL);
    CHNE(strstr(obuf, "(cast"), NULL);
    PASS();
}
TH_REG("rpi", 15, "a template type parameter names a type", rpi15)
