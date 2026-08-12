/* tmlir.c -- the MLIR path, seen from outside.
 *
 * Two halves. Reading is about the vendored reader surviving being embedded:
 * Booth's main and corec's heap coexisting, the boundary in mlir_fe.c holding,
 * and a file we cannot read failing loudly rather than quietly producing an
 * empty module. Lowering is about src/mlir/lower.c putting the right opcode in
 * the right place, and refusing by name when it cannot. His parser has its own
 * suite upstream and is not retested here. */

#include "tharns.h"
#include "mlir_fe.h"

static char obuf[TH_BUFSZ];

static int ml_run(const char *args)
{
    char cmd[TH_BUFSZ];
    snprintf(cmd, TH_BUFSZ, BC_BIN " %s", args);
    return th_run(cmd, obuf, TH_BUFSZ);
}

/* ---- Reading, through the binary ---- */

static void mlr01(void)
{
    int rc = ml_run("--mlir --pp tests/mlir/a.mlir");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func.func") != NULL);
    PASS();
}
TH_REG("mlr", 1, "smallest module parses", mlr01)

static void mlr02(void)
{
    /* Triton IR is the reason the dialect matters here: Booth already has a
     * Triton frontend, so this file is the one that will eventually go down
     * both paths and have its BIR diffed. */
    int rc = ml_run("--mlir --pp tests/mlir/triton_mm.ttir");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "tt.") != NULL);
    PASS();
}
TH_REG("mlr", 2, "triton matmul dialect", mlr02)

static void mlr03(void)
{
    /* scf, cf, memref and arith all appear across the corpus. If any one of
     * them regressed, one of these would stop parsing. */
    static const char *const files[] = {
        "tests/mlir/simple.mlir", "tests/mlir/effect.mlir",
        "tests/mlir/b.mlir", "tests/mlir/c.mlir", "tests/mlir/d.mlir",
        "tests/mlir/t1.mlir", "tests/mlir/t2.mlir", "tests/mlir/t3.mlir",
        "tests/mlir/add1.ttir", "tests/mlir/add_kernel.ttir",
        "tests/mlir/conv2d.ttir", "tests/mlir/sumrow.ttir",
        "tests/mlir/matmul1.ttir",
        "tests/mlir/chunked_cross_entropy_forward.ttir",
        "tests/mlir/mix.mlir", "tests/mlir/binops.mlir",
        "tests/mlir/floats.mlir", "tests/mlir/reject.mlir",
        "tests/mlir/consts.mlir",
    };
    char cmd[TH_BUFSZ];
    size_t i;

    for (i = 0; i < sizeof files / sizeof files[0]; i++) {
        snprintf(cmd, TH_BUFSZ, "--mlir --pp %s", files[i]);
        CHEQ(ml_run(cmd), 0);
    }
    PASS();
}
TH_REG("mlr", 3, "whole corpus parses", mlr03)

/* ---- Lowering to BIR ---- */

static void mlr12(void)
{
    int rc = ml_run("--mlir --ir tests/mlir/mix.mlir");

    CHEQ(rc, 0);
    /* Arguments become parameters, the constant folds into the operand slot
     * rather than taking an instruction, and the compare keeps its predicate. */
    CHECK(strstr(obuf, "func @mix(i32 %0, i32 %1)") != NULL);
    CHECK(strstr(obuf, "add i32 %0, %1") != NULL);
    CHECK(strstr(obuf, "mul i32 %2, 7") != NULL);
    CHECK(strstr(obuf, "icmp slt") != NULL);
    CHECK(strstr(obuf, "zext i1") != NULL);
    CHECK(strstr(obuf, "ret i32") != NULL);
    PASS();
}
TH_REG("mlr", 12, "arith lowers to BIR", mlr12)

static void mlr13(void)
{
    /* Every integer binop in the subset in one function, so a mapping that
     * goes to the wrong opcode shows up as the wrong mnemonic here rather
     * than as wrong arithmetic three backends later. */
    int rc = ml_run("--mlir --ir tests/mlir/binops.mlir");

    CHEQ(rc, 0);
    CHECK(strstr(obuf, "add i32") != NULL);
    CHECK(strstr(obuf, "sub i32") != NULL);
    CHECK(strstr(obuf, "mul i32") != NULL);
    CHECK(strstr(obuf, "sdiv i32") != NULL);
    CHECK(strstr(obuf, "udiv i32") != NULL);
    CHECK(strstr(obuf, "srem i32") != NULL);
    CHECK(strstr(obuf, "urem i32") != NULL);
    CHECK(strstr(obuf, "and i32") != NULL);
    CHECK(strstr(obuf, "or i32") != NULL);
    CHECK(strstr(obuf, "xor i32") != NULL);
    CHECK(strstr(obuf, "shl i32") != NULL);
    CHECK(strstr(obuf, "lshr i32") != NULL);
    CHECK(strstr(obuf, "ashr i32") != NULL);
    PASS();
}
TH_REG("mlr", 13, "every integer binop maps", mlr13)

static void mlr14(void)
{
    int rc = ml_run("--mlir --ir tests/mlir/floats.mlir");

    CHEQ(rc, 0);
    CHECK(strstr(obuf, "fadd f32") != NULL);
    CHECK(strstr(obuf, "fsub f32") != NULL);
    CHECK(strstr(obuf, "fmul f32") != NULL);
    CHECK(strstr(obuf, "fdiv f32") != NULL);
    PASS();
}
TH_REG("mlr", 14, "float binops map", mlr14)

static void mlr18(void)
{
    /* The reader builds `1 : f32` as an integer attribute, so a lowering that
     * trusts the attribute kind rather than the result type writes the bits
     * 0x1 where 1.0f belongs, and compiles quietly. If that comes back the
     * fold below stops folding, because the constant no longer matches its
     * type. */
    int rc = ml_run("--mlir --ir tests/mlir/consts.mlir");

    CHEQ(rc, 0);
    CHECK(strstr(obuf, "ret f32 2.5") != NULL);
    PASS();
}
TH_REG("mlr", 18, "integer literal on a float type", mlr18)

static void mlr15(void)
{
    /* The whole point of the boundary. An op we do not handle must stop the
     * lowering and name itself, not vanish and leave a function that computes
     * something else. */
    int rc = ml_run("--mlir --ir tests/mlir/reject.mlir");

    CHNE(rc, 0);
    CHECK(strstr(obuf, "math.exp") != NULL);
    CHECK(strstr(obuf, "outside the accepted subset") != NULL);
    CHECK(strstr(obuf, "nothing emitted") != NULL);
    /* And nothing was printed as if it had worked. */
    CHECK(strstr(obuf, "Booth IR") == NULL);
    PASS();
}
TH_REG("mlr", 15, "unhandled op is refused loudly", mlr15)

static void mlr16(void)
{
    /* MLIR in, machine code out, with no LLVM anywhere. The backends are
     * covered elsewhere; what this pins is that the BIR the lowering builds is
     * well formed enough for them to accept it. */
    CHEQ(ml_run("--mlir --cpu -o tests/mlir/out.o tests/mlir/mix.mlir"), 0);
    CHEQ(ml_run("--mlir --rv64 -o tests/mlir/out.o tests/mlir/mix.mlir"), 0);
    CHEQ(ml_run("--mlir --nvidia-ptx -o tests/mlir/out.ptx tests/mlir/mix.mlir"), 0);
    CHEQ(ml_run("--mlir --amdgpu-bin --gfx1100 -o tests/mlir/out.bin tests/mlir/mix.mlir"), 0);
    remove("tests/mlir/out.o");
    remove("tests/mlir/out.ptx");
    remove("tests/mlir/out.bin");
    PASS();
}
TH_REG("mlr", 16, "MLIR reaches every backend", mlr16)

static void mlr17(void)
{
    /* --pp reprints instead of lowering, which is how you tell a misreading
     * from a bad file. It must not also emit IR. */
    int rc = ml_run("--mlir --pp tests/mlir/mix.mlir");

    CHEQ(rc, 0);
    CHECK(strstr(obuf, "arith.addi") != NULL);
    CHECK(strstr(obuf, "Booth IR") == NULL);
    PASS();
}
TH_REG("mlr", 17, "reprint does not lower", mlr17)

/* ---- Through the API ---- */

static void mlr04(void)
{
    static const char src[] =
        "module {\n"
        "  func.func @main() -> i32 {\n"
        "    %0 = arith.constant 42 : i32\n"
        "    return %0 : i32\n"
        "  }\n"
        "}\n";
    ml_ctx_t *C = ml_open(1u << 20);

    CHECK(C != NULL);
    CHEQ(ml_parse(C, src, sizeof src - 1), 0);
    ml_close(C);
    PASS();
}
TH_REG("mlr", 4, "parse from memory", mlr04)

static void mlr05(void)
{
    /* Two contexts at once. corec brings its heap up exactly once however
     * many times ml_open is called, and this is the only thing that would
     * notice if that ever stopped being true. */
    ml_ctx_t *a = ml_open(1u << 20);
    ml_ctx_t *b = ml_open(1u << 20);

    CHECK(a != NULL);
    CHECK(b != NULL);
    CHECK(a != b);
    CHEQ(ml_parse(a, "module { }\n", 11), 0);
    CHEQ(ml_parse(b, "module { }\n", 11), 0);
    ml_close(b);
    ml_close(a);
    PASS();
}
TH_REG("mlr", 5, "two contexts, one heap", mlr05)

static void mlr10(void)
{
    /* func.func used to build its argument values and only attach them to the
     * entry block after the body had already been parsed, so any body that
     * named an argument failed with "undefined SSA value". Every Fortran or
     * Triton function we will ever be handed has arguments, so this is the
     * shape that matters most. Fixed in the vendored copy. */
    static const char src[] =
        "module {\n"
        "  func.func @f(%arg0: i32, %arg1: i32) -> i32 {\n"
        "    %0 = arith.addi %arg0, %arg1 : i32\n"
        "    return %0 : i32\n"
        "  }\n"
        "}\n";
    ml_ctx_t *C = ml_open(1u << 20);
    FILE *f;
    char back[512];
    size_t n;

    CHECK(C != NULL);
    CHEQ(ml_parse(C, src, sizeof src - 1), 0);

    /* Parsing is not enough on its own. If the arguments were dropped rather
     * than bound, the reprint would come back without them. */
    f = tmpfile();
    CHECK(f != NULL);
    CHEQ(ml_echo(C, f), 0);
    rewind(f);
    n = fread(back, 1, sizeof back - 1, f);
    back[n] = '\0';
    fclose(f);
    ml_close(C);

    CHECK(strstr(back, "%arg0") != NULL);
    CHECK(strstr(back, "%arg1") != NULL);
    CHECK(strstr(back, "arith.addi") != NULL);
    PASS();
}
TH_REG("mlr", 10, "func.func binds its arguments", mlr10)

static void mlr11(void)
{
    /* Arguments must not leak out of the function that declared them. Two
     * functions, the second naming the first's argument, has to fail. */
    static const char src[] =
        "module {\n"
        "  func.func @f(%arg0: i32) -> i32 {\n"
        "    return %arg0 : i32\n"
        "  }\n"
        "  func.func @g() -> i32 {\n"
        "    return %arg0 : i32\n"
        "  }\n"
        "}\n";
    ml_ctx_t *C = ml_open(1u << 20);

    CHECK(C != NULL);
    CHNE(ml_parse(C, src, sizeof src - 1), 0);
    ml_close(C);
    PASS();
}
TH_REG("mlr", 11, "arguments stay in their function", mlr11)

static void mlr09(void)
{
    /* The reader interns types process-wide and hands out handles into the
     * arena that was current when it first saw each one. Left alone, a
     * context closed here would leave the next parse walking freed memory,
     * which showed up as an intermittent fault three tests later rather than
     * anywhere near the cause. Same types every round, deliberately. */
    static const char src[] =
        "module {\n"
        "  func.func @f(%a: i32, %b: i64) -> i32 {\n"
        "    return %a : i32\n"
        "  }\n"
        "}\n";
    int i;

    for (i = 0; i < 32; i++) {
        ml_ctx_t *C = ml_open(1u << 20);

        CHECK(C != NULL);
        CHEQ(ml_parse(C, src, sizeof src - 1), 0);
        ml_close(C);
    }
    PASS();
}
TH_REG("mlr", 9, "types survive a closed context", mlr09)

static void mlr06(void)
{
    /* Reprinting is how a lowering gets debugged, so it has to survive a
     * round trip rather than merely not crash. */
    static const char src[] = "module {\n  func.func @f() {\n    return\n  }\n}\n";
    ml_ctx_t *C = ml_open(1u << 20);
    FILE *f;
    char back[512];
    size_t n;

    CHECK(C != NULL);
    CHEQ(ml_parse(C, src, sizeof src - 1), 0);

    f = tmpfile();
    CHECK(f != NULL);
    CHEQ(ml_echo(C, f), 0);
    rewind(f);
    n = fread(back, 1, sizeof back - 1, f);
    back[n] = '\0';
    fclose(f);
    ml_close(C);

    CHECK(strstr(back, "func.func") != NULL);
    CHECK(strstr(back, "@f") != NULL);
    PASS();
}
TH_REG("mlr", 6, "reprint round trip", mlr06)

static void mlr07(void)
{
    /* An empty file is not a module. Reading it as one would mean every
     * later stage worked on nothing and reported success. */
    ml_ctx_t *C = ml_open(1u << 20);

    CHECK(C != NULL);
    CHNE(ml_parse(C, "", 0), 0);
    CHNE(ml_echo(C, stdout), 0);
    ml_close(C);
    PASS();
}
TH_REG("mlr", 7, "empty input refused", mlr07)

static void mlr08(void)
{
    ml_ctx_t *C = ml_open(1u << 20);

    CHECK(C != NULL);
    CHNE(ml_parse(C, NULL, 0), 0);
    ml_close(C);
    ml_close(NULL);
    PASS();
}
TH_REG("mlr", 8, "null input and null close", mlr08)
