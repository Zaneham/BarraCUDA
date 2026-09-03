/* tphase.c -- phase output tests
 * Does each pipeline stage produce something resembling output? */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- phase: preprocessor ---- */

static void pha01(void)
{
    int rc = th_run(BC_BIN " --pp tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strlen(obuf) > 0);
    PASS();
}
TH_REG("pha", 1, "preprocessor runs", pha01)

static void pha02(void)
{
    int rc = th_run(BC_BIN " --pp tests/comment_macro_skip.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "COMMENT_MACRO should remain literal") != NULL);
    CHECK(strstr(obuf, "23 should remain literal") == NULL);
    PASS();
}
TH_REG("pha", 2, "preprocessor comment opaque", pha02)

static void pha03(void)
{
    int rc = th_run(BC_BIN " --lex tests/comment_quotes.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 error(s)") != NULL);
    PASS();
}
TH_REG("pha", 3, "lexer comment quotes", pha03)

/* ---- phase: parser (AST dump) ---- */

static void pha04(void)
{
    int rc = th_run(BC_BIN " --parse tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    PASS();
}
TH_REG("pha", 4, "parser dumps an AST", pha04)

/* A function pointer declarator used to parse as a call expression, which
   cost nothing at parse time and everything later. */
static void pha05(void)
{
    int rc = th_run(BC_BIN " --parse tests/fnptr.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 parse error(s)") != NULL);
    CHECK(strstr(obuf, "(ident cb_t)") != NULL);
    CHECK(strstr(obuf, "(ident hook)") != NULL);
    CHECK(strstr(obuf, "call") == NULL);
    PASS();
}
TH_REG("pha", 5, "function pointers survive the parser", pha05)

/* Nested structs matter here: the enclosing name is saved and restored,
   so inner's constructor must not be mistaken for outer's. */
static void pha06(void)
{
    int rc = th_run(BC_BIN " --parse tests/ctor.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 parse error(s)") != NULL);
    CHECK(strstr(obuf, "(ident vec3)") != NULL);
    CHECK(strstr(obuf, "(ident inner)") != NULL);
    CHECK(strstr(obuf, "(ident sum)") != NULL);
    PASS();
}
TH_REG("pha", 6, "constructors survive the parser", pha06)

/* ---- phase: IR ---- */

static void pha07(void)
{
    int rc = th_run(BC_BIN " --ir tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func") != NULL);
    PASS();
}
TH_REG("pha", 7, "IR comes out", pha07)

/* ---- phase: semantic analysis ---- */

static void pha08(void)
{
    int rc = th_run(BC_BIN " --sema tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    PASS();
}
TH_REG("pha", 8, "semantic analysis runs", pha08)

/* ---- phase: declaration specifiers Booth does not model ---- */

/* _Noreturn parsed as a type name, so the void after it was a syntax error,
   and 61 of llama.cpp's 67 CUDA files start with one via GGML_NORETURN. */
static void pha09(void)
{
    int rc = th_run(BC_BIN " --parse tests/noretn.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 parse error(s)") != NULL);
    CHECK(strstr(obuf, "(ident nrt_kern)") != NULL);
    CHECK(strstr(obuf, "(ident nrt_bump)") != NULL);
    PASS();
}
TH_REG("pha", 9, "_Noreturn and [[attributes]] are accepted", pha09)

/* ---- phase: preprocessor, again ---- */

/* ppcyc_a and ppcyc_b include each other. Without #pragma once the depth
   guard is all that stops them, and it stopped ggml-cuda too. */
static void pha10(void)
{
    int rc = th_run(BC_BIN " --pp tests/ppcycle.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "error") == NULL);
    CHECK(strstr(obuf, "3 + 4") != NULL);
    PASS();
}
TH_REG("pha", 10, "#pragma once breaks an include cycle", pha10)

/* Three that travel together: a body kept its trailing // comment, an
   argument list that closed on a later line was never joined, and ... in a
   parameter list ended up in the body rather than naming __VA_ARGS__. */
static void pha11(void)
{
    int rc = th_run(BC_BIN " --pp tests/ppvarg.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "...") == NULL);
    CHECK(strstr(obuf, "decl") == NULL);
    CHECK(strstr(obuf, "the hint is dropped") == NULL);
    CHECK(strstr(obuf, "int ppv_add(int a, int b);") != NULL);
    CHECK(strstr(obuf, ">> 2") != NULL);
    CHECK(strstr(obuf, "\"a, b\"") != NULL);
    PASS();
}
TH_REG("pha", 11, "variadic macros and multi-line calls", pha11)

/* The output buffer used to fill and keep going, unterminated, so the lexer
   read on into the source buffer that follows it and reported whatever it
   found there. The symptom was an unterminated block comment. */
static void pha12(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx tests/ppovfl.cu -o ppovfl_test.ptx",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "E053") != NULL);
    CHECK(strstr(obuf, "E002") == NULL);
    CHECK(strstr(obuf, "E020") == NULL);
    remove("ppovfl_test.ptx");
    PASS();
}
TH_REG("pha", 12, "a truncated expansion is diagnosed", pha12)
