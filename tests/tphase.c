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
