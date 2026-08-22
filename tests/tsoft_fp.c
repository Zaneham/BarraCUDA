/* tsoft_fp.c -- Host-compiled tests for the IEEE-754 soft-float
 * runtime. Validates the math against bit-exact expected values
 * for IEEE corner cases (NaN, Inf, signed zero, etc.) and against
 * the host FPU for general arithmetic so any drift from real
 * hardware behaviour shows up immediately.
 *
 * The runtime is FTZ/DAZ by default, so tests that exercise
 * subnormals are guarded with SFP_STRICT_IEEE and currently mark
 * SKIP. When the strict-mode work lands, flipping the flag and
 * removing the SKIPs is the regression-proof migration path. */

#include "tharns.h"
#include "booth/soft_fp.h"
#include "../runtime/device/soft_fp_internal.h"
#include <math.h>
#include <string.h>

/* Bit-pattern helpers so tests can talk in canonical hex. */
static float fbits(uint32_t b)
{
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}
static uint32_t bbits(float f)
{
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

/* ---- Bit-pattern constants ---- */

#define F_POS_ZERO   0x00000000u
#define F_NEG_ZERO   0x80000000u
#define F_POS_ONE    0x3F800000u   /* +1.0     */
#define F_NEG_ONE    0xBF800000u   /* -1.0     */
#define F_POS_TWO    0x40000000u   /* +2.0     */
#define F_NEG_TWO    0xC0000000u   /* -2.0     */
#define F_POS_INF    0x7F800000u
#define F_NEG_INF    0xFF800000u
#define F_QNAN       0x7FC00000u
#define F_SMALLEST_NORMAL  0x00800000u  /* 2^-126   */
#define F_LARGEST_NORMAL   0x7F7FFFFFu  /* close to 2^128 */

/* ---- Negation ---- */

static void sfp01(void)
{
    CHEQ(bbits(__negsf2(fbits(F_POS_ONE))), F_NEG_ONE);
    CHEQ(bbits(__negsf2(fbits(F_NEG_ONE))), F_POS_ONE);
    PASS();
}
TH_REG("sfp", 1, "neg one", sfp01);

static void sfp02(void)
{
    CHEQ(bbits(__negsf2(fbits(F_POS_ZERO))), F_NEG_ZERO);
    CHEQ(bbits(__negsf2(fbits(F_NEG_ZERO))), F_POS_ZERO);
    PASS();
}
TH_REG("sfp", 2, "neg zero", sfp02);

static void sfp03(void)
{
    CHEQ(bbits(__negsf2(fbits(F_POS_INF))), F_NEG_INF);
    CHEQ(bbits(__negsf2(fbits(F_NEG_INF))), F_POS_INF);
    PASS();
}
TH_REG("sfp", 3, "neg inf", sfp03);

/* ---- Addition: canonical values ---- */

static void sfp04(void)
{
    /* 1.0 + 1.0 = 2.0 */
    CHEQ(bbits(__addsf3(fbits(F_POS_ONE), fbits(F_POS_ONE))), F_POS_TWO);
    PASS();
}
TH_REG("sfp", 4, "add one plus one", sfp04);

static void sfp05(void)
{
    /* 1.0 + (-1.0) = +0.0 per round-to-nearest. */
    CHEQ(bbits(__addsf3(fbits(F_POS_ONE), fbits(F_NEG_ONE))), F_POS_ZERO);
    PASS();
}
TH_REG("sfp", 5, "add one minus one", sfp05);

static void sfp06(void)
{
    /* +0 + +0 = +0; -0 + -0 = -0; +0 + -0 = +0. */
    CHEQ(bbits(__addsf3(fbits(F_POS_ZERO), fbits(F_POS_ZERO))), F_POS_ZERO);
    CHEQ(bbits(__addsf3(fbits(F_NEG_ZERO), fbits(F_NEG_ZERO))), F_NEG_ZERO);
    CHEQ(bbits(__addsf3(fbits(F_POS_ZERO), fbits(F_NEG_ZERO))), F_POS_ZERO);
    PASS();
}
TH_REG("sfp", 6, "add zero zero", sfp06);

static void sfp07(void)
{
    /* Inf + Inf = Inf; Inf - Inf = NaN. */
    CHEQ(bbits(__addsf3(fbits(F_POS_INF), fbits(F_POS_INF))), F_POS_INF);
    CHEQ(bbits(__addsf3(fbits(F_POS_INF), fbits(F_NEG_INF))), F_QNAN);
    CHEQ(bbits(__addsf3(fbits(F_POS_INF), fbits(F_POS_ONE))), F_POS_INF);
    PASS();
}
TH_REG("sfp", 7, "add inf", sfp07);

static void sfp08(void)
{
    /* NaN + anything = NaN. */
    CHEQ(bbits(__addsf3(fbits(F_QNAN), fbits(F_POS_ONE))), F_QNAN);
    CHEQ(bbits(__addsf3(fbits(F_POS_ONE), fbits(F_QNAN))), F_QNAN);
    PASS();
}
TH_REG("sfp", 8, "add NaN", sfp08);

/* ---- Addition vs host FPU on random normal values ---- */

static uint32_t prng_state = 1u;
static uint32_t prng_u32(void)
{
    /* xorshift32, plenty good for spreading test inputs around. */
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    return prng_state;
}

/* Generate a random fp32 that is finite and normal (no NaN, Inf,
 * or subnormal). */
static float random_normal_float(void)
{
    uint32_t bits = prng_u32();
    /* Force exponent into normal range [1, 254]. */
    uint32_t exp = 1u + (prng_u32() % 254u);
    bits = (bits & 0x807FFFFFu) | (exp << 23);
    return fbits(bits);
}

static void sfp09(void)
{
    prng_state = 0xC0FFEEu;
    for (int i = 0; i < 1000; i++) {
        float a = random_normal_float();
        float b = random_normal_float();
        float want = a + b;
        float got  = __addsf3(a, b);
        /* If host result is in subnormal range we expect FTZ to
         * differ, so skip those. Same if NaN or Inf appears for
         * some reason in the host result; those cases are covered
         * by the canonical tests above. */
        uint32_t wb = bbits(want);
        uint32_t we = (wb >> 23) & 0xFFu;
        if (we == 0u || we == 0xFFu) continue;
        if (bbits(got) != wb) {
            printf("  ADD mismatch: 0x%08X + 0x%08X => host 0x%08X soft 0x%08X\n",
                   bbits(a), bbits(b), wb, bbits(got));
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("sfp", 9, "add vs host", sfp09);

/* ---- Multiplication ---- */

static void sfp10(void)
{
    CHEQ(bbits(__mulsf3(fbits(F_POS_ONE), fbits(F_POS_TWO))), F_POS_TWO);
    PASS();
}
TH_REG("sfp", 10, "MUL one times two", sfp10);

static void sfp11(void)
{
    CHEQ(bbits(__mulsf3(fbits(F_NEG_ONE), fbits(F_POS_TWO))), F_NEG_TWO);
    CHEQ(bbits(__mulsf3(fbits(F_NEG_ONE), fbits(F_NEG_ONE))), F_POS_ONE);
    PASS();
}
TH_REG("sfp", 11, "MUL sign", sfp11);

static void sfp12(void)
{
    /* Inf * 0 is NaN per IEEE. */
    CHEQ(bbits(__mulsf3(fbits(F_POS_INF), fbits(F_POS_ZERO))), F_QNAN);
    CHEQ(bbits(__mulsf3(fbits(F_POS_ZERO), fbits(F_POS_INF))), F_QNAN);
    PASS();
}
TH_REG("sfp", 12, "MUL inf zero", sfp12);

static void sfp13(void)
{
    /* Inf * finite_positive = Inf; Inf * finite_negative = -Inf. */
    CHEQ(bbits(__mulsf3(fbits(F_POS_INF), fbits(F_POS_ONE))), F_POS_INF);
    CHEQ(bbits(__mulsf3(fbits(F_POS_INF), fbits(F_NEG_ONE))), F_NEG_INF);
    PASS();
}
TH_REG("sfp", 13, "MUL inf finite", sfp13);

static void sfp14(void)
{
    /* Sign-of-zero rule: +0 * -1 = -0. */
    CHEQ(bbits(__mulsf3(fbits(F_POS_ZERO), fbits(F_NEG_ONE))), F_NEG_ZERO);
    CHEQ(bbits(__mulsf3(fbits(F_NEG_ZERO), fbits(F_NEG_ONE))), F_POS_ZERO);
    PASS();
}
TH_REG("sfp", 14, "MUL zero sign", sfp14);

static void sfp15(void)
{
    prng_state = 0xBADBEEFu;
    for (int i = 0; i < 1000; i++) {
        float a = random_normal_float();
        float b = random_normal_float();
        float want = a * b;
        float got  = __mulsf3(a, b);
        uint32_t wb = bbits(want);
        uint32_t we = (wb >> 23) & 0xFFu;
        if (we == 0u || we == 0xFFu) continue;
        if (bbits(got) != wb) {
            printf("  MUL mismatch: 0x%08X * 0x%08X => host 0x%08X soft 0x%08X\n",
                   bbits(a), bbits(b), wb, bbits(got));
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("sfp", 15, "MUL vs host", sfp15);

/* ---- Division ---- */

static void sfp16(void)
{
    /* 1.0 / 2.0 = 0.5 */
    CHEQ(bbits(__divsf3(fbits(F_POS_ONE), fbits(F_POS_TWO))), 0x3F000000u);
    PASS();
}
TH_REG("sfp", 16, "DIV one by two", sfp16);

static void sfp17(void)
{
    /* x / 0 = signed Inf for finite x; 0/0 = NaN. */
    CHEQ(bbits(__divsf3(fbits(F_POS_ONE), fbits(F_POS_ZERO))), F_POS_INF);
    CHEQ(bbits(__divsf3(fbits(F_NEG_ONE), fbits(F_POS_ZERO))), F_NEG_INF);
    CHEQ(bbits(__divsf3(fbits(F_POS_ZERO), fbits(F_POS_ZERO))), F_QNAN);
    PASS();
}
TH_REG("sfp", 17, "DIV by zero", sfp17);

static void sfp18(void)
{
    /* Inf / Inf = NaN. */
    CHEQ(bbits(__divsf3(fbits(F_POS_INF), fbits(F_POS_INF))), F_QNAN);
    PASS();
}
TH_REG("sfp", 18, "DIV inf inf", sfp18);

static void sfp19(void)
{
    prng_state = 0xDEADBABEu;
    for (int i = 0; i < 1000; i++) {
        float a = random_normal_float();
        float b = random_normal_float();
        float want = a / b;
        float got  = __divsf3(a, b);
        uint32_t wb = bbits(want);
        uint32_t we = (wb >> 23) & 0xFFu;
        if (we == 0u || we == 0xFFu) continue;
        if (bbits(got) != wb) {
            printf("  DIV mismatch: 0x%08X / 0x%08X => host 0x%08X soft 0x%08X\n",
                   bbits(a), bbits(b), wb, bbits(got));
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("sfp", 19, "DIV vs host", sfp19);

/* ---- Comparisons ---- */

static void sfp20(void)
{
    CHEQ(__eqsf2(fbits(F_POS_ONE), fbits(F_POS_ONE)), 0);
    CHEQ(__eqsf2(fbits(F_POS_ONE), fbits(F_POS_TWO)), 1);
    /* +0 == -0 by IEEE rule. */
    CHEQ(__eqsf2(fbits(F_POS_ZERO), fbits(F_NEG_ZERO)), 0);
    PASS();
}
TH_REG("sfp", 20, "compare equal", sfp20);

static void sfp21(void)
{
    CHECK(__ltsf2(fbits(F_NEG_ONE), fbits(F_POS_ONE)) < 0);
    CHECK(__ltsf2(fbits(F_POS_ONE), fbits(F_NEG_ONE)) > 0);
    CHEQ(__ltsf2(fbits(F_POS_ONE), fbits(F_POS_ONE)), 0);
    PASS();
}
TH_REG("sfp", 21, "compare less", sfp21);

static void sfp22(void)
{
    /* NaN comparisons are unordered; the magnitude comparisons
     * return a positive value per libgcc convention. */
    CHECK(__unordsf2(fbits(F_QNAN), fbits(F_POS_ONE)));
    CHECK(__ltsf2  (fbits(F_QNAN), fbits(F_POS_ONE)) > 0);
    CHECK(__lesf2  (fbits(F_QNAN), fbits(F_POS_ONE)) > 0);
    PASS();
}
TH_REG("sfp", 22, "compare NaN", sfp22);

/* ---- Integer to float ---- */

static void sfp23(void)
{
    CHEQ(bbits(__floatsisf(0)),  F_POS_ZERO);
    CHEQ(bbits(__floatsisf(1)),  F_POS_ONE);
    CHEQ(bbits(__floatsisf(-1)), F_NEG_ONE);
    CHEQ(bbits(__floatsisf(2)),  F_POS_TWO);
    PASS();
}
TH_REG("sfp", 23, "int to float", sfp23);

static void sfp24(void)
{
    /* 16777216 = 2^24 is exactly representable. */
    CHEQ(bbits(__floatsisf(16777216)),  0x4B800000u);
    CHEQ(bbits(__floatsisf(-16777216)), 0xCB800000u);
    PASS();
}
TH_REG("sfp", 24, "int to float large", sfp24);

/* ---- Float to integer ---- */

static void sfp25(void)
{
    CHEQ(__fixsfsi(fbits(F_POS_ZERO)),  0);
    CHEQ(__fixsfsi(fbits(F_POS_ONE)),   1);
    CHEQ(__fixsfsi(fbits(F_NEG_ONE)),  -1);
    CHEQ(__fixsfsi(fbits(F_POS_TWO)),   2);
    PASS();
}
TH_REG("sfp", 25, "float to int", sfp25);

static void sfp26(void)
{
    /* 1.5 = 0x3FC00000 should truncate to 1. */
    CHEQ(__fixsfsi(fbits(0x3FC00000u)),  1);
    /* -1.5 truncates to -1. */
    CHEQ(__fixsfsi(fbits(0xBFC00000u)), -1);
    PASS();
}
TH_REG("sfp", 26, "float to int truncates", sfp26);

static void sfp27(void)
{
    /* Inf clamps to INT_MAX / INT_MIN. */
    CHEQ(__fixsfsi(fbits(F_POS_INF)), (int32_t)0x7FFFFFFF);
    CHEQ(__fixsfsi(fbits(F_NEG_INF)), (int32_t)0x80000000);
    /* NaN returns 0 per convention. */
    CHEQ(__fixsfsi(fbits(F_QNAN)), 0);
    PASS();
}
TH_REG("sfp", 27, "float to int overflow", sfp27);

/* ---- Subnormal handling (strict-only; SKIP by default) ----
 *
 * In FTZ/DAZ mode (the default for v0.5) subnormal inputs flush
 * to zero on the way in and subnormal outputs flush on the way
 * out. These tests would only pass when SFP_STRICT_IEEE is on;
 * we keep them visible so the strict-mode work has a known test
 * surface to revive. */

static void sfp28(void)
{
#if SFP_STRICT_IEEE
    SKIP("strict-mode subnormal handling not yet implemented");
#else
    /* Smallest positive subnormal is 0x00000001. Under FTZ, any
     * arithmetic involving it should treat it as +0. */
    float sub  = fbits(0x00000001u);
    float zero = fbits(F_POS_ZERO);
    CHEQ(bbits(__addsf3(sub, zero)), F_POS_ZERO);
    PASS();
#endif
}
TH_REG("sfp", 28, "subnormal unpacks to zero", sfp28);
