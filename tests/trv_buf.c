/* trv_buf.c -- RV32IM code buffer tests. */

#include "tharns.h"
#include "rv_buf.h"
#include "rv_enc.h"

static rv_buf_t B;

/* ---- init zeros out the buffer ---- */

static void rvb01(void)
{
    memset(&B, 0xAB, sizeof(B));
    rv_buf_init(&B);
    CHEQ(rv_buf_n_words(&B), 0u);
    CHEQ(rv_buf_nbytes(&B), 0u);
    PASS();
}
TH_REG("rvb", 1, "init zero", rvb01);

/* ---- emit appends and returns the slot index ---- */

static void rvb02(void)
{
    rv_buf_init(&B);
    int i0 = rv_buf_emit(&B, rv_addi(RV_X1, RV_X0, 1));
    int i1 = rv_buf_emit(&B, rv_addi(RV_X2, RV_X0, 2));
    int i2 = rv_buf_emit(&B, rv_addi(RV_X3, RV_X0, 3));
    CHEQ(i0, 0);
    CHEQ(i1, 1);
    CHEQ(i2, 2);
    CHEQ(rv_buf_n_words(&B), 3u);
    CHEQ(rv_buf_nbytes(&B), 12u);
    PASS();
}
TH_REG("rvb", 2, "emit seq", rvb02);

/* ---- patch overwrites a previously emitted word ---- */

static void rvb03(void)
{
    rv_buf_init(&B);
    uint32_t orig = rv_addi(RV_X1, RV_X0, 0);
    uint32_t want = rv_addi(RV_X1, RV_X0, 42);
    int idx = rv_buf_emit(&B, orig);
    CHEQ(idx, 0);
    CHEQ(rv_buf_data(&B)[0], orig);
    CHEQ(rv_buf_patch(&B, (uint32_t)idx, want), 0);
    CHEQ(rv_buf_data(&B)[0], want);
    PASS();
}
TH_REG("rvb", 3, "patch basic", rvb03);

/* ---- patch past end is rejected ---- */

static void rvb04(void)
{
    rv_buf_init(&B);
    rv_buf_emit(&B, rv_nop());
    CHEQ(rv_buf_patch(&B, 5u, rv_nop()), -1);
    PASS();
}
TH_REG("rvb", 4, "patch out of bounds", rvb04);

/* ---- offset arithmetic for branches ---- */

static void rvb05(void)
{
    /* Branch at word 0 jumping to word 4 is +16 bytes. */
    rv_buf_init(&B);
    CHEQ(rv_buf_offset(&B, 0u, 4u), 16);
    PASS();
}
TH_REG("rvb", 5, "offset forward", rvb05);

static void rvb06(void)
{
    /* Branch at word 10 jumping back to word 3 is -28 bytes. */
    rv_buf_init(&B);
    CHEQ(rv_buf_offset(&B, 10u, 3u), -28);
    PASS();
}
TH_REG("rvb", 6, "offset backward", rvb06);

/* ---- overflow returns -1 rather than corrupting memory ---- */

static void rvb07(void)
{
    rv_buf_init(&B);
    for (uint32_t i = 0; i < RV_BUF_MAX_WORDS; i++) {
        int idx = rv_buf_emit(&B, rv_nop());
        CHEQ(idx, (int)i);
    }
    CHEQ(rv_buf_emit(&B, rv_nop()), -1);
    CHEQ(rv_buf_n_words(&B), RV_BUF_MAX_WORDS);
    PASS();
}
TH_REG("rvb", 7, "overflow", rvb07);
