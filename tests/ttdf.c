/* ttdf.c -- Tile DataFlow IR shape tests.
 * Build modules by hand, dump them, check the bookkeeping holds. */

#include "tharns.h"
#include "tdf.h"
#include "noc.h"
#include "rv_enc.h"

static char obuf[TH_BUFSZ];

/* CB-arc emit comparison buffers. Each rv_buf_t is 16 KiB, so keep them in
 * BSS rather than on MinGW's lazily-committed stack. */
static rv_buf_t EA, EB;

static int bufs_eq(const rv_buf_t *a, const rv_buf_t *b)
{
    uint32_t n = rv_buf_n_words(a);
    if (n != rv_buf_n_words(b)) return 0;
    return memcmp(rv_buf_data(a), rv_buf_data(b), n * 4u) == 0;
}

static int has_word(const rv_buf_t *b, uint32_t w)
{
    uint32_t n = rv_buf_n_words(b);
    const uint32_t *d = rv_buf_data(b);
    for (uint32_t i = 0; i < n; i++) if (d[i] == w) return 1;
    return 0;
}

/* bir_module_t is a multi-megabyte arena and absolutely will not fit
 * on the stack. We never dereference this in the lowering, the
 * lowering just compares pointers, so one BSS sentinel is enough
 * to stand in for "some real BIR module" across the tests. */
static bir_module_t fake_mod_a;

/* td_mod_t is 24 KiB. On MinGW that pushes past the lazy stack-guard
 * commit boundary and faults inside memset before td_init can return.
 * Every test that wants a module reaches for this shared BSS slot
 * and resets it by calling td_init, exactly like Metalium reuses an
 * ECB across transactions. */
static td_mod_t M;

/* ---- Helpers ---- */

static td_tag_t tile_f32(uint16_t r, uint16_t c, uint8_t layout)
{
    td_tag_t t;
    t.rows   = r;
    t.cols   = c;
    t.dtype  = BIR_TYPE_FLOAT;
    t.layout = layout;
    t._pad   = 0;
    return t;
}

/* MinGW has no fmemopen, so we go through tmpfile() which is C89.
 * Slower than dirt but the dump tests do not need to be fast. */
static int dump_to_buf(const td_mod_t *M, char *buf, int bufsz)
{
    FILE *fp = tmpfile();
    if (!fp) return -1;
    td_dump(M, fp);
    rewind(fp);
    size_t n = fread(buf, 1, (size_t)(bufsz - 1), fp);
    buf[n] = '\0';
    fclose(fp);
    return 0;
}

/* ---- init: zero state, target set ---- */

static void tdf01(void)
{
    /* M is the shared static above */
    /* dirty the struct first so we can tell init really wiped it */
    memset(&M, 0xAB, sizeof(M));
    td_init(&M, TD_TGT_AMD);
    CHEQ(M.nrgn, 0);
    CHEQ(M.ncha, 0);
    CHEQ(M.narc, 0);
    CHEQ(M.target, TD_TGT_AMD);
    PASS();
}
TH_REG("tdf", 1, "init zero", tdf01);

/* ---- solo region: the AMD/NVIDIA degenerate shape ---- */

static void tdf02(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_NVIDIA);
    uint16_t r = td_mkrgn(&M, TD_RG_SOLO);
    CHEQ(r, 0);
    CHEQ(M.nrgn, 1);
    CHEQ(M.rgns[0].id, 0);
    CHEQ(M.rgns[0].role, TD_RG_SOLO);
    CHEQ(M.ncha, 0);
    CHEQ(M.narc, 0);
    PASS();
}
TH_REG("tdf", 2, "solo region", tdf02);

/* ---- three-region fission: reader/compute/writer ---- */

static void tdf03(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);

    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    uint16_t wrt = td_mkrgn(&M, TD_RG_WRT);
    CHEQ(rdr, 0);
    CHEQ(cmp, 1);
    CHEQ(wrt, 2);
    CHEQ(M.nrgn, 3);
    CHEQ(M.rgns[0].role, TD_RG_RDR);
    CHEQ(M.rgns[1].role, TD_RG_CMP);
    CHEQ(M.rgns[2].role, TD_RG_WRT);
    PASS();
}
TH_REG("tdf", 3, "three regions", tdf03);

/* ---- channels link two regions with tile shape and depth ---- */

static void tdf04(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);

    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t ch  = td_link(&M, rdr, cmp, tag, /*depth=*/4);
    CHEQ(ch, 0);
    CHEQ(M.ncha, 1);
    CHEQ(M.chans[0].prod, rdr);
    CHEQ(M.chans[0].cons, cmp);
    CHEQ(M.chans[0].depth, 4);
    CHEQ(M.chans[0].tag.rows, 32);
    CHEQ(M.chans[0].tag.cols, 32);
    CHEQ(M.chans[0].tag.dtype, BIR_TYPE_FLOAT);
    CHEQ(M.chans[0].tag.layout, TD_LAY_INTRL);
    PASS();
}
TH_REG("tdf", 4, "channel link", tdf04);

/* ---- link with bogus producer rejected ---- */

static void tdf05(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    td_mkrgn(&M, TD_RG_CMP);
    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t ch  = td_link(&M, 0, /*nonsense*/ 99, tag, 4);
    CHEQ(ch, TD_BAD_ID);
    CHEQ(M.ncha, 0);
    PASS();
}
TH_REG("tdf", 5, "link bad region", tdf05);

/* ---- arcs: full CB pipeline (push, wait, pop) plus a NoC read ---- */

static void tdf06(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t ch  = td_link(&M, rdr, cmp, tag, 4);

    uint16_t a0 = td_mkarc(&M, rdr, TD_AR_RD,   TD_BAD_ID, 1, /*anchor=*/10);
    uint16_t a1 = td_mkarc(&M, rdr, TD_AR_PUSH, ch,        1, 11);
    uint16_t a2 = td_mkarc(&M, cmp, TD_AR_WAIT, ch,        1, 20);
    uint16_t a3 = td_mkarc(&M, cmp, TD_AR_POP,  ch,        1, 25);

    CHEQ(a0, 0);
    CHEQ(a1, 1);
    CHEQ(a2, 2);
    CHEQ(a3, 3);
    CHEQ(M.narc, 4);

    CHEQ(M.arcs[0].kind, TD_AR_RD);
    CHEQ(M.arcs[0].chan, TD_BAD_ID);
    CHEQ(M.arcs[1].kind, TD_AR_PUSH);
    CHEQ(M.arcs[1].chan, ch);
    CHEQ(M.arcs[2].kind, TD_AR_WAIT);
    CHEQ(M.arcs[3].kind, TD_AR_POP);
    PASS();
}
TH_REG("tdf", 6, "arcs pipeline", tdf06);

/* ---- CB arc with unknown channel rejected ---- */

static void tdf07(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    td_mkrgn(&M, TD_RG_CMP);
    uint16_t a = td_mkarc(&M, 0, TD_AR_PUSH, /*nonsense*/ 7, 1, 0);
    CHEQ(a, TD_BAD_ID);
    CHEQ(M.narc, 0);
    PASS();
}
TH_REG("tdf", 7, "arc bad channel", tdf07);

/* ---- lookups return NULL for out-of-range ids ---- */

static void tdf08(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_AMD);
    td_mkrgn(&M, TD_RG_SOLO);
    CHECK(td_rgn(&M, 0) != NULL);
    CHECK(td_rgn(&M, 1) == NULL);
    CHECK(td_chan(&M, 0) == NULL);
    CHECK(td_arc(&M, 0) == NULL);
    PASS();
}
TH_REG("tdf", 8, "lookup out of range", tdf08);

/* ---- dump emits something readable for both shapes ---- */

static void tdf09(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_AMD);
    td_mkrgn(&M, TD_RG_SOLO);

    memset(obuf, 0, sizeof(obuf));
    CHEQ(dump_to_buf(&M, obuf, sizeof(obuf)), 0);
    CHECK(strstr(obuf, "target=AMD") != NULL);
    CHECK(strstr(obuf, "regions:  1") != NULL);
    CHECK(strstr(obuf, "region 0: SOLO") != NULL);
    PASS();
}
TH_REG("tdf", 9, "dump solo", tdf09);

static void tdf10(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t ch  = td_link(&M, rdr, cmp, tag, 4);
    td_mkarc(&M, rdr, TD_AR_PUSH, ch, 1, 0);
    td_mkarc(&M, cmp, TD_AR_WAIT, ch, 1, 0);

    memset(obuf, 0, sizeof(obuf));
    CHEQ(dump_to_buf(&M, obuf, sizeof(obuf)), 0);
    CHECK(strstr(obuf, "target=TENSIX") != NULL);
    CHECK(strstr(obuf, "region 0: RDR") != NULL);
    CHECK(strstr(obuf, "region 1: CMP") != NULL);
    CHECK(strstr(obuf, "chan 0: rgn0 -> rgn1") != NULL);
    CHECK(strstr(obuf, "32x32 fINTRL") != NULL);
    CHECK(strstr(obuf, "rgn0 PUSH") != NULL);
    CHECK(strstr(obuf, "rgn1 WAIT") != NULL);
    PASS();
}
TH_REG("tdf", 10, "dump pipeline", tdf10);

/* ---- lower: AMD/NVIDIA solo passthrough ---- */

static void tdf11(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_AMD);
    uint16_t r = td_mkrgn(&M, TD_RG_SOLO);
    /* Body is just a sentinel pointer for the passthrough test;
     * the lowering does not dereference it. */
    M.rgns[r].body = &fake_mod_a;

    td_lout_t out;
    CHEQ(td_lower(&M, &out), BC_OK);
    CHEQ(out.nmods, 1);
    CHECK(out.mods[0] == &fake_mod_a);
    CHEQ(out.owns[0], 0);
    PASS();
}
TH_REG("tdf", 11, "lower solo AMD", tdf11);

static void tdf12(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_NVIDIA);
    uint16_t r = td_mkrgn(&M, TD_RG_SOLO);
    M.rgns[r].body = &fake_mod_a;

    td_lout_t out;
    CHEQ(td_lower(&M, &out), BC_OK);
    CHEQ(out.nmods, 1);
    CHECK(out.mods[0] == &fake_mod_a);
    PASS();
}
TH_REG("tdf", 12, "lower solo NVIDIA", tdf12);

/* ---- lower: SOLO without a body is rejected ---- */

static void tdf13(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_AMD);
    td_mkrgn(&M, TD_RG_SOLO);
    td_lout_t out;
    CHEQ(td_lower(&M, &out), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 13, "lower solo no body", tdf13);

/* ---- lower: AMD with the wrong region role is rejected ---- */

static void tdf14(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_AMD);
    uint16_t r = td_mkrgn(&M, TD_RG_CMP);
    M.rgns[r].body = &fake_mod_a;
    td_lout_t out;
    CHEQ(td_lower(&M, &out), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 14, "lower wrong role", tdf14);

/* ---- lower: AMD with channels present is rejected ---- */

static void tdf15(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_AMD);
    uint16_t r = td_mkrgn(&M, TD_RG_SOLO);
    uint16_t r2 = td_mkrgn(&M, TD_RG_SOLO);   /* second region for the link */
    M.rgns[r].body = &fake_mod_a;
    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    td_link(&M, r, r2, tag, 4);

    td_lout_t out;
    /* lowering should refuse because nrgn != 1 (and there is a channel) */
    CHEQ(td_lower(&M, &out), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 15, "lower solo with channel", tdf15);

/* ---- lower: Tensix passes SOLO through for now (stub) ---- */

static void tdf16(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    uint16_t r = td_mkrgn(&M, TD_RG_SOLO);
    M.rgns[r].body = &fake_mod_a;
    td_lout_t out;
    CHEQ(td_lower(&M, &out), BC_OK);
    CHEQ(out.nmods, 1);
    CHECK(out.mods[0] == &fake_mod_a);
    PASS();
}
TH_REG("tdf", 16, "lower tensix solo", tdf16);

/* ---- lower: Tensix fission not yet implemented, must say so ---- */

static void tdf17(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    td_mkrgn(&M, TD_RG_RDR);
    td_mkrgn(&M, TD_RG_CMP);
    td_mkrgn(&M, TD_RG_WRT);
    td_lout_t out;
    /* Not implemented yet; lowering must refuse rather than scribble. */
    CHEQ(td_lower(&M, &out), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 17, "lower tensix fission stub", tdf17);

/* ---- build_solo_from_bir: happy path ---- */

static void tdf18(void)
{
    /* M is the shared static above */
    CHEQ(td_build_solo_from_bir(&M, TD_TGT_AMD, &fake_mod_a), BC_OK);
    CHEQ(M.target, TD_TGT_AMD);
    CHEQ(M.nrgn, 1);
    CHEQ(M.rgns[0].role, TD_RG_SOLO);
    CHECK(M.rgns[0].body == &fake_mod_a);
    PASS();
}
TH_REG("tdf", 18, "build solo ok", tdf18);

/* ---- build_solo_from_bir: NULL body refused ---- */

static void tdf19(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_AMD);
    CHEQ(td_build_solo_from_bir(&M, TD_TGT_AMD, NULL), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 19, "build solo null", tdf19);

/* ---- build_solo + lower round-trips back to the same BIR pointer ---- */

static void tdf20(void)
{
    /* M is the shared static above */
    CHEQ(td_build_solo_from_bir(&M, TD_TGT_NVIDIA, &fake_mod_a), BC_OK);
    td_lout_t out;
    CHEQ(td_lower(&M, &out), BC_OK);
    CHEQ(out.nmods, 1);
    CHECK(out.mods[0] == &fake_mod_a);
    PASS();
}
TH_REG("tdf", 20, "build then lower", tdf20);

/* ---- td_tile_bytes returns the right size per dtype ---- */
/* Wormhole tiles are 32x32, so 4096 bytes at fp32 and 2048 at fp16/bf16.
 * Those numbers feed NoC alignment, CB depth and runtime args, so a
 * regression here ripples everywhere. */

static void tdf21(void)
{
    td_tag_t t = tile_f32(32, 32, TD_LAY_INTRL);
    CHEQ(td_tile_bytes(t), 32u * 32u * 4u);
    PASS();
}
TH_REG("tdf", 21, "tile bytes fp32", tdf21);

static void tdf22(void)
{
    td_tag_t t;
    t.rows = 32; t.cols = 32;
    t.dtype = BIR_TYPE_BFLOAT; t.layout = TD_LAY_INTRL; t._pad = 0;
    CHEQ(td_tile_bytes(t), 32u * 32u * 2u);
    PASS();
}
TH_REG("tdf", 22, "tile bytes bf16", tdf22);

/* ---- placement assigns increasing offsets to every channel ---- */
/*
 * Two-channel module, both fp32 32x32 depth 2. First channel sits
 * at TD_L1_CB_BASE, second sits one (tile-data + FIFO) block above
 * it, rounded to 16-byte alignment. */

static void tdf23(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    uint16_t r0 = td_mkrgn(&M, TD_RG_RDR);
    uint16_t r1 = td_mkrgn(&M, TD_RG_CMP);
    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t c0 = td_link(&M, r0, r1, tag, 2);
    uint16_t c1 = td_link(&M, r0, r1, tag, 2);

    CHEQ(td_place_l1(&M), BC_OK);

    CHEQ(M.chans[c0].l1_off, TD_L1_CB_BASE);
    /* one tile-data block (4096 * 2 = 8192) + FIFO (8) rounded
     * up to 16 = 8208 = 0x2010 between channels */
    CHEQ(M.chans[c1].l1_off, TD_L1_CB_BASE + 0x2010u);
    PASS();
}
TH_REG("tdf", 23, "place two channels", tdf23);

/* ---- placement refuses when the budget is exceeded ---- */
/*
 * Construct a channel whose data buffer alone is larger than the
 * entire CB region. Placement should refuse cleanly rather than
 * silently wrap or scribble outside L1. */

static void tdf24(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    uint16_t r0 = td_mkrgn(&M, TD_RG_RDR);
    uint16_t r1 = td_mkrgn(&M, TD_RG_CMP);
    /* 1024x1024 fp32 tile = 4 MiB, far beyond a single L1 */
    td_tag_t tag;
    tag.rows = 1024; tag.cols = 1024;
    tag.dtype = BIR_TYPE_FLOAT; tag.layout = TD_LAY_INTRL; tag._pad = 0;
    td_link(&M, r0, r1, tag, 2);
    CHEQ(td_place_l1(&M), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 24, "place budget", tdf24);

/* ---- NoC address encoder: spec bit layout ---- */
/*
 * MemoryMap.md unicast format: local[35:0] | x[41:36] | y[47:42].
 * Pick values that exercise each field's high bit so a bad shift
 * shows up immediately. */

static void tdf25(void)
{
    /* (x=1, y=2, local=0x1000) packs to 0x1000 | (1<<36) | (2<<42),
     * which is 0x0000081000001000. */
    CHEQ(td_noc_addr(1, 2, 0x1000ull), 0x0000081000001000ull);
    PASS();
}
TH_REG("tdf", 25, "NoC addr basic", tdf25);

static void tdf26(void)
{
    /* X and Y are 6-bit fields. Anything above bit 5 must be
     * masked off rather than spilling into the reserved zone. */
    CHEQ(td_noc_addr(0xFF, 0xFF, 0ull),
         (0x3Full << 36) | (0x3Full << 42));
    PASS();
}
TH_REG("tdf", 26, "NoC addr coord masking", tdf26);

static void tdf27(void)
{
    /* Local address is 36 bits, anything above must be masked off
     * since the high 28 bits of the 64-bit word are reserved. */
    CHEQ(td_noc_addr(0, 0, 0xFFFFFFFFFFFFFFFFull),
         TD_NOC_LOCAL_MASK);
    PASS();
}
TH_REG("tdf", 27, "NoC addr local 36bits", tdf27);

/* ---- orchestrator: RD goes on NoC 0, WR on NoC 1 ---- */
/*
 * Build a three-region pipeline by hand with one read arc and one
 * write arc, run the orchestrator, check the noc_id and length
 * fields come out per the spec. */

static void tdf28(void)
{
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    uint16_t wrt = td_mkrgn(&M, TD_RG_WRT);

    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t c_in  = td_link(&M, rdr, cmp, tag, 2);
    uint16_t c_out = td_link(&M, cmp, wrt, tag, 2);

    uint16_t rd = td_mkarc(&M, rdr, TD_AR_RD, c_in,  1, 0);
    uint16_t wr = td_mkarc(&M, wrt, TD_AR_WR, c_out, 1, 0);

    CHEQ(td_noc_orchestrate(&M), BC_OK);
    CHEQ(M.arcs[rd].noc_id, TD_NOC0);
    CHEQ(M.arcs[rd].length, 32u * 32u * 4u);
    CHEQ(M.arcs[wr].noc_id, TD_NOC1);
    CHEQ(M.arcs[wr].length, 32u * 32u * 4u);
    PASS();
}
TH_REG("tdf", 28, "NoC orchestrate dirs", tdf28);

/* ---- orchestrator: refuses transfers above 8 KiB ---- */
/*
 * 64x64 fp32 = 16 KiB which is double the per-request ceiling.
 * The orchestrator must refuse rather than truncate; the splitting
 * pass that handles oversize transfers is not implemented yet. */

static void tdf29(void)
{
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    td_tag_t big;
    big.rows = 64; big.cols = 64;
    big.dtype = BIR_TYPE_FLOAT; big.layout = TD_LAY_INTRL; big._pad = 0;
    uint16_t ch = td_link(&M, rdr, cmp, big, 1);
    td_mkarc(&M, rdr, TD_AR_RD, ch, 1, 0);
    CHEQ(td_noc_orchestrate(&M), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 29, "NoC orchestrate oversize", tdf29);

/* ---- orchestrator leaves CB arcs untouched ---- */
/*
 * PUSH/WAIT/RSV/POP do not cross the NoC; their noc_id and length
 * fields should be left at zero, not silently mutated. */

static void tdf30(void)
{
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    td_tag_t tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t ch = td_link(&M, rdr, cmp, tag, 2);
    uint16_t push = td_mkarc(&M, rdr, TD_AR_PUSH, ch, 1, 0);
    uint16_t wait = td_mkarc(&M, cmp, TD_AR_WAIT, ch, 1, 0);

    CHEQ(td_noc_orchestrate(&M), BC_OK);
    CHEQ(M.arcs[push].noc_id, 0);
    CHEQ(M.arcs[push].length, 0u);
    CHEQ(M.arcs[wait].noc_id, 0);
    CHEQ(M.arcs[wait].length, 0u);
    PASS();
}
TH_REG("tdf", 30, "NoC skips circular buffer arcs", tdf30);

/* ---- integration: fission on test_dce.cu (one stored ptr) ---- */
/*
 * dce_chain stores to its first pointer param and never loads from
 * one, so fission should produce a single CMP -> WRT channel and
 * leave the RDR side empty. The compile is wedged through the
 * AMDGPU backend purely so the C99 pipeline runs all the way to
 * BIR; --tdf-fission then bails before any AMDGPU work happens. */

static void tdf31(void)
{
    int rc = th_run(BC_BIN " --tdf-fission --amdgpu-bin -o "
                    "/tmp/_tdf_fission_test.hsaco tests/test_dce.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "target=TENSIX") != NULL);
    CHECK(strstr(obuf, "regions:  3") != NULL);
    CHECK(strstr(obuf, "channels: 1") != NULL);
    CHECK(strstr(obuf, "region 0: RDR") != NULL);
    CHECK(strstr(obuf, "region 1: CMP") != NULL);
    CHECK(strstr(obuf, "region 2: WRT") != NULL);
    /* The single channel runs CMP -> WRT, not RDR -> anything. */
    CHECK(strstr(obuf, "rgn1 -> rgn2") != NULL);
    CHECK(strstr(obuf, "rgn0 -> rgn1") == NULL);
    /* Placement should have moved the channel off the default
     * zero offset onto the CB region floor. */
    /* CB region now starts at 0x8100, after the 256-byte runtime
     * arg slab inserted between code and CBs in rt_args.h. */
    CHECK(strstr(obuf, "l1=0x8100") != NULL);
    CHECK(strstr(obuf, "l1=0x0\n") == NULL);
    /* One output channel means exactly one WR NoC arc, no RD. */
    CHECK(strstr(obuf, "WR chan") != NULL);
    CHECK(strstr(obuf, "RD chan") == NULL);
    PASS();
}
TH_REG("tdf", 31, "fission store only", tdf31);

/* ---- integration: fission on matmul (1 store + 2 loads) ---- */
/*
 * canonical.cu's first kernel is matmul_general(C, A, B, M, N, K).
 * C is written, A and B are read, M/N/K are scalars and ignored.
 * The fissioned graph should therefore have one CMP -> WRT channel
 * (for C) plus two RDR -> CMP channels (for A and B), six arcs
 * total. */

static void tdf32(void)
{
    int rc = th_run(BC_BIN " --tdf-fission --amdgpu-bin -o "
                    "/tmp/_tdf_fission_matmul.hsaco tests/canonical.cu",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "target=TENSIX") != NULL);
    CHECK(strstr(obuf, "regions:  3")  != NULL);
    CHECK(strstr(obuf, "channels: 3")  != NULL);
    /* matmul has two read params (A, B) and one write (C). Per channel
     * the fissioned graph emits: producer PUSH + consumer WAIT + a
     * single NoC arc (RD for inputs, WR for outputs). Three channels
     * therefore produce nine arcs. */
    CHECK(strstr(obuf, "arcs:     9")  != NULL);
    /* At least one CMP -> WRT and at least one RDR -> CMP. */
    CHECK(strstr(obuf, "rgn1 -> rgn2") != NULL);
    CHECK(strstr(obuf, "rgn0 -> rgn1") != NULL);
    /* Reads use NoC 0, writes use NoC 1 per the orchestrator's
     * direction-of-flow heuristic from RoutingPaths.md. */
    CHECK(strstr(obuf, "RD chan") != NULL);
    CHECK(strstr(obuf, "WR chan") != NULL);
    CHECK(strstr(obuf, "noc0 len=4096") != NULL);
    CHECK(strstr(obuf, "noc1 len=4096") != NULL);
    PASS();
}
TH_REG("tdf", 32, "fission matmul shape", tdf32);

/* ---- region overflow returns TD_BAD_ID after the table fills ---- */

static void tdf33(void)
{
    /* M is the shared static above */
    td_init(&M, TD_TGT_TENSIX);
    for (int i = 0; i < TD_MAX_RGNS; i++) {
        uint16_t r = td_mkrgn(&M, TD_RG_CMP);
        CHNE(r, TD_BAD_ID);
    }
    uint16_t over = td_mkrgn(&M, TD_RG_CMP);
    CHEQ(over, TD_BAD_ID);
    CHEQ(M.nrgn, TD_MAX_RGNS);
    PASS();
}
TH_REG("tdf", 33, "region overflow", tdf33);

/* ---- CB-arc emit lowers each kind to the right primitive at placed addrs ---- */

static void tdf34(void)
{
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    td_tag_t  tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t  ch  = td_link(&M, rdr, cmp, tag, /*depth=*/2);
    CHEQ(td_place_l1(&M), BC_OK);

    /* counters sit in the FIFO header right after the tile-data buffer. */
    uint32_t data_b = td_tile_bytes(M.chans[ch].tag) * (uint32_t)M.chans[ch].depth;
    uint32_t recv   = M.chans[ch].l1_off + data_b;
    uint32_t freec  = recv + 4u;
    /* producer and consumer share the tile (coords 0,0), so the NoC address
     * the atomic targets equals the bare L1 address. */

    /* RSV -> reserve_back(free): producer waits on the free counter. */
    uint16_t a_rsv = td_mkarc(&M, rdr, TD_AR_RSV, ch, 1, 0);
    rv_buf_init(&EA); rv_buf_init(&EB);
    CHEQ(td_emit_cb_arc(&M, &M.arcs[a_rsv], &EA), BC_OK);
    tt_cb_reserve_back(&EB, freec, 1);
    CHECK(bufs_eq(&EA, &EB));

    /* WAIT -> wait_front(recv): consumer waits on the recv counter. */
    uint16_t a_wait = td_mkarc(&M, cmp, TD_AR_WAIT, ch, 1, 0);
    rv_buf_init(&EA); rv_buf_init(&EB);
    CHEQ(td_emit_cb_arc(&M, &M.arcs[a_wait], &EA), BC_OK);
    tt_cb_wait_front(&EB, recv, 1);
    CHECK(bufs_eq(&EA, &EB));

    /* PUSH -> sem_inc(recv): producer bumps the consumer's recv counter. */
    uint16_t a_push = td_mkarc(&M, rdr, TD_AR_PUSH, ch, 1, 0);
    rv_buf_init(&EA); rv_buf_init(&EB);
    CHEQ(td_emit_cb_arc(&M, &M.arcs[a_push], &EA), BC_OK);
    tt_cb_push_back(&EB, recv, 0, 1);
    CHECK(bufs_eq(&EA, &EB));

    /* POP -> sem_inc(free): consumer bumps the producer's free counter. */
    uint16_t a_pop = td_mkarc(&M, cmp, TD_AR_POP, ch, 1, 0);
    rv_buf_init(&EA); rv_buf_init(&EB);
    CHEQ(td_emit_cb_arc(&M, &M.arcs[a_pop], &EA), BC_OK);
    tt_cb_pop_front(&EB, freec, 0, 1);
    CHECK(bufs_eq(&EA, &EB));
    PASS();
}
TH_REG("tdf", 34, "emit circular buffer arcs", tdf34);

/* ---- the CB emitter refuses NoC arcs (those are emitted elsewhere) ---- */

static void tdf35(void)
{
    td_init(&M, TD_TGT_TENSIX);
    uint16_t rdr = td_mkrgn(&M, TD_RG_RDR);
    uint16_t cmp = td_mkrgn(&M, TD_RG_CMP);
    td_tag_t  tag = tile_f32(32, 32, TD_LAY_INTRL);
    uint16_t  ch  = td_link(&M, rdr, cmp, tag, 2);
    td_place_l1(&M);
    uint16_t a_rd = td_mkarc(&M, rdr, TD_AR_RD, ch, 1, 0);
    rv_buf_init(&EA);
    CHEQ(td_emit_cb_arc(&M, &M.arcs[a_rd], &EA), BC_ERR_TDF);
    PASS();
}
TH_REG("tdf", 35, "emit circular buffer rejects NoC", tdf35);

/* ---- reader loop: DRAM -> L1, read barrier, back-jump ---- */

static void tdf36(void)
{
    rv_buf_init(&EA);
    CHEQ(td_emit_dma_loop(&EA, /*is_write=*/0,
                          /*dram_slot=*/0, /*ntiles_slot=*/1,
                          /*l1_buf=*/0x8100u, /*depth=*/2u,
                          /*dram_mid=*/0u, /*l1_mid=*/0u,
                          /*tile_bytes=*/4096u,
                          /*recv=*/0xA100u, /*free=*/0xA104u), BC_OK);
    uint32_t n = rv_buf_n_words(&EA);
    CHECK(n > 40u);
    /* DRAM(a3) -> L1(a4): target-low from a3, return-low from a4. */
    CHECK(has_word(&EA, rv_sw(RV_A3, RV_T0, 0x00)));
    CHECK(has_word(&EA, rv_sw(RV_A4, RV_T0, 0x0C)));
    /* read barrier polls RD_REQ_SENT (0x14) vs RD_RESP_RECEIVED (0x08). */
    CHECK(has_word(&EA, rv_lw(RV_T1, RV_T0, 0x14)));
    CHECK(has_word(&EA, rv_lw(RV_T2, RV_T0, 0x08)));
    /* loop closes with a back jump: jal zero, <negative>. */
    uint32_t last = rv_buf_data(&EA)[n - 1u];
    CHEQ(last & 0x7fu, 0x6fu);          /* JAL opcode      */
    CHEQ((last >> 7) & 0x1fu, 0u);      /* rd == zero      */
    PASS();
}
TH_REG("tdf", 36, "reader loop shape", tdf36);

/* ---- writer loop: L1 -> DRAM, write-ack barrier ---- */

static void tdf37(void)
{
    rv_buf_init(&EA);
    CHEQ(td_emit_dma_loop(&EA, /*is_write=*/1,
                          0, 1, 0x8100u, 2u, 0u, 0u, 4096u,
                          0xA100u, 0xA104u), BC_OK);
    /* L1(a4) -> DRAM(a3): target-low from a4, return-low from a3. */
    CHECK(has_word(&EA, rv_sw(RV_A4, RV_T0, 0x00)));
    CHECK(has_word(&EA, rv_sw(RV_A3, RV_T0, 0x0C)));
    /* write barrier polls WR_ACK_RECEIVED (0x04), reader's never does. */
    CHECK(has_word(&EA, rv_lw(RV_T2, RV_T0, 0x04)));
    PASS();
}
TH_REG("tdf", 37, "writer loop shape", tdf37);

/*
 * The --tdf and --tdf-fission flags have to appear in every one of main.c's
 * four mode lists. They were missing from all of them, so the "no mode
 * selected" fallback set mode_parse and the flags silently dumped an AST
 * instead of running the pass. Only driving the binary catches this.
 */

#ifdef _WIN32
#define TDF_KATH ".\\kath"
#else
#define TDF_KATH "./kath"
#endif
#define TDF_OUT "tdf_flag_out.txt"

/* Run kath with the given flag over a fixture, return its stdout. */
static const char *run_flag(const char *flag)
{
    static char buf[TH_BUFSZ];
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s %s tests/rv_shared.cu > %s 2>&1",
             TDF_KATH, flag, TDF_OUT);
    buf[0] = '\0';
    if (system(cmd) != 0) return buf;
    FILE *fp = fopen(TDF_OUT, "rb");
    if (!fp) return buf;
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = '\0';
    fclose(fp);
    return buf;
}

static void tdf38(void)
{
    const char *o = run_flag("--tdf");
    CHECK(strstr(o, "TDF module") != NULL);
    CHECK(strstr(o, "translation_unit") == NULL);
    PASS();
}
TH_REG("tdf", 38, "flag reaches pass", tdf38);

static void tdf39(void)
{
    const char *o = run_flag("--tdf-fission");
    CHECK(strstr(o, "TDF module") != NULL);
    CHECK(strstr(o, "translation_unit") == NULL);
    /* Fission rewrites the graph into the three-region pipeline. */
    CHECK(strstr(o, "RDR") != NULL);
    CHECK(strstr(o, "CMP") != NULL);
    CHECK(strstr(o, "WRT") != NULL);
    PASS();
}
TH_REG("tdf", 39, "fission flag reaches pass", tdf39);
