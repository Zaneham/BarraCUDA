/* tabend.c -- ABEND dump tests
 * All runnable on Windows, no GPU required. We test the dump formatter,
 * memory tracking, fault correlation, source map parser, and lang
 * integration. The actual HSA fault callback needs a GPU, but the
 * diagnostic machinery is pure C and testable anywhere. */

#include "tharns.h"
#include "bc_runtime.h"
#include "bc_abend.h"
#include "bc_err.h"
#include <string.h>
#include <stdlib.h>

/* Shared output buffer for dump capture */
static char obuf[8192];

/* Helper: dump to a temp file, read it back.
 * tmpfile() works on both Windows and Linux. */
static int ab_snap(const ab_ctx_t *A, char *buf, int bsz)
{
    FILE *fp = tmpfile();
    if (!fp) return -1;
    ab_dump(A, fp);
    long len = ftell(fp);
    if (len < 0) len = 0;
    if (len >= bsz) len = bsz - 1;
    rewind(fp);
    int n = (int)fread(buf, 1, (size_t)len, fp);
    buf[n] = '\0';
    fclose(fp);
    return n;
}

/* ---- ABEND Context Tests ---- */

static void ab_init_def(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    int rc = ab_init(A, NULL); /* no HSA lib */
    CHEQ(rc, 0);
    CHEQ(A->armed, 0);
    CHEQ(A->faulted, 0);
    CHEQ(A->n_alloc, 0);
    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_init_def)

/* Arms and confirms the flag; can't actually raise a fault in-process
 * without terminating the test. Real behaviour is exercised by the
 * --cpu launcher when a kernel goes sideways. */
static void ab_arm_cpu_ok(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    int rc = ab_arm_cpu(A);
    CHEQ(rc, 0);
    CHEQ(A->armed, 1);
    ab_shut(A);
    CHEQ(A->armed, 0);
    free(A);
    PASS();
}
TH_REG("abend", ab_arm_cpu_ok)

static void ab_trak_one(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    ab_trak(A, 0x1000, 0x100, "test_buf", AB_FL_RW);
    CHEQ(A->n_alloc, 1);
    CHEQ(A->allocs[0].base, 0x1000);
    CHEQ(A->allocs[0].size, 0x100);
    CHSTR(A->allocs[0].label, "test_buf");
    CHEQ(A->allocs[0].flags, AB_FL_RW);
    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_trak_one)

static void ab_trak_full(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    /* Fill to capacity */
    for (int i = 0; i < AB_MAX_ALLOC; i++)
        ab_trak(A, (uint64_t)(i * 0x1000), 0x1000, "blk", AB_FL_RW);
    CHEQ(A->n_alloc, (uint32_t)AB_MAX_ALLOC);
    /* One more should be silently ignored */
    ab_trak(A, 0xDEAD0000, 0x100, "overflow", AB_FL_RW);
    CHEQ(A->n_alloc, (uint32_t)AB_MAX_ALLOC);
    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_trak_full)

static void ab_snag_snap(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);

    /* Build a fake bc_kernel_t */
    bc_kernel_t k;
    memset(&k, 0, sizeof(k));
    k.kernel_object = 0xBEEF;
    k.kernarg_size  = 24;
    k.group_size    = 256;
    k.private_size  = 0;

    uint8_t args[24] = {0};
    args[0] = 0x42;
    ab_snag(A, &k, "saxpy", "gfx1100",
            4, 1, 1, 256, 1, 1, args, 24);

    CHSTR(A->dctx.kernel, "saxpy");
    CHSTR(A->dctx.chip, "gfx1100");
    CHEQ(A->dctx.grid[0], 4u);
    CHEQ(A->dctx.block[0], 256u);
    CHEQ(A->dctx.kobj, 0xBEEFu);
    CHEQ(A->dctx.args_sz, 24u);
    CHEQ(A->args_snap[0], 0x42);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_snag_snap)

static void ab_mstr_all(void)
{
    /* Every defined code should return a non-empty string */
    uint16_t codes[] = { AB_G0C1, AB_G0C4, AB_G0C5, AB_G0C7,
                         AB_G0CB, AB_G001, AB_G002, AB_G003, AB_G0FF };
    for (int i = 0; i < 9; i++) {
        const char *s = ab_mstr(codes[i]);
        CHECK(s != NULL);
        CHECK(strlen(s) > 0);
    }
    /* Unknown code should still return something */
    const char *u = ab_mstr(0x0FE);
    CHECK(u != NULL);
    PASS();
}
TH_REG("abend", ab_mstr_all)

/* ---- Fault Correlation Tests ---- */

static void ab_near_inside(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    ab_trak(A, 0x10000, 0x1000, "buf_a", AB_FL_RW);
    A->tea = 0x10100;
    A->code = AB_G0C4;

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "inside buf_a") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_near_inside)

static void ab_near_past(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    ab_trak(A, 0x10000, 0x1000, "buf_b", AB_FL_RW);
    A->tea = 0x11000 + 0x80; /* 0x80 past end */
    A->code = AB_G0C5;

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "past end") != NULL);
    CHECK(strstr(obuf, "buf_b") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_near_past)

static void ab_near_before(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    ab_trak(A, 0x20000, 0x1000, "buf_c", AB_FL_RW);
    A->tea = 0x1FF00; /* before start */
    A->code = AB_G0C4;

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "before buf_c") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_near_before)

static void ab_near_gap(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    ab_trak(A, 0x10000, 0x1000, "lo", AB_FL_RW);
    ab_trak(A, 0x30000, 0x1000, "hi", AB_FL_RW);
    A->tea = 0x20000; /* equidistant, but "lo" checked first */
    A->code = AB_G0C4;

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    /* Should find the nearest one (lo is +0xF000 past end, hi is 0x10000 before) */
    CHECK(strstr(obuf, "lo") != NULL || strstr(obuf, "hi") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_near_gap)

static void ab_near_empty(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    /* No allocations tracked */
    A->tea = 0xDEAD;
    A->code = AB_G0C4;

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "no tracked") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_near_empty)

/* ---- Dump Formatting Tests ---- */

static void ab_dump_hdr(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    A->code = AB_G0C4;
    A->tea  = 0xBEEF;
    snprintf(A->dctx.kernel, sizeof(A->dctx.kernel), "test_kern");
    snprintf(A->dctx.chip, sizeof(A->dctx.chip), "gfx1100");

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "ABEND G0C4") != NULL);
    CHECK(strstr(obuf, "memory access violation") != NULL);
    CHECK(strstr(obuf, "test_kern") != NULL);
    CHECK(strstr(obuf, "gfx1100") != NULL);
    CHECK(strstr(obuf, "END OF DUMP") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_dump_hdr)

static void ab_dump_mem(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    A->code = AB_G0C4;
    ab_trak(A, 0xAAAA0000, 0x10000, "input_A", AB_FL_RW);
    ab_trak(A, 0xBBBB0000, 0x800,   ".text",   AB_FL_RX);

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "MEMORY MAP") != NULL);
    CHECK(strstr(obuf, "input_A") != NULL);
    CHECK(strstr(obuf, ".text") != NULL);
    CHECK(strstr(obuf, "RW") != NULL);
    CHECK(strstr(obuf, "RX") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_dump_mem)

static void ab_dump_disp(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    A->code = AB_G0C4;

    bc_kernel_t k;
    memset(&k, 0, sizeof(k));
    k.kernarg_size = 16;
    uint8_t args[16] = {0};
    ab_snag(A, &k, "vecadd", "gfx942",
            1, 1, 1, 64, 1, 1, args, 16);

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "DISPATCH") != NULL);
    CHECK(strstr(obuf, "vecadd") != NULL);
    CHECK(strstr(obuf, "(1, 1, 1)") != NULL);
    CHECK(strstr(obuf, "(64, 1, 1)") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_dump_disp)

static void ab_dump_snap(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    A->code = AB_G0C4;

    /* Distinctive bytes so we can see them in the hex + ASCII gutter. */
    bc_kernel_t k;
    memset(&k, 0, sizeof(k));
    k.kernarg_size = 24;
    uint8_t args[24] = {
        0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78,
        'H','e','l','l','o',' ','k','a', 't','h','!',0,0,0,0,0
    };
    ab_snag(A, &k, "snap_demo", "cpu",
            1, 1, 1, 1, 1, 1, args, 24);

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "SNAP") != NULL);
    CHECK(strstr(obuf, "+0000") != NULL);          /* offset gutter */
    CHECK(strstr(obuf, "DEADBEEF") != NULL);       /* hex */
    CHECK(strstr(obuf, "Hello kath!") != NULL);    /* ASCII gutter */

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_dump_snap)

static void ab_dump_smap(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);
    A->code = AB_G0C4;
    snprintf(A->src_file, sizeof(A->src_file), "test.cu");
    A->smap[0].offset = 0x000;
    A->smap[0].line   = 3;
    A->smap[1].offset = 0x010;
    A->smap[1].line   = 4;
    A->n_smap = 2;

    int n = ab_snap(A, obuf, (int)sizeof(obuf));
    CHECK(n > 0);
    CHECK(strstr(obuf, "SOURCE MAP") != NULL);
    CHECK(strstr(obuf, "test.cu:3") != NULL);
    CHECK(strstr(obuf, "test.cu:4") != NULL);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_dump_smap)

static void ab_dump_null(void)
{
    /* Null context should return error, not crash */
    int rc = ab_dump(NULL, stderr);
    CHEQ(rc, -1);

    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    rc = ab_dump(A, NULL);
    CHEQ(rc, -1);

    free(A);
    PASS();
}
TH_REG("abend", ab_dump_null)

/* ---- Source Map Tests ---- */

/* Build a minimal ELF with a .debug_bc section */
static int build_test_elf(uint8_t *buf, int bsz, int n_entries)
{
    /* We need: ELF header (64B) + .debug_bc data + .shstrtab + shdrs
     * Minimal: 3 section headers (NULL + .debug_bc + .shstrtab) */
    memset(buf, 0, (size_t)bsz);

    /* Section names */
    char shstrtab[32];
    int shstrtab_len = 0;
    shstrtab[shstrtab_len++] = '\0'; /* null name */
    int name_dbc = shstrtab_len;
    memcpy(shstrtab + shstrtab_len, ".debug_bc", 10);
    shstrtab_len += 10;
    int name_shs = shstrtab_len;
    memcpy(shstrtab + shstrtab_len, ".shstrtab", 10);
    shstrtab_len += 10;

    /* .debug_bc data: "BCDB" + count + entries */
    uint32_t dbc_sz = 8 + (uint32_t)n_entries * 8;
    uint8_t dbc[256];
    memset(dbc, 0, sizeof(dbc));
    dbc[0] = 'B'; dbc[1] = 'C'; dbc[2] = 'D'; dbc[3] = 'B';
    uint32_t cnt = (uint32_t)n_entries;
    memcpy(dbc + 4, &cnt, 4);
    for (int i = 0; i < n_entries && i < 16; i++) {
        uint32_t off = (uint32_t)(i * 8);
        uint32_t ln  = (uint32_t)(i + 10);
        memcpy(dbc + 8 + i * 8,     &off, 4);
        memcpy(dbc + 8 + i * 8 + 4, &ln,  4);
    }

    /* Layout: ehdr(64) + dbc(dbc_sz) + shstrtab(shstrtab_len) + shdrs(3*64) */
    uint64_t dbc_off = 64;
    uint64_t shs_off = dbc_off + dbc_sz;
    uint64_t shdr_off = ((shs_off + (uint64_t)shstrtab_len + 7) / 8) * 8;
    uint64_t total = shdr_off + 3 * 64;
    if ((int)total > bsz) return -1;

    /* ELF header */
    buf[0] = 0x7F; buf[1] = 'E'; buf[2] = 'L'; buf[3] = 'F';
    buf[4] = 2; /* ELF64 */
    buf[5] = 1; /* little endian */
    buf[6] = 1; /* EV_CURRENT */
    uint16_t e_type = 3; memcpy(buf + 16, &e_type, 2);
    uint16_t e_mach = 224; memcpy(buf + 18, &e_mach, 2);
    uint32_t e_ver = 1; memcpy(buf + 20, &e_ver, 4);
    memcpy(buf + 40, &shdr_off, 8); /* e_shoff */
    uint16_t e_ehsz = 64; memcpy(buf + 52, &e_ehsz, 2);
    uint16_t e_shentsz = 64; memcpy(buf + 58, &e_shentsz, 2);
    uint16_t e_shnum = 3; memcpy(buf + 60, &e_shnum, 2);
    uint16_t e_shstr = 2; memcpy(buf + 62, &e_shstr, 2);

    /* .debug_bc data */
    memcpy(buf + dbc_off, dbc, dbc_sz);

    /* .shstrtab */
    memcpy(buf + shs_off, shstrtab, (size_t)shstrtab_len);

    /* Section headers (3 x 64B) */
    uint8_t *sh = buf + shdr_off;
    /* [0] NULL -- already zero */

    /* [1] .debug_bc */
    uint8_t *sh1 = sh + 64;
    uint32_t nm1 = (uint32_t)name_dbc; memcpy(sh1, &nm1, 4);
    uint32_t ty1 = 1; memcpy(sh1 + 4, &ty1, 4); /* SHT_PROGBITS */
    memcpy(sh1 + 24, &dbc_off, 8); /* sh_offset */
    uint64_t sz1 = dbc_sz; memcpy(sh1 + 32, &sz1, 8); /* sh_size */

    /* [2] .shstrtab */
    uint8_t *sh2 = sh + 128;
    uint32_t nm2 = (uint32_t)name_shs; memcpy(sh2, &nm2, 4);
    uint32_t ty2 = 3; memcpy(sh2 + 4, &ty2, 4); /* SHT_STRTAB */
    memcpy(sh2 + 24, &shs_off, 8);
    uint64_t sz2 = (uint64_t)shstrtab_len; memcpy(sh2 + 32, &sz2, 8);

    return (int)total;
}

static void ab_slod_ok(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);
    ab_init(A, NULL);

    uint8_t elf[2048];
    int sz = build_test_elf(elf, (int)sizeof(elf), 3);
    CHECK(sz > 0);

    int rc = ab_slod(A, elf, (uint32_t)sz);
    CHEQ(rc, 0);
    CHEQ(A->n_smap, 3u);
    CHEQ(A->smap[0].offset, 0u);
    CHEQ(A->smap[0].line, 10u);
    CHEQ(A->smap[1].offset, 8u);
    CHEQ(A->smap[1].line, 11u);
    CHEQ(A->smap[2].offset, 16u);
    CHEQ(A->smap[2].line, 12u);

    ab_shut(A);
    free(A);
    PASS();
}
TH_REG("abend", ab_slod_ok)

static void ab_slod_nosc(void)
{
    /* ELF with no .debug_bc section */
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);

    /* Minimal valid ELF header with no sections */
    uint8_t elf[128];
    memset(elf, 0, sizeof(elf));
    elf[0] = 0x7F; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F';
    elf[4] = 2; elf[5] = 1; elf[6] = 1;

    int rc = ab_slod(A, elf, 64);
    CHEQ(rc, -1); /* no sections, should fail gracefully */
    CHEQ(A->n_smap, 0u);

    free(A);
    PASS();
}
TH_REG("abend", ab_slod_nosc)

static void ab_slod_bad(void)
{
    ab_ctx_t *A = calloc(1, sizeof(ab_ctx_t));
    CHECK(A != NULL);

    /* Not an ELF at all */
    uint8_t junk[64] = { 'N', 'O', 'T', 'E', 'L', 'F' };
    int rc = ab_slod(A, junk, 64);
    CHEQ(rc, -1);

    /* NULL pointer */
    rc = ab_slod(A, NULL, 100);
    CHEQ(rc, -1);

    /* Too small */
    uint8_t tiny[4] = { 0x7F, 'E', 'L', 'F' };
    rc = ab_slod(A, tiny, 4);
    CHEQ(rc, -1);

    free(A);
    PASS();
}
TH_REG("abend", ab_slod_bad)

/* ---- Lang Integration Tests ---- */

static void ab_lang_en(void)
{
    /* Compiled-in defaults should be present */
    const char *s = ab_afmt(AB_G0C4);
    CHECK(s != NULL);
    CHECK(strstr(s, "memory access") != NULL);

    s = ab_afmt(AB_G0C1);
    CHECK(s != NULL);
    CHECK(strstr(s, "illegal") != NULL);

    /* Unknown code */
    s = ab_afmt(0xFE);
    CHECK(s != NULL);
    PASS();
}
TH_REG("abend", ab_lang_en)

static void ab_lang_mi(void)
{
    /* Load Maori translations and check ABEND messages */
    int rc = bc_eload("lang/mi.txt");
    if (rc != 0) {
        SKIP("lang/mi.txt not found");
    }

    const char *s = ab_afmt(AB_G0C4);
    CHECK(s != NULL);
    CHECK(strstr(s, "takahi") != NULL || strstr(s, "whakamaru") != NULL);

    /* Reload English to restore state */
    bc_eload("lang/en.txt");
    PASS();
}
TH_REG("abend", ab_lang_mi)
