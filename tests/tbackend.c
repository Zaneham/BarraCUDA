/* tbackend.c -- backend descriptor contract checks. Broken
 * descriptor at test time beats broken kernel at runtime. */

#include "tharns.h"
#include "backend.h"
#include <string.h>

static void be_list_ok(void)
{
    uint32_t n = 0;
    for (uint32_t i = 0; be_list[i] != NULL && i < 64; i++) {
        const be_desc_t *b = be_list[i];
        CHECK(b->name != NULL && b->name[0] != '\0');
        CHECK(b->is_on != NULL);
        CHECK(b->isel  != NULL);
        CHECK(b->emit  != NULL);
        n++;
    }
    CHECK(n >= 8);
    PASS();
}
TH_REG("backend", be_list_ok)

static void be_find_hits(void)
{
    static const char * const known[] = {
        "amdgpu", "nvptx", "tensix", "tensix-rv32",
        "metal", "intel-spirv", "cpu-x86-64", "cpu-rv64"
    };
    for (uint32_t i = 0; i < sizeof(known)/sizeof(known[0]); i++) {
        const be_desc_t *b = be_find(known[i]);
        CHECK(b != NULL);
        CHSTR(b->name, known[i]);
    }
    PASS();
}
TH_REG("backend", be_find_hits)

static void be_find_miss(void)
{
    CHECK(be_find(NULL) == NULL);
    CHECK(be_find("") == NULL);
    CHECK(be_find("this-is-not-a-real-backend") == NULL);
    PASS();
}
TH_REG("backend", be_find_miss)

static void be_feats_sane(void)
{
    const uint32_t all =
        BE_F_SIMT | BE_F_SCALAR | BE_F_ATOMIC | BE_F_SHARED |
        BE_F_WARP | BE_F_MFMA | BE_F_BARRIER | BE_F_DIV |
        BE_F_SCRATCH | BE_F_TRANSC | BE_F_F16 | BE_F_F64 |
        BE_F_BF16 | BE_F_MULTIOUT;
    for (uint32_t i = 0; be_list[i] != NULL && i < 64; i++) {
        CHEQ(be_list[i]->feats & ~all, 0u);
    }
    PASS();
}
TH_REG("backend", be_feats_sane)
