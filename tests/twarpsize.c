#include "tharns.h"

static void wsz01(void) {
    char obuf[TH_BUFSZ];
    
    // 1. Verify gfx942 produces Wave64 (constant 64)
    th_run(BC_BIN " --ir --amdgpu --gfx942 tests/test_warp_size.cu", obuf, sizeof(obuf));
    CHECK(strstr(obuf, "store i32 64") != NULL);

    // 2. Verify gfx1100 produces Wave32 (constant 32)
    th_run(BC_BIN " --ir --amdgpu --gfx1100 tests/test_warp_size.cu", obuf, sizeof(obuf));
    CHECK(strstr(obuf, "store i32 32") != NULL);

    PASS();
}
TH_REG("wsz", 1, "warpsize", wsz01)
