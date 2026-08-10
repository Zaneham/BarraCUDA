/* tinline.c -- __device__ call inlining (issue #101).
 * The GPU backends have no calling convention for device functions, so the
 * inliner must splice every call away before isel. The scalar CPU backend
 * emits real calls and must be left alone. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* On a GPU target every device call is inlined, so no call survives in the IR
 * (the standalone device bodies are inlined into their callers too). */
static void inl01(void)
{
    int rc = th_run(BC_BIN " --amdgpu --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);
    PASS();
}

/* And the kernel compiles the whole way to a .hsaco with the calls gone. */
static void inl02(void)
{
    const char *out = "test_inline.hsaco";
    int rc = th_run(BC_BIN " --amdgpu-bin tests/device_calls.cu "
                    "-o test_inline.hsaco", obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(th_exist(out));
    remove(out);
    PASS();
}

/* NVIDIA and Tensix isel cannot emit a call either, so they inline too. */
static void inl03(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);

    rc = th_run(BC_BIN " --tensix --ir tests/device_calls.cu",
                obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);
    PASS();
}

/* The CPU backend has a real SysV call ABI, so the inliner must not touch it:
 * the device calls stay as calls. */
static void inl04(void)
{
    int rc = th_run(BC_BIN " --cpu --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") != NULL);
    PASS();
}

TH_REG("inl", 1, "a GPU kernel ends up with no calls", inl01);
TH_REG("inl", 2, "a GPU binary inlines", inl02);
TH_REG("inl", 3, "other GPU targets inline too", inl03);
TH_REG("inl", 4, "CPU keeps its calls", inl04);
