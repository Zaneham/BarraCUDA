/* test_shfl_runner.hip — execution test for warp shuffle and vote intrinsics */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 64
#define BLOCK 64
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP error %d: %s at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file.hsaco>\n", argv[0]); return 1; }
    hipModule_t mod;
    HIP_CHECK(hipModuleLoad(&mod, argv[1]));
    hipFunction_t kern;
    HIP_CHECK(hipModuleGetFunction(&kern, mod, "test_shfl"));
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = (i < 16) ? 1.0f : -1.0f; h_out[i] = -1.0f; }
    float *d_out, *d_in;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_in,  N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(float), hipMemcpyHostToDevice));
    struct { float *out; float *in; } args = { d_out, d_in };
    size_t arg_size = sizeof(args);
    void *config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END
    };
    HIP_CHECK(hipModuleLaunchKernel(kern, 1,1,1, BLOCK,1,1, 0,0, NULL, config));
    HIP_CHECK(hipDeviceSynchronize());
    printf("[ok] kernel launched\n");
    HIP_CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));
    int pass = 0, fail = 0;
    for (int i = 0; i < N; i++) {
        float s0 = (i < 63) ? h_in[i + 1] : h_in[i];
        float s1 = h_in[0];
        float s2 = (i > 0) ? h_in[i - 1] : h_in[0];
        float s3 = (i < 63) ? h_in[i + 1] : h_in[i];
        float s4 = h_in[i ^ 1];
        /* __ballot(val > 0.0f) — first 16 lanes positive -> mask 0x0000FFFFu (65535) */
        unsigned int b = 0x0000FFFFu;
        int a0 = 1; /* __any(val > 0.0f) */
        int a1 = 0; /* __all(val > 0.0f) */
        float expected = s0 + s1 + s2 + s3 + s4 + (float)b + (float)a0 + (float)a1;
        if (fabsf(h_out[i] - expected) < 0.5f) pass++;
        else { if (fail < 5) printf("[FAIL] i=%d: got %.1f expected %.1f\n", i, h_out[i], expected); fail++; }
    }
    printf("=== %d/%d passed", pass, N);
    if (fail) printf(", %d FAILED", fail);
    printf(" ===\n");
    hipFree(d_out); hipFree(d_in); hipModuleUnload(mod);
    return fail ? 1 : 0;
}
