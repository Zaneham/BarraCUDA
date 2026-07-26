/* test_guard.cu -- a divergent early-return guard, the opening line of most
 * real kernels. It must lower to s_andn2 on EXEC, not s_and_saveexec then
 * s_endpgm: the latter ends the whole wave and a ragged launch loses the
 * in-range tail of its last wave. */
extern "C" __global__ void guard(float *y, float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a * x[i] + y[i];
}
