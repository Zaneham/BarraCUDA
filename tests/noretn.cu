_Noreturn void nrt_die(const char *file, int line, const char *fmt, ...);

[[noreturn]] void nrt_stop(int code);

__device__ int nrt_bump(int a, [[maybe_unused]] int slack)
{
    return a + 1;
}

__global__ void nrt_kern(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = nrt_bump(data[i], 0);
    }
}
