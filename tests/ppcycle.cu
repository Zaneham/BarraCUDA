#include "ppcyc_a.cuh"
#include "ppcyc_b.cuh"

__global__ void ppcyc_kern(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] + PPCYC_A + PPCYC_B;
    }
}
