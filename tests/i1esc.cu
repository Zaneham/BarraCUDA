__device__ int ident(int x) { return x; }

__global__ void i1esc(int *out, int a, int b)
{
    __shared__ int shr;

    out[0] = (a == 0) || (b == 0);
    out[1] = (a == 0) && (b == 0);
    out[2] = (a == 0) + 5;
    out[3] = (b == 0) * 3;
    out[4] = ident(a == 0);
    out[5] = !a;
    out[6] = (int)((float)(a == 0) * 4.0f);

    shr = (a == b);
    __syncthreads();
    out[7] = shr;

    atomicAdd(&out[8], (b == 0));
}
