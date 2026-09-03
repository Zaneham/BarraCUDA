__global__ void bglb(bool *out, const bool *in)
{
    int i = threadIdx.x;
    out[i] = in[i];
}

__global__ void bshr(bool *out, const bool *in)
{
    int i = threadIdx.x;
    __shared__ bool sh[64];
    sh[i] = in[i];
    __syncthreads();
    out[i] = sh[i];
}

__global__ void bloc(bool *out, const bool *in)
{
    int i = threadIdx.x;
    bool loc[8];
    loc[i & 7] = in[i];
    out[i] = loc[i & 7];
}

__global__ void bsize(unsigned long long *out)
{
    out[0] = sizeof(bool);
}
