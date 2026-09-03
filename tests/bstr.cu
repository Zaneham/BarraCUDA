__global__ void ccopy(char *out, const char *in)
{
    int i = threadIdx.x;
    out[i] = in[i];
}
