struct Pad { char t; int v; };

__global__ void bpad(int *out, const Pad *in)
{
    int i = threadIdx.x;
    out[i] = in[i].v;
}
