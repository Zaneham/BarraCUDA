/* Bit counting: the half of ballot that reads the mask back.
 * __ffs is 1-based and answers 0 for a zero input, which is the one case
 * that needs a guard rather than a bare ctz. */

__global__ void bitcount(unsigned *out, const unsigned *in)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned x = in[i];
    out[i * 4 + 0] = (unsigned)__popc(x);
    out[i * 4 + 1] = (unsigned)__clz(x);
    out[i * 4 + 2] = (unsigned)__ffs(x);
    out[i * 4 + 3] = __brev(x);
}

/* Stream compaction, the reason any of this is here: each lane's output
 * slot is the number of lanes below it that also passed. */
__global__ void compact(unsigned *out, const unsigned *in, unsigned *n)
{
    int i = threadIdx.x;
    unsigned keep = in[i] != 0u;
    unsigned mask = __ballot(keep);
    unsigned below = mask & ((1u << threadIdx.x) - 1u);
    if (keep) out[__popc(below)] = in[i];
    if (threadIdx.x == 0) *n = (unsigned)__popc(mask);
}
