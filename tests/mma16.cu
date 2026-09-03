/* Every mma.sync shape Booth knows, one kernel each. */
extern "C" __global__ void mma16(const short *a, const short *b, float *d,
                                 int lda, int ldb, int ldd)
{ __builtin_mma_m16n16k16_f16(a, lda, b, ldb, d, ldd); }

extern "C" __global__ void mmab16(const short *a, const short *b, float *d,
                                  int lda, int ldb, int ldd)
{ __builtin_mma_m16n16k16_bf16(a, lda, b, ldb, d, ldd); }

extern "C" __global__ void mma8(const short *a, const short *b, float *d,
                                int lda, int ldb, int ldd)
{ __builtin_mma_m16n16k8_f16(a, lda, b, ldb, d, ldd); }

extern "C" __global__ void mmab8(const short *a, const short *b, float *d,
                                 int lda, int ldb, int ldd)
{ __builtin_mma_m16n16k8_bf16(a, lda, b, ldb, d, ldd); }
