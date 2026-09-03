extern "C" __global__ void mf16(const float *a, const float *b, float *acc)
{ __builtin_mfma_f32_16x16x16_f16(a, b, acc); }
