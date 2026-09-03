extern "C" __global__ void mfbf(const float *a, const float *b, float *acc)
{ __builtin_mfma_f32_16x16x16_bf16(a, b, acc); }
