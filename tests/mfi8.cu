extern "C" __global__ void mfi8(const int *a, const int *b, int *acc)
{ __builtin_mfma_i32_16x16x16_i8(a, b, acc); }
