__device__ int tu_mul7(int x) { return x * 7; }

__global__ void tu_klib(int *o) { o[0] = tu_mul7(2); }
