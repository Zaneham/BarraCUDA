static __device__ int scale(int x) { return x + 100; }

__global__ void tu_kb(int *o) { o[threadIdx.x] = scale(1); }
