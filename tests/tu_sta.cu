static __device__ int scale(int x) { return x + 1; }

__global__ void tu_ka(int *o) { o[threadIdx.x] = scale(1); }
