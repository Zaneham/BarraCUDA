#include "tu_hdr.cuh"

__global__ void tu_kh2(float *o) { o[1] = tu_sq(3.0f) + tu_bias; }
