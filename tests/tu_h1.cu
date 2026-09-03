#include "tu_hdr.cuh"

__global__ void tu_kh1(float *o) { o[0] = tu_sq(2.0f) + tu_bias; }
