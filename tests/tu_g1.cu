extern __constant__ float tu_gk;

__global__ void tu_kg1(float *o) { o[0] = tu_gk * 2.0f; }
