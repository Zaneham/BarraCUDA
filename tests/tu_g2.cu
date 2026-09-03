__constant__ float tu_gk;

__global__ void tu_kg2(float *o) { o[1] = tu_gk * 3.0f; }
