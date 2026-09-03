template<typename T>
__global__ void tu_tsc(T *o, T k) { o[threadIdx.x] = o[threadIdx.x] * k; }

void tu_h1(float *d) { tu_tsc<<<4, 256>>>(d, 2.0f); }
