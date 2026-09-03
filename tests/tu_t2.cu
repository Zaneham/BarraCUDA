template<typename T>
__global__ void tu_tsc(T *o, T k) { o[threadIdx.x] = o[threadIdx.x] * k; }

void tu_h2(float *d) { tu_tsc<<<8, 128>>>(d, 3.0f); }
