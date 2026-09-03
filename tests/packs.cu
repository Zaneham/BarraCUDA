#include <stdio.h>

__device__ float add3(float a, float b, float c)
{
    return a + b + c;
}

template<class ... A>
__global__ void pk_sum(float *out, A... a)
{
    out[0] = (0.0f + ... + (float)a);
}

template<typename T, typename... A>
__global__ void pk_cnt(int *out, T t, A... a)
{
    out[0] = (int)sizeof...(a) + (int)sizeof...(A) + (int)t;
}

template<typename... A>
__global__ void pk_call(float *out, A... a)
{
    out[0] = add3(a...);
}

template<typename... A>
__global__ void pk_any(int *out, A... a)
{
    int r = 0;
    if ((((int)a == 0) || ...)) r = 1;
    out[0] = r;
}

template<typename... A>
__global__ void pk_seq(int *out, A... a)
{
    int r = 0;
    (..., (r = r * 10 + (int)a));
    out[0] = r;
}

template<typename... A>
__device__ void pk_drop(A&&...)
{
}

template<typename...>
__device__ int pk_none(void)
{
    return 0;
}

template<int... N, typename T>
__device__ T pk_first(T x)
{
    return x;
}

__device__ int pk_varg(int a, ...)
{
    return a;
}

int main(void)
{
    float *d;
    int *n;
    cudaMalloc(&d, 64);
    cudaMalloc(&n, 64);

    pk_sum<<<1, 1>>>(d, 1.0f, 2.0f, 3.0f, 4.0f);
    pk_sum<<<1, 1>>>(d, 1.0f, 2.0f);
    pk_cnt<<<1, 1>>>(n, 1, 2, 3);
    pk_call<<<1, 1>>>(d, 1.5f, 2.5f, 3.5f);
    pk_any<<<1, 1>>>(n, 1, 0, 2);
    pk_seq<<<1, 1>>>(n, 1, 2, 3);

    cudaFree(d);
    cudaFree(n);
    return 0;
}
