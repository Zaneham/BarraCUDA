#define PPV_DROP(...)
#define PPV_KEEP(first, ...) first + __VA_ARGS__
#define PPV_NAME(...) #__VA_ARGS__
#define PPV_SHIFT 2 // shifting by a byte and a bit

#define PPV_WRAP(decl, hint) decl

PPV_DROP(9, 9)

PPV_WRAP(
    __device__ int ppv_add(int a, int b), // the hint is dropped
    "use ppv_add2 instead");

__device__ int ppv_add(int a, int b)
{
    return PPV_KEEP(a, b) >> PPV_SHIFT;
}

__global__ void ppv_kern(int *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = ppv_add(data[i], PPV_SHIFT);
    }
}

const char *ppv_text = PPV_NAME(a, b);
