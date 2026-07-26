/* Constructors and destructors, which carry no return type. */

struct vec3 {
    float x, y, z;
    vec3() { x = 0.0f; y = 0.0f; z = 0.0f; }
    vec3(float v) { x = v; y = v; z = v; }
    ~vec3() { }
    float sum() { return x + y + z; }
};

struct outer {
    struct inner { inner() { } };
    int n;
    outer() { n = 0; }
};

__global__ void k(float *out)
{
    out[0] = 1.0f;
}
