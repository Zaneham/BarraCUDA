#include <base/types.h>
#include <base/math.h>

static const float S6 = -0.01445205f;
static const float S5 =  0.09838380f;
static const float S4 = -0.01243614f;
static const float S3 = -0.64157278f;
static const float S2 = -0.00077364f;
static const float S1 =  1.5708527f;
static const float S0 = -0.000000992f;

static const double DPI_D = 6.28318530717958647692;
static const double PI_D  = 3.14159265358979323846;
static const float  PI2_F = 1.57079632679f;
static const float  PI_F  = 3.14159265f;

static inline float poly_sincos(float z) {
    float p = S6 * z + S5;
    p = p * z + S4;
    p = p * z + S3;
    p = p * z + S2;
    p = p * z + S1;
    p = p * z + S0;
    return p;
}

static inline void reduce_to_quarter(float x, float* y, float* sin_sign, float* cos_sign) {
    double xd = (double)x;
    double q = xd / DPI_D;
    double k = base_round(q);
    double rd = xd - k * DPI_D;
    if (rd > PI_D) {
        rd -= DPI_D;
    } else if (rd < -PI_D) {
        rd += DPI_D;
    }
    float r = (float)rd;
    bool flip_sin = (r < 0.0f);
    if (flip_sin) {
        r = -r;
    }
    *sin_sign = flip_sin ? -1.0f : 1.0f;
    *cos_sign = 1.0f;
    if (r > PI2_F) {
        *y = PI_F - r;
        *cos_sign = -1.0f;
    } else {
        *y = r;
    }
}

float fast_sinf(float x) {
    if (x == 0.0f) return 0.0f;
    float y, sin_s, cos_s;
    reduce_to_quarter(x, &y, &sin_s, &cos_s);
    float z = y / PI2_F;
    float sy = poly_sincos(z);
    return sin_s * sy;
}

float fast_cosf(float x) {
    if (x == 0.0f) return 1.0f;
    float y, sin_s, cos_s;
    reduce_to_quarter(x, &y, &sin_s, &cos_s);
    float z = y / PI2_F;
    float cy = poly_sincos(1.0f - z);
    return cos_s * cy;
}

float fast_tanf(float x) {
    float s = fast_sinf(x);
    float c = fast_cosf(x);
    if (c == 0.0f) {
        return (s < 0.0f ? -INFINITY : INFINITY);
    }
    return s / c;
}
