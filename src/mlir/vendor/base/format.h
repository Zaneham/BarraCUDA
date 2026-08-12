#pragma once

#include <base/types.h>
#include <base/stdarg.h>
#include <base/string.h>
#include <base/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARG_INT8,
    ARG_UINT8,
    ARG_INT16,
    ARG_UINT16,
    ARG_INT32,
    ARG_UINT32,
    ARG_INT64,
    ARG_UINT64,
    ARG_DOUBLE,
    ARG_STRING,
    ARG_STRING2,
    ARG_POINTER,
    ARG_VECTOR_INT64
} ArgType;

string format_explicit_varg(Arena *arena, string fmt, size_t arg_count,
        va_list ap);
string format_explicit(Arena *arena, string fmt, size_t arg_count, ...);

#define GET_ARG_COUNT(_0, _1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define COUNT_ARGS(...) GET_ARG_COUNT(0 __VA_OPT__(,) __VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)

// Map all types to explicit ArgType enum values
// Using fixed-width types for consistency across platforms
#define A(x) _Generic((x), \
    char*:   ARG_STRING, \
    string:  ARG_STRING2, \
    double:  ARG_DOUBLE, \
    int8_t:  ARG_INT8, \
    uint8_t: ARG_UINT8, \
    int16_t: ARG_INT16, \
    uint16_t: ARG_UINT16, \
    int32_t: ARG_INT32, \
    uint32_t: ARG_UINT32, \
    int64_t: ARG_INT64, \
    uint64_t: ARG_UINT64, \
    void*:   ARG_POINTER, \
    Arena*:  ARG_POINTER, \
    vector_i64: ARG_VECTOR_INT64 \
    ), (x)

#define APPLY_A0()
#define APPLY_A1(a) A(a)
#define APPLY_A2(a, b) A(a), A(b)
#define APPLY_A3(a, b, c) A(a), A(b), A(c)
#define APPLY_A4(a, b, c, d) A(a), A(b), A(c), A(d)
#define APPLY_A5(a, b, c, d, e) A(a), A(b), A(c), A(d), A(e)
#define APPLY_A6(a, b, c, d, e, f) A(a), A(b), A(c), A(d), A(e), A(f)
#define APPLY_A7(a, b, c, d, e, f, g) A(a), A(b), A(c), A(d), A(e), A(f), A(g)
#define APPLY_A8(a, b, c, d, e, f, g, h) A(a), A(b), A(c), A(d), A(e), A(f), A(g), A(h)

#define APPLY_A_FOR_COUNT_0 APPLY_A0
#define APPLY_A_FOR_COUNT_1 APPLY_A1
#define APPLY_A_FOR_COUNT_2 APPLY_A2
#define APPLY_A_FOR_COUNT_3 APPLY_A3
#define APPLY_A_FOR_COUNT_4 APPLY_A4
#define APPLY_A_FOR_COUNT_5 APPLY_A5
#define APPLY_A_FOR_COUNT_6 APPLY_A6
#define APPLY_A_FOR_COUNT_7 APPLY_A7
#define APPLY_A_FOR_COUNT_8 APPLY_A8

#define CONCAT_AFTER_EXPAND(prefix, count) prefix ## count
#define APPLY_WITH_COUNT(count, ...) CONCAT_AFTER_EXPAND(APPLY_A_FOR_COUNT_, count)(__VA_ARGS__)

#define format(arena, fmt, ...) \
    format_explicit(arena, fmt, COUNT_ARGS(__VA_ARGS__) __VA_OPT__(,) APPLY_WITH_COUNT(COUNT_ARGS(__VA_ARGS__) __VA_OPT__(,) __VA_ARGS__))
#ifdef __cplusplus
}
#endif
