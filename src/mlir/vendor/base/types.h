#pragma once

// Basic integer types and a few macros.
//
// corec defines its own size_t/uint8_t/int64_t/etc. typedefs and integer
// limit macros. These typedefs use the same underlying primitive types as
// every libc we target (size_t = unsigned long on 64-bit Unix and
// uint64_t/Windows; int64_t = long long; uint8_t = unsigned char; ...).
// C11 and C++ both permit redundant identical typedef redeclarations, so
// these definitions can coexist with system <stddef.h>/<stdint.h>/<cstdint>
// in the same TU.
//
// Macros (NULL, SIZE_MAX, INT*_MAX, FLT_MAX, true/false) are guarded with
// #ifndef so they don't clash with the system definitions if those happen
// to be included first (e.g. via libc++ headers in a hosted C++ TU).
//
// The result: corec headers never include any system or libc header, while
// remaining safe to include in hosted C/C++ TUs (e.g. when bridging the
// mlir C API to upstream LLVM/MLIR).

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

// On Linux x86_64, long is 64 bits, so use unsigned long for uint64_t
// On other platforms (especially Windows), long is 32 bits, so use unsigned long long
#if defined(__linux__) && defined(__x86_64__)
typedef unsigned long uint64_t;
#else
typedef unsigned long long uint64_t;
#endif

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;

#if defined(__linux__) && defined(__x86_64__)
typedef long int64_t;
#else
typedef signed long long int64_t;
#endif

// Pointer-sized integer type and size types

#if defined(_WIN32) && defined(_WIN64)
    // For 64 bit Windows the long is 4 bytes, but pointer is 8 bytes
    typedef uint64_t uintptr_t;
    typedef int64_t ptrdiff_t;
#else
    // For 32 bit platforms and wasm64 the long and a pointer is 4 bytes, for
    // 64 bit macOS/Linux the long and pointer is 8 bytes
    typedef unsigned long uintptr_t;
    typedef long ptrdiff_t;
#endif

#if defined(_WIN32) && defined(_WIN64)
    // 64 bit Windows has 8 byte size_t (but 4 byte long)
    typedef uint64_t size_t;
    typedef int64_t ssize_t;
#else
    // All other platforms have long and size_t the same number of bytes (4 or
    // 8)
    typedef unsigned long size_t;
    typedef signed long ssize_t;
#endif

#ifndef NULL
#  ifdef __cplusplus
#    define NULL nullptr
#  else
#    define NULL ((void*)0)
#  endif
#endif

#ifndef __cplusplus
// In C, bool is provided by <stdbool.h> (a macro for _Bool). Mirror that.
#  ifndef bool
#    define bool _Bool
#  endif
#  ifndef true
#    define true 1
#  endif
#  ifndef false
#    define false 0
#  endif
#endif

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#ifndef INT8_C
#define INT8_C(value) value
#define UINT8_C(value) value##u
#define INT16_C(value) value
#define UINT16_C(value) value##u
#define INT32_C(value) value
#define UINT32_C(value) value##u
#define INT64_C(value) value##ll
#define UINT64_C(value) value##ull
#define INTMAX_C(value) INT64_C(value)
#define UINTMAX_C(value) UINT64_C(value)
#endif

#ifndef UINT16_MAX
#define UINT16_MAX ((uint16_t)0xFFFFu)
#endif
#ifndef INT32_MAX
#define INT32_MAX ((int32_t)0x7FFFFFFF)
#endif
#ifndef UINT32_MAX
#define UINT32_MAX ((uint32_t)0xFFFFFFFFu)
#endif
#ifndef INT64_MAX
#define INT64_MAX ((int64_t)0x7FFFFFFFFFFFFFFFll)
#endif
#ifndef UINT64_MAX
#define UINT64_MAX ((uint64_t)0xFFFFFFFFFFFFFFFFull)
#endif
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

#define array_size(a) (sizeof(a) / sizeof((a)[0]))

#ifdef __cplusplus
}
#endif
