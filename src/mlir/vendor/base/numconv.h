#pragma once

#include <base/types.h>
#include <base/stdarg.h>

// Number to string conversion functions for base/
// Self-contained implementations with no external dependencies

// Convert unsigned 64-bit integer to string
// Returns length of string written (not including null terminator)

#ifdef __cplusplus
extern "C" {
#endif

size_t uint64_to_str(uint64_t val, char* buf);

// Convert signed 64-bit integer to string
// Returns length of string written (not including null terminator)
size_t int64_to_str(int64_t val, char* buf);

// Convert int to string (calls int64_to_str)
// Returns length of string written (not including null terminator)
size_t int_to_str(int val, char* buf);

// Convert double to string with specified precision
// Returns length of string written (not including null terminator)
// precision: number of decimal places (-1 for default of 6)
size_t double_to_str(double val, char* buf, int precision);
size_t double_to_str_e(double val, char* buf, int precision);

// Convert unsigned 64-bit integer to hexadecimal string
// Returns length of string written (not including null terminator)
// uppercase: 0 for lowercase (a-f), non-zero for uppercase (A-F)
size_t uint64_to_hex_str(uint64_t val, char* buf, int uppercase);

// Simple vsnprintf/snprintf implementations for base/ (only for nostdlib builds)
// Supports: %d, %i, %u, %ld, %li, %lu, %lld, %lli, %llu, %zu, %x, %X, %lx, %lX, %llx, %llX, %p, %c, %s, %f, %g, %.Nf, %%
// Returns number of characters written (not including null terminator)
int base_vsnprintf(char *str, size_t size, const char *format, va_list args);

// Simple snprintf implementation for base/
// Supports: %d, %i, %u, %ld, %li, %lu, %lld, %lli, %llu, %zu, %x, %X, %lx, %lX, %llx, %llX, %p, %c, %s, %f, %g, %.Nf, %%
// Returns number of characters written (not including null terminator)
int base_snprintf(char *str, size_t size, const char *format, ...);
#ifdef __cplusplus
}
#endif
