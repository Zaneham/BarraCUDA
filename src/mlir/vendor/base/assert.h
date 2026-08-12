#pragma once

// Base assertion facility - self-contained, no dependencies on stdio/stdlib
// Used by base/ tests and re-exported by stdlib/assert.h

#ifdef __cplusplus
extern "C" {
#endif

void __assert_fail(const char *assertion, const char *file, unsigned int line, const char *function);

#undef assert
#ifdef NDEBUG
#define assert(condition) ((void)0)
#else
#define assert(condition) ((condition) ? (void)0 : __assert_fail("Assertion failed '" #condition "'", __FILE__, __LINE__, __func__))
#endif
#ifdef __cplusplus
}
#endif
