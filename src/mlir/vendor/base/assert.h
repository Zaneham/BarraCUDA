#pragma once

// Base assertion facility - self-contained, no dependencies on stdio/stdlib
// Used by base/ tests and re-exported by stdlib/assert.h
//
// Named corec_assert_fail rather than __assert_fail, which is what a
// freestanding build wants but which glibc also declares and defines. Hosted,
// the declaration is redundant and the definition quietly overrides libc's for
// the whole program.

#ifdef __cplusplus
extern "C" {
#endif

void corec_assert_fail(const char *assertion, const char *file, unsigned int line, const char *function);

#undef assert
#ifdef NDEBUG
#define assert(condition) ((void)0)
#else
#define assert(condition) ((condition) ? (void)0 : corec_assert_fail("Assertion failed '" #condition "'", __FILE__, __LINE__, __func__))
#endif
#ifdef __cplusplus
}
#endif
