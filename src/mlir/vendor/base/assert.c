#include <base/assert.h>
#include <base/io.h>
#include <base/mem.h>
#include <platform/platform.h>
#include <base/numconv.h>

/* After proper strings and formatting, this should just be:

void __assert_fail(const char *assertion, const char *file, unsigned int line, const char *function) {
    printf("Assertion failed: (%s) at '%s:%u' in function '%s'\n", assertion, file, line, function);
    exit(1);
}

However, we do not want to depend on arenas working, we only want to use the
lower-level API, no other base dependencies, so that asser() can be used
anywhere in base.
*/

void __assert_fail(const char *assertion, const char *file, unsigned int line, const char *function) {
    // Simple assertion failure handler using only base/ dependencies
    writeln_loc(PLATFORM_STDERR_FD, assertion, file, line, function);
    platform_exit(1);
}
