#pragma once

#include <base/io.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FATAL_ERROR(x) do { PRINT_ERR(x); base_abort(); } while (0)

// Process exit for base/
void base_exit(int status);
void base_abort(void);
#ifdef __cplusplus
}
#endif
