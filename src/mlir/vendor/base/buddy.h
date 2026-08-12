#pragma once

#include <base/types.h>

#ifdef __cplusplus
extern "C" {
#endif

void buddy_init(void);

// Allocate memory from the buddy allocator.
// Returns NULL on allocation failure.
// If actual_size is not NULL, stores the actual usable size allocated (which may be
// larger than requested due to power-of-2 rounding).
void *buddy_alloc(size_t size, size_t *actual_size);

void buddy_free(void *ptr);

// Print detailed statistics about the buddy allocator state
void buddy_print_stats();
#ifdef __cplusplus
}
#endif
