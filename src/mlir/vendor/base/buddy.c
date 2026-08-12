#include <base/buddy.h>
#include <platform/platform.h>
#include <base/types.h>
#include <base/assert.h>
#include <base/io.h>
#include <base/numconv.h>
#include <base/mem.h>
#include <base/exit.h>

#define MIN_PAGE_SIZE 4096UL
#define MAX_ORDER 20 // 2^20 * 4KB = 4GB

/*
 * Header for each memory block (free or allocated).
 * This header is stored "inline" at the beginning of each block of memory.
 */
struct buddy_block {
    // The order of the block. Positive if free, negative if allocated.
    int order;
    struct buddy_block *prev;
    struct buddy_block *next;
};

struct list_head {
    struct buddy_block *first;
};

// free_lists[i] contains a doubly-linked list of free blocks of order i.
static struct list_head free_lists[MAX_ORDER + 1];
static void *heap_base;

static void list_add(struct list_head *lh, struct buddy_block *p) {
    p->next = lh->first;
    p->prev = NULL;
    if (lh->first) {
        lh->first->prev = p;
    }
    lh->first = p;
}

static void list_remove(struct list_head *lh, struct buddy_block *p) {
    if (p->prev) {
        p->prev->next = p->next;
    } else {
        lh->first = p->next;
    }
    if (p->next) {
        p->next->prev = p->prev;
    }
}

static void add_memory(void *mem, size_t bytes) {
    uintptr_t start = (uintptr_t)mem;
    uintptr_t end = start + bytes;

    // Align start address up to a MIN_PAGE_SIZE boundary
    uintptr_t mis = start % MIN_PAGE_SIZE;
    if (mis) {
        start += MIN_PAGE_SIZE - mis;
    }

    // Carve the memory region into the largest possible power-of-two blocks
    while (start + MIN_PAGE_SIZE <= end) {
        int order = 0;
        size_t block_size = MIN_PAGE_SIZE;
        while ((block_size << 1) <= (end - start) && (start % (block_size << 1) == 0) && order < MAX_ORDER) {
            block_size <<= 1;
            order++;
        }

        struct buddy_block *p = (struct buddy_block *)start;
        p->order = order;
        list_add(&free_lists[order], p);

        start += block_size;
    }
}

void buddy_init(void) {
    heap_base = platform_heap_base();
    for (int o = 0; o <= MAX_ORDER; o++) {
        free_lists[o].first = NULL;
    }
    size_t initial_size = platform_heap_size();
    if (initial_size > 0) {
        add_memory(heap_base, initial_size);
    }
}

// Helper to append a string to a buffer (simple strcat that knows buffer size)
static void str_append_safe(char *dest, const char *src, size_t dest_size) {
    size_t dest_len = base_strlen(dest);
    size_t src_len = base_strlen(src);
    size_t space_left = dest_size - dest_len - 1; // -1 for null terminator

    if (src_len > space_left) {
        src_len = space_left;
    }

    base_memcpy(dest + dest_len, src, src_len);
    dest[dest_len + src_len] = '\0';
}

void buddy_print_stats() {
    int fd = PLATFORM_STDOUT_FD;
    writeln(fd, "");
    writeln(fd, "=== Buddy Allocator Statistics ===");
    writeln(fd, "");

    // Calculate total free and allocated bytes per order
    size_t free_counts[MAX_ORDER + 1];
    size_t allocated_counts[MAX_ORDER + 1];
    size_t total_free_bytes = 0;
    size_t total_allocated_bytes = 0;

    for (int o = 0; o <= MAX_ORDER; o++) {
        free_counts[o] = 0;
        allocated_counts[o] = 0;
    }

    // Count free blocks
    for (int o = 0; o <= MAX_ORDER; o++) {
        struct buddy_block *block = free_lists[o].first;
        while (block) {
            size_t block_size = MIN_PAGE_SIZE << o;
            total_free_bytes += block_size;
            free_counts[o]++;
            block = block->next;
        }
    }

    // Count allocated blocks by scanning all committed memory
    uintptr_t heap_start = (uintptr_t)heap_base;
    uintptr_t heap_end = heap_start + platform_heap_size();

    // Scan through memory looking for allocated blocks
    // This is a heuristic scan - we check each MIN_PAGE_SIZE aligned address
    for (uintptr_t addr = heap_start; addr < heap_end; addr += MIN_PAGE_SIZE) {
        struct buddy_block *block = (struct buddy_block *)addr;
        // Check if this looks like an allocated block (negative order)
        if (block->order < 0) {
            int order = -block->order - 1;
            if (order >= 0 && order <= MAX_ORDER) {
                size_t block_size = MIN_PAGE_SIZE << order;
                // Verify the block is within bounds and properly aligned
                if ((addr % block_size) == 0 && addr + block_size <= heap_end) {
                    total_allocated_bytes += block_size;
                    allocated_counts[order]++;
                    // Skip past this allocated block
                    addr += block_size - MIN_PAGE_SIZE; // -MIN_PAGE_SIZE because loop adds it
                }
            }
        }
    }

    size_t committed_bytes = platform_heap_size();

    // Helper to print size_t value with label
    #define PRINT_SIZE(label, value) do { \
        char buf[32]; \
        size_t len = uint64_to_str((value), buf); \
        buf[len] = '\0'; \
        write_all(fd, (ciovec_t[]){(ciovec_t){(label), base_strlen(label)}}, 1); \
        write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1); \
        write_all(fd, (ciovec_t[]){(ciovec_t){buf, len}}, 1); \
        write_all(fd, (ciovec_t[]){(ciovec_t){"\n", 1}}, 1); \
    } while(0)

    // Helper to print floating point MiB with 2 decimal places
    #define PRINT_MIB(label, bytes) do { \
        double mib = (double)(bytes) / (1024.0 * 1024.0); \
        char buf[32]; \
        size_t len = double_to_str(mib, buf, 2); \
        buf[len] = '\0'; \
        write_all(fd, (ciovec_t[]){(ciovec_t){(label), base_strlen(label)}}, 1); \
        write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1); \
        write_all(fd, (ciovec_t[]){(ciovec_t){buf, len}}, 1); \
        write_all(fd, (ciovec_t[]){(ciovec_t){"\n", 1}}, 1); \
    } while(0)

    // Print memory overview
    writeln(fd, "Memory Overview:");
    PRINT_SIZE("  Committed (bytes): ", committed_bytes);
    PRINT_MIB("  Committed (MiB):   ", committed_bytes);
    PRINT_SIZE("  Free (bytes):      ", total_free_bytes);
    PRINT_MIB("  Free (MiB):        ", total_free_bytes);
    PRINT_SIZE("  Allocated (bytes): ", total_allocated_bytes);
    PRINT_MIB("  Allocated (MiB):   ", total_allocated_bytes);
    if (committed_bytes > 0) {
        double utilization = ((double)total_allocated_bytes * 100.0) / (double)committed_bytes;
        char util_buf[32];
        size_t util_len = double_to_str(utilization, util_buf, 2);
        util_buf[util_len] = '\0';
        write_all(fd, (ciovec_t[]){(ciovec_t){"  Utilization (%):   ", 21}}, 1);
        write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1);
        write_all(fd, (ciovec_t[]){(ciovec_t){util_buf, util_len}}, 1);
        write_all(fd, (ciovec_t[]){(ciovec_t){"\n", 1}}, 1);
    }
    writeln(fd, "");

    // Print per-order breakdown (all orders)
    writeln(fd, "Per-Order Breakdown (all orders 0-20):");
    writeln(fd, "  'Free' = blocks in free list, 'Allocated' = blocks given to user");
    writeln(fd, "Order  BlockSize        Free  Allocated   FreeMiB  AllocMiB");
    writeln(fd, "-----  --------------  -----  ---------  --------  --------");

    int total_orders_with_free = 0;
    int total_orders_with_allocated = 0;
    for (int o = 0; o <= MAX_ORDER; o++) {
        if (free_counts[o] > 0) total_orders_with_free++;
        if (allocated_counts[o] > 0) total_orders_with_allocated++;
    }

    for (int o = 0; o <= MAX_ORDER; o++) {
            size_t block_size = MIN_PAGE_SIZE << o;
            size_t free_bytes = free_counts[o] * block_size;
            size_t alloc_bytes = allocated_counts[o] * block_size;

            // Print order (width 5, right-aligned)
            char order_str[32];
            size_t order_len = int_to_str(o, order_str);
            order_str[order_len] = '\0';
            for (size_t i = order_len; i < 5; i++) {
                write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1);
            }
            write_all(fd, (ciovec_t[]){(ciovec_t){order_str, order_len}}, 1);
            write_all(fd, (ciovec_t[]){(ciovec_t){"  ", 2}}, 1);

            // Print block size (width 14, right-aligned)
            char size_str[32];
            size_t size_len = uint64_to_str(block_size, size_str);
            size_str[size_len] = '\0';
            for (size_t i = size_len; i < 14; i++) {
                write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1);
            }
            write_all(fd, (ciovec_t[]){(ciovec_t){size_str, size_len}}, 1);
            write_all(fd, (ciovec_t[]){(ciovec_t){"  ", 2}}, 1);

            // Print free count (width 5, right-aligned)
            char free_count_str[32];
            size_t free_count_len = int_to_str((int)free_counts[o], free_count_str);
            free_count_str[free_count_len] = '\0';
            for (size_t i = free_count_len; i < 5; i++) {
                write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1);
            }
            write_all(fd, (ciovec_t[]){(ciovec_t){free_count_str, free_count_len}}, 1);
            write_all(fd, (ciovec_t[]){(ciovec_t){"  ", 2}}, 1);

            // Print allocated count (width 9, right-aligned)
            char alloc_count_str[32];
            size_t alloc_count_len = int_to_str((int)allocated_counts[o], alloc_count_str);
            alloc_count_str[alloc_count_len] = '\0';
            for (size_t i = alloc_count_len; i < 9; i++) {
                write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1);
            }
            write_all(fd, (ciovec_t[]){(ciovec_t){alloc_count_str, alloc_count_len}}, 1);
            write_all(fd, (ciovec_t[]){(ciovec_t){"  ", 2}}, 1);

            // Print free MiB (width 9, right-aligned, 2 decimal places)
            char free_mib_str[32];
            double free_mib = (double)free_bytes / (1024.0 * 1024.0);
            size_t free_mib_len = double_to_str(free_mib, free_mib_str, 2);
            free_mib_str[free_mib_len] = '\0';
            for (size_t i = free_mib_len; i < 9; i++) {
                write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1);
            }
            write_all(fd, (ciovec_t[]){(ciovec_t){free_mib_str, free_mib_len}}, 1);
            write_all(fd, (ciovec_t[]){(ciovec_t){"  ", 2}}, 1);

            // Print alloc MiB (width 9, right-aligned, 2 decimal places)
            char alloc_mib_str[32];
            double alloc_mib = (double)alloc_bytes / (1024.0 * 1024.0);
            size_t alloc_mib_len = double_to_str(alloc_mib, alloc_mib_str, 2);
            alloc_mib_str[alloc_mib_len] = '\0';
            for (size_t i = alloc_mib_len; i < 9; i++) {
                write_all(fd, (ciovec_t[]){(ciovec_t){" ", 1}}, 1);
            }
            write_all(fd, (ciovec_t[]){(ciovec_t){alloc_mib_str, alloc_mib_len}}, 1);

            write_all(fd, (ciovec_t[]){(ciovec_t){"\n", 1}}, 1);
    }

    // Print summary
    writeln(fd, "");
    char summary1[128], summary2[128];
    char free_orders_str[16], alloc_orders_str[16];
    size_t free_orders_len = int_to_str(total_orders_with_free, free_orders_str);
    free_orders_str[free_orders_len] = '\0';
    size_t alloc_orders_len = int_to_str(total_orders_with_allocated, alloc_orders_str);
    alloc_orders_str[alloc_orders_len] = '\0';

    base_strcpy(summary1, "Summary: ");
    str_append_safe(summary1, free_orders_str, sizeof(summary1));
    str_append_safe(summary1, " orders have free blocks (buddy has memory at these sizes)", sizeof(summary1));
    writeln(fd, summary1);

    base_strcpy(summary2, "         ");
    str_append_safe(summary2, alloc_orders_str, sizeof(summary2));
    str_append_safe(summary2, " orders have allocated blocks (user is using these sizes)", sizeof(summary2));
    writeln(fd, summary2);
    writeln(fd, "");

    // Print alignment diagnostics for large orders
    writeln(fd, "Alignment Diagnostics:");
    writeln(fd, "  (Buddy allocator requires blocks to be aligned to their size)");
    uintptr_t current_top = (uintptr_t)heap_base + platform_heap_size();

    char top_str[32];
    size_t top_len = uint64_to_str(current_top, top_str);
    top_str[top_len] = '\0';
    write_all(fd, (ciovec_t[]){(ciovec_t){"  Current heap top:    ", 23}}, 1);
    write_all(fd, (ciovec_t[]){(ciovec_t){top_str, top_len}}, 1);
    write_all(fd, (ciovec_t[]){(ciovec_t){"\n", 1}}, 1);

    writeln(fd, "");
    writeln(fd, "  Alignment status for large orders:");
    for (int o = 9; o <= 14; o++) {
        size_t alignment = MIN_PAGE_SIZE << o;
        uintptr_t misalignment = current_top % alignment;
        size_t size_mib = alignment >> 20;

        char line[128];
        char order_str[32], size_str[32], status[64];
        size_t order_str_len = int_to_str(o, order_str);
        order_str[order_str_len] = '\0';
        size_t size_str_len = int_to_str((int)size_mib, size_str);
        size_str[size_str_len] = '\0';

        if (misalignment == 0) {
            base_strcpy(status, "ALIGNED");
        } else {
            char mis_str[32];
            size_t mis_str_len = uint64_to_str(misalignment, mis_str);
            mis_str[mis_str_len] = '\0';
            base_strcpy(status, "MISALIGNED by ");
            str_append_safe(status, mis_str, sizeof(status));
            str_append_safe(status, " bytes", sizeof(status));
        }

        base_strcpy(line, "    Order ");
        str_append_safe(line, order_str, sizeof(line));
        str_append_safe(line, " (", sizeof(line));
        str_append_safe(line, size_str, sizeof(line));
        str_append_safe(line, " MiB): ", sizeof(line));
        str_append_safe(line, status, sizeof(line));

        writeln(fd, line);
    }
    writeln(fd, "");

    // Print warnings
    writeln(fd, "DIAGNOSIS:");
    int has_warnings = 0;
    for (int o = 9; o <= MAX_ORDER; o++) {
        size_t alignment = MIN_PAGE_SIZE << o;
        uintptr_t misalignment = current_top % alignment;
        if (misalignment != 0 && free_counts[o] == 0) {
            if (!has_warnings) {
                writeln(fd, "  *** ALIGNMENT BUG DETECTED ***");
                writeln(fd, "  Cannot allocate large blocks because heap top is misaligned.");
                writeln(fd, "  Even though memory is available, add_memory() cannot create");
                writeln(fd, "  properly aligned blocks at these orders:");
                writeln(fd, "");
                has_warnings = 1;
            }
            char msg[128];
            char order_str[32], size_str[32], padding_str[32];
            size_t order_str_len = int_to_str(o, order_str);
            order_str[order_str_len] = '\0';
            size_t size_str_len = int_to_str((int)(alignment >> 20), size_str);
            size_str[size_str_len] = '\0';

            // Calculate padding needed to align
            size_t padding_needed = (alignment - misalignment) % alignment;
            size_t padding_len = uint64_to_str(padding_needed, padding_str);
            padding_str[padding_len] = '\0';

            base_strcpy(msg, "    Order ");
            str_append_safe(msg, order_str, sizeof(msg));
            str_append_safe(msg, " (", sizeof(msg));
            str_append_safe(msg, size_str, sizeof(msg));
            str_append_safe(msg, " MiB): needs ", sizeof(msg));
            str_append_safe(msg, padding_str, sizeof(msg));
            str_append_safe(msg, " bytes padding", sizeof(msg));
            writeln(fd, msg);
        }
    }

    if (!has_warnings) {
        writeln(fd, "  No alignment issues detected.");
    }
    writeln(fd, "");
    writeln(fd, "=== End Statistics ===");
    writeln(fd, "");

    #undef PRINT_SIZE
    #undef PRINT_MIB
}

static void *buddy_alloc_order(int order) {
    assert(order >= 0 && order <= MAX_ORDER);

    // Find the smallest available block that is large enough
    int current_order;
    for (current_order = order; current_order <= MAX_ORDER; current_order++) {
        if (free_lists[current_order].first) {
            break; // Found a suitable block
        }
    }

    // If no block is available, grow the heap
    if (current_order > MAX_ORDER) {
        size_t required_size = MIN_PAGE_SIZE << order;
        size_t alignment = MIN_PAGE_SIZE << order;

        // Calculate current heap top and alignment padding needed
        uintptr_t current_top = (uintptr_t)heap_base + platform_heap_size();
        uintptr_t aligned_top = (current_top + alignment - 1) / alignment * alignment;
        size_t padding = aligned_top - current_top;

        // Total bytes to grow: padding + required_size
        size_t total_grow = padding + required_size;

        // Round up to PLATFORM_WASM_PAGE_SIZE boundary
        size_t grow_by = ((total_grow + PLATFORM_WASM_PAGE_SIZE - 1) / PLATFORM_WASM_PAGE_SIZE) * PLATFORM_WASM_PAGE_SIZE;

        void *new_mem = platform_heap_grow(grow_by);
        if (!new_mem) {
            buddy_print_stats();
            writeln_int(PLATFORM_STDERR_FD, "order =", order);
            writeln_int(PLATFORM_STDERR_FD, "required_size =", (int)required_size);
            writeln_int(PLATFORM_STDERR_FD, "grow_by =", (int)grow_by);
            writeln_int(PLATFORM_STDERR_FD, "padding =", (int)padding);
            writeln_int(PLATFORM_STDERR_FD, "current_top % alignment =", (int)(current_top % alignment));
            FATAL_ERROR("platform_heap_grow(grow_by) failed");
        }
        add_memory(new_mem, grow_by);
        return buddy_alloc_order(order); // Retry allocation
    }

    // We have a block of order 'current_order'. Remove it from its free list.
    struct buddy_block *p = free_lists[current_order].first;
    list_remove(&free_lists[current_order], p);

    // Split the block until it's the desired size
    while (current_order > order) {
        current_order--;
        size_t half_size = MIN_PAGE_SIZE << current_order;
        struct buddy_block *buddy = (struct buddy_block *)((uintptr_t)p + half_size);
        buddy->order = current_order;
        list_add(&free_lists[current_order], buddy);
    }

    // Mark the block as allocated by making its order negative
    p->order = -(order + 1);

    // Return the pointer to the memory *after* our inline header
    return (void *)(p + 1);
}

void *buddy_alloc(size_t size, size_t *actual_size) {
    assert(size > 0);

    // Add space for our header to the requested size
    size_t size_with_header = size + sizeof(struct buddy_block);

    // Calculate the order required for the allocation
    int order = 0;
    size_t block_size = MIN_PAGE_SIZE;
    while (block_size < size_with_header) {
        block_size <<= 1;
        order++;
        if (order > MAX_ORDER) {
            return NULL; // Request is too large
        }
    }

    // If caller wants to know the actual size, calculate it
    if (actual_size) {
        *actual_size = block_size - sizeof(struct buddy_block);
    }

    return buddy_alloc_order(order);
}

void buddy_free(void *ptr) {
    assert(ptr);

    // Get the block header from the user-provided pointer
    struct buddy_block *p = ((struct buddy_block *)ptr) - 1;

    // Retrieve the original order and mark the block as free
    int order = -p->order - 1;
    if (order < 0 || order > MAX_ORDER) {
        return; // Invalid pointer or heap corruption
    }

    uintptr_t heap_end = (uintptr_t)heap_base + platform_heap_size();

    // Coalesce with buddy if possible
    while (order < MAX_ORDER) {
        size_t block_size = MIN_PAGE_SIZE << order;
        uintptr_t p_addr = (uintptr_t)p;
        uintptr_t buddy_addr = p_addr ^ block_size;

        // Ensure the buddy is within the heap bounds before accessing it
        if (buddy_addr < (uintptr_t)heap_base || buddy_addr >= heap_end) {
            break;
        }

        struct buddy_block *buddy = (struct buddy_block *)buddy_addr;

        if (buddy->order != order) {
            break; // Buddy is not free or not the same size
        }

        // Buddy is free and of the same order, so merge them.
        list_remove(&free_lists[order], buddy);

        // The merged block starts at the lower of the two addresses
        if (buddy_addr < p_addr) {
            p = buddy;
        }

        order++;
    }

    p->order = order;
    list_add(&free_lists[order], p);
}
