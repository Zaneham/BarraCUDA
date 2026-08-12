#include <base/arena.h>
#include <base/buddy.h>
#include <base/types.h>
#include <base/assert.h>
#include <base/exit.h>
#include <base/io.h>

// All allocations will be aligned to this boundary (must be a power of two).
#define ARENA_ALIGNMENT 16
// New chunks will be at least this large.
#define MIN_CHUNK_SIZE 4096

// Represents a single chunk of memory obtained from the buddy allocator.
struct arena_chunk {
    struct arena_chunk *next;
    // Total size of the block returned by buddy_alloc for this chunk.
    size_t size;
    // The data area for this chunk begins immediately after this struct.
};

// The main arena structure. Its definition is hidden from the public API.
struct arena_s {
    struct arena_chunk *first_chunk;
    struct arena_chunk *current_chunk;
    char *current_ptr;
    size_t remaining_in_chunk;
    size_t default_chunk_size;
};

// Aligns a value up to the nearest multiple of ARENA_ALIGNMENT.
static inline uintptr_t align_up(uintptr_t val) {
    return (val + ARENA_ALIGNMENT - 1) & ~(uintptr_t)(ARENA_ALIGNMENT - 1);
}

Arena *arena_create(size_t initial_size) {
    // Allocate the arena controller struct itself.
    Arena *arena = buddy_alloc(sizeof(Arena), NULL);
    if (!arena) {
        FATAL_ERROR("buddy_alloc failed for Arena");
    }

    if (initial_size < MIN_CHUNK_SIZE) {
        initial_size = MIN_CHUNK_SIZE;
    }
    arena->default_chunk_size = initial_size;
    arena->first_chunk = NULL;

    // Allocate the first chunk.
    // Request enough space for the chunk header, the caller's requested size,
    // and any padding that might be needed to align the data pointer.
    size_t requested_size = sizeof(struct arena_chunk) + initial_size + ARENA_ALIGNMENT;
    size_t actual_size;
    struct arena_chunk *first = buddy_alloc(requested_size, &actual_size);
    if (!first) {
        //buddy_free(arena);
        FATAL_ERROR("buddy_alloc failed for size");
    }
    first->next = NULL;
    first->size = actual_size;

    // Initialize arena state to point to the start of the first chunk.
    arena->first_chunk = first;
    arena->current_chunk = first;

    uintptr_t data_start = align_up((uintptr_t)(first + 1));
    uintptr_t chunk_end = (uintptr_t)first + actual_size;

    arena->current_ptr = (char *)data_start;
    arena->remaining_in_chunk = (data_start < chunk_end) ? (chunk_end - data_start) : 0;

    return arena;
}

void *arena_alloc(Arena *arena, size_t size) {
    assert(arena);
    assert(size > 0);

    size_t aligned_size = (size + ARENA_ALIGNMENT - 1) & ~(size_t)(ARENA_ALIGNMENT - 1);

try_alloc:
    // If the current chunk has enough space, perform a simple bump allocation.
    if (aligned_size <= arena->remaining_in_chunk) {
        void *ptr = arena->current_ptr;
        arena->current_ptr += aligned_size;
        arena->remaining_in_chunk -= aligned_size;
        return ptr;
    }

    // Not enough space. If a next chunk already exists (from previous use), move to it.
    if (arena->current_chunk && arena->current_chunk->next) {
        arena->current_chunk = arena->current_chunk->next;

        struct arena_chunk* chunk = arena->current_chunk;
        uintptr_t data_start = align_up((uintptr_t)(chunk + 1));
        uintptr_t chunk_end = (uintptr_t)chunk + chunk->size;

        arena->current_ptr = (char *)data_start;
        arena->remaining_in_chunk = (data_start < chunk_end) ? (chunk_end - data_start) : 0;

        goto try_alloc; // Retry allocation in the next chunk.
    }

    // No more reusable chunks are available, so allocate a new one.
    size_t new_chunk_data_size = arena->default_chunk_size;
    if (aligned_size > new_chunk_data_size) {
        new_chunk_data_size = aligned_size; // Ensure the new chunk is large enough.
    }

    // Request: header + data + alignment padding
    // The usable data area starts at align_up(chunk + 1), so we may lose up to ARENA_ALIGNMENT bytes
    size_t requested_size = sizeof(struct arena_chunk) + new_chunk_data_size + ARENA_ALIGNMENT;

    // Allocate and get the actual size buddy provides (rounded up to power-of-2)
    size_t actual_size;
    struct arena_chunk *new_chunk = buddy_alloc(requested_size, &actual_size);
    if (!new_chunk) {
        FATAL_ERROR("buddy_alloc failed");
    }

    new_chunk->next = NULL;
    // Store the actual size we got from buddy, not what we requested
    new_chunk->size = actual_size;

    // Link the new chunk to the end of the list.
    if (arena->current_chunk) {
        arena->current_chunk->next = new_chunk;
    } else {
        arena->first_chunk = new_chunk;
    }
    arena->current_chunk = new_chunk;

    // Set the allocation pointer to the start of the new chunk.
    // data_start is after the arena_chunk header, aligned up
    uintptr_t data_start = align_up((uintptr_t)(new_chunk + 1));
    // chunk_end is based on the actual size buddy gave us
    uintptr_t chunk_end = (uintptr_t)new_chunk + actual_size;
    arena->current_ptr = (char *)data_start;
    arena->remaining_in_chunk = (data_start < chunk_end) ? (chunk_end - data_start) : 0;

    // Retry the allocation now that we have a new, sufficiently large chunk.
    goto try_alloc;
}

void arena_destroy(Arena *arena) {
    assert(arena);
    struct arena_chunk *current = arena->first_chunk;
    while (current) {
        struct arena_chunk *next = current->next;
        buddy_free(current);
        current = next;
    }
    buddy_free(arena);
}

arena_pos_t arena_get_pos(Arena *arena) {
    assert(arena);
    return (arena_pos_t){
        .chunk = arena->current_chunk,
        .ptr = arena->current_ptr
    };
}

void arena_reset(Arena *arena, arena_pos_t pos) {
    assert(arena);
    assert(pos.chunk);
    assert(pos.ptr);

    // Restore the state from the saved position
    arena->current_chunk = pos.chunk;
    arena->current_ptr = pos.ptr;

    // Recalculate the remaining size in the restored chunk
    uintptr_t chunk_end = (uintptr_t)pos.chunk + pos.chunk->size;
    uintptr_t current_pos = (uintptr_t)pos.ptr;

    arena->remaining_in_chunk = (current_pos < chunk_end) ? (chunk_end - current_pos) : 0;
}

size_t arena_chunk_count(Arena *arena) {
    assert(arena);

    size_t count = 0;
    struct arena_chunk *chunk = arena->first_chunk;
    while (chunk) {
        count++;
        chunk = chunk->next;
    }
    return count;
}

size_t arena_current_chunk_index(Arena *arena) {
    assert(arena);
    assert(arena->current_chunk);

    size_t index = 0;
    struct arena_chunk *chunk = arena->first_chunk;
    while (chunk && chunk != arena->current_chunk) {
        index++;
        chunk = chunk->next;
    }
    return index;
}
