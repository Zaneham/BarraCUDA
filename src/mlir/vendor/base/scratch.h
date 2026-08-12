#pragma once

#include <base/types.h>
#include <base/arena.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Arena *arena;
    arena_pos_t saved_pos;
} Scratch;

// Internally there are 2 scratch arenas. You can use a scratch in the
// region marked by scratch_begin*() and scratch_end().

// Use if there are no other arenas that you allocate from in the scratch region.
// Always returns the first scratch arena.
Scratch scratch_begin();

// Use if there is another arena that you allocate from in the scratch region.
// Pass the other arena as an argument `conflict`.
// Returns a scratch arena that does not conflict with `conflict`.
/*
The example below shows the case when you need to call
scratch_begin_avoid_conflict(). The inner_fn() is returning result to the caller
using the `outer_arena`, which was allocated using scratch_begin() by the caller
in `outer_fn`. The inner arena must be allocated using
scratch_begin_avoid_conflict() otherwise both the outer_temp and outer_temp2
will point to the same memory and we get a conflict (the result of outer_temp
will be lost).

    char* inner_fn(Arena *outer_arena) {
        Scratch inner = scratch_begin_avoid_conflict(outer_arena);
        char *result = arena_alloc(outer_arena, 50);
        ...
        scratch_end(inner);
        return result;
    }

    void outer_fn() {
        Scratch outer = scratch_begin();
        char *outer_temp = inner_fn(outer.arena);
        char *outer_temp2 = arena_alloc(outer.arena, 50);
        ...
        scratch_end(outer);
    }
*/
Scratch scratch_begin_avoid_conflict(Arena *conflict);

// Use if there is another arena available in the region but you do not push
// into it in the scratch region. The scratch will be created in `arena`.
Scratch scratch_begin_from_arena(Arena *arena);

// Marks the end of the scratch region. Resets the arena that was used to
// create the scratch to the position before the scratch.
void scratch_end(Scratch scratch);
#ifdef __cplusplus
}
#endif
