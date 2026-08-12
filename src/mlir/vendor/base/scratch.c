#include <base/scratch.h>
#include <platform/platform.h>
#include <base/exit.h>

Scratch scratch_begin_from_arena(Arena *arena) {
    return (Scratch){.arena=arena, .saved_pos=arena_get_pos(arena)};
}

Scratch scratch_begin() {
    return scratch_begin_avoid_conflict(NULL);
}

Arena* scratch_arenas[2] = {NULL, NULL};

static void init_scratch_arenas(void) {
    scratch_arenas[0] = arena_create(1024);
    scratch_arenas[1] = arena_create(1024);
}

Scratch scratch_begin_avoid_conflict(Arena *conflict) {
    if (scratch_arenas[0] == NULL) {
        init_scratch_arenas();
    }
    for (int i = 0; i < 2; i++) {
        if (scratch_arenas[i] != conflict) {
            return scratch_begin_from_arena(scratch_arenas[i]);
        }
    }
    FATAL_ERROR("Cannot find conflict-free arena.");
    return (Scratch){NULL,{0}};
}

void scratch_end(Scratch scratch) {
    arena_reset(scratch.arena, scratch.saved_pos);
}
