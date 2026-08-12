#include <base/strbuf.h>
#include <base/mem.h>
#include <base/assert.h>

// Initial capacity granted on the first append when the buffer was created
// with capacity zero. Large enough that short strings never grow, small
// enough that we don't waste arena space on buffers that stay tiny.
#define STRBUF_MIN_CAP 64

strbuf strbuf_make(void) {
    return (strbuf){NULL, 0, 0};
}

strbuf strbuf_make_cap(Arena *arena, uint64_t cap) {
    strbuf b = (strbuf){NULL, 0, 0};
    if (cap > 0) {
        b.str = arena_new_array(arena, char, cap);
        b.cap = cap;
    }
    return b;
}

// Grow the buffer so that at least `min_cap` total bytes are available.
// Doubling strategy: `new_cap = max(cap * 2, min_cap, STRBUF_MIN_CAP)`.
static void strbuf_grow(Arena *arena, strbuf *b, uint64_t min_cap) {
    if (min_cap <= b->cap) return;
    uint64_t new_cap = b->cap ? b->cap : STRBUF_MIN_CAP;
    while (new_cap < min_cap) {
        new_cap *= 2;
    }
    char *new_buf = arena_new_array(arena, char, new_cap);
    if (b->size > 0) {
        base_memcpy(new_buf, b->str, b->size);
    }
    b->str = new_buf;
    b->cap = new_cap;
}

void strbuf_reserve(Arena *arena, strbuf *b, uint64_t min_cap) {
    assert(arena);
    assert(b);
    strbuf_grow(arena, b, min_cap);
}

void strbuf_append_bytes(Arena *arena, strbuf *b, const void *p, uint64_t n) {
    assert(arena);
    assert(b);
    if (n == 0) return;
    strbuf_grow(arena, b, b->size + n);
    base_memcpy(b->str + b->size, p, n);
    b->size += n;
}

void strbuf_append(Arena *arena, strbuf *b, string s) {
    strbuf_append_bytes(arena, b, s.str, s.size);
}

void strbuf_append_char(Arena *arena, strbuf *b, char c) {
    assert(arena);
    assert(b);
    strbuf_grow(arena, b, b->size + 1);
    b->str[b->size++] = c;
}

void strbuf_append_cstr(Arena *arena, strbuf *b, const char *s) {
    assert(s);
    strbuf_append_bytes(arena, b, s, base_strlen(s));
}

string strbuf_to_string(strbuf b) {
    return (string){b.str, b.size};
}
