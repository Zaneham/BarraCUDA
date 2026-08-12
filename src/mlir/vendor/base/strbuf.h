#pragma once

#include <base/types.h>
#include <base/arena.h>
#include <base/string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Growable byte buffer with explicit capacity.
//
// `strbuf` exists for the "build up a string by appending to it" workload,
// where the read-only `string` (pointer + length view) type is the wrong
// abstraction — there is no capacity to track and every `str_concat` would
// have to copy the accumulator.
//
// The arena that owns the backing memory is NOT stored inside `strbuf`;
// callers pass it to every mutating call. This lets a caller:
//   - build a temporary buffer in a scratch arena, copy the final view
//     into a long-lived arena, then `scratch_end`; or
//   - build directly in a long-lived arena when the result must outlive
//     the call.
//
// On growth, `strbuf` allocates a fresh buffer (capacity doubled) from the
// arena and memcpys the existing contents. The old buffer is left as
// arena slack — bounded at ≤2× the final size geometrically. Use
// `strbuf_make_cap()` with a reasonable initial capacity to avoid early
// reallocations in hot loops.

typedef struct {
    char    *str;   // mutable buffer (NOT NUL-terminated; same convention as `string`)
    uint64_t size;  // current length
    uint64_t cap;   // allocated capacity
} strbuf;

// Initialise to an empty, unallocated buffer (size=0, cap=0).
// Equivalent to a zero-initialised struct literal `(strbuf){0}`.
strbuf strbuf_make(void);

// Initialise with at least `cap` bytes of capacity pre-reserved. Useful
// for hot loops and large outputs to avoid repeated doublings. If
// `cap == 0` no allocation is performed; the first append will allocate.
strbuf strbuf_make_cap(Arena *arena, uint64_t cap);

// Append bytes. Grows by doubling when needed.
void strbuf_append(Arena *arena, strbuf *b, string s);
void strbuf_append_char(Arena *arena, strbuf *b, char c);
void strbuf_append_cstr(Arena *arena, strbuf *b, const char *s);
void strbuf_append_bytes(Arena *arena, strbuf *b, const void *p, uint64_t n);

// Ensure the buffer has at least `min_cap` bytes of total capacity. No-op
// if already sufficient.
void strbuf_reserve(Arena *arena, strbuf *b, uint64_t min_cap);

// View the current contents as an immutable `string` (no copy).
// Subsequent mutating calls may invalidate the returned view if growth
// moves the underlying buffer.
string strbuf_to_string(strbuf b);

#ifdef __cplusplus
}
#endif
