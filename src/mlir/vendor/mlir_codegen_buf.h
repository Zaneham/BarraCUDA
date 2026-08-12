// Shared code-generation byte-buffer + endian/LEB helpers.
//
// Extracted verbatim from mlir_aarch64_to_macho.c so the architecture
// assemblers (mlir_aarch64_asm.c, mlir_x64_asm.c) and the binary-format
// containers (mlir_macho.c, mlir_elf.c) can share one growable-buffer
// implementation instead of each keeping a file-local copy.
//
// Header-only `static inline` so no extra translation unit has to be wired
// into every build script; unused helpers in a given TU are dropped silently.
//
// NOTE: this header is part of the tinyC self-host source set (it is included
// by mlir_aarch64_to_macho.c / mlir_aarch64_asm.c, which tinyc.wasm compiles),
// so it must stay within the tinyC C subset: no global aggregate initializers,
// no file-scope arrays of structs / char*.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint8_t *data; size_t len, cap; } Buf;

static inline void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 1024;
    while (b->len + add > nc) nc *= 2;
    b->data = (uint8_t *)realloc(b->data, nc);
    b->cap = nc;
}
static inline void buf_u8(Buf *b, uint8_t v) { buf_grow(b, 1); b->data[b->len++] = v; }
static inline void buf_append(Buf *b, const void *p, size_t n) {
    buf_grow(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static inline void buf_pad_to(Buf *b, size_t target) {
    if (target <= b->len) return;
    buf_grow(b, target - b->len);
    memset(b->data + b->len, 0, target - b->len);
    b->len = target;
}
static inline void buf_le16(Buf *b, uint16_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
}
static inline void buf_le32(Buf *b, uint32_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
    buf_u8(b, (uint8_t)((v >> 16) & 0xff));
    buf_u8(b, (uint8_t)((v >> 24) & 0xff));
}
static inline void buf_le64(Buf *b, uint64_t v) {
    for (int i = 0; i < 8; i++) buf_u8(b, (uint8_t)((v >> (8 * i)) & 0xff));
}
static inline void buf_be32(Buf *b, uint32_t v) {
    buf_u8(b, (uint8_t)((v >> 24) & 0xff));
    buf_u8(b, (uint8_t)((v >> 16) & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
    buf_u8(b, (uint8_t)(v & 0xff));
}
static inline void buf_uleb(Buf *b, uint64_t v) {
    do {
        uint8_t byte = (uint8_t)(v & 0x7f);
        v >>= 7;
        if (v) byte |= 0x80;
        buf_u8(b, byte);
    } while (v);
}
static inline void buf_cstr(Buf *b, const char *s) {
    while (*s) buf_u8(b, (uint8_t)*s++);
    buf_u8(b, 0);
}
static inline void buf_strn(Buf *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) buf_u8(b, (uint8_t)s[i]);
    buf_u8(b, 0);
}
static inline void buf_patch_le32(Buf *b, size_t pos, uint32_t v) {
    b->data[pos + 0] = (uint8_t)(v & 0xff);
    b->data[pos + 1] = (uint8_t)((v >> 8) & 0xff);
    b->data[pos + 2] = (uint8_t)((v >> 16) & 0xff);
    b->data[pos + 3] = (uint8_t)((v >> 24) & 0xff);
}
static inline void buf_patch_le64(Buf *b, size_t pos, uint64_t v) {
    for (int k = 0; k < 8; k++)
        b->data[pos + (size_t)k] = (uint8_t)(v >> (8 * k));
}
