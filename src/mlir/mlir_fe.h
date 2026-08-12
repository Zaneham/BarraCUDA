/* MLIR frontend boundary. Booth is c99 and the vendored reader is not, so
 * nothing of corec's or MLIR's may appear here. Opaque pointers only. */

#ifndef MLIR_FE_H
#define MLIR_FE_H

#include <stddef.h>
#include <stdio.h>

typedef struct ml_ctx ml_ctx_t;

/* Parsing never allocates outside the arena, so a module that does not fit is
 * a refusal rather than a crawl into swap. A parsed module owns everything it
 * names, which costs ml_parse a reset and rules out calling it from threads. */
ml_ctx_t   *ml_open(size_t arena_bytes);
void        ml_close(ml_ctx_t *C);

/* Returns 0 on success. The reader reports its own syntax errors on stderr. */
int         ml_parse(ml_ctx_t *C, const char *src, size_t len);

/* Reprints what was read, which is how you tell a misreading from a bad file. */
int         ml_echo(const ml_ctx_t *C, FILE *out);

/* MLIR handles, for lower.c only. Opaque so nothing else is tempted, and null
 * before a successful ml_parse. */
void       *ml_root(const ml_ctx_t *C);
void       *ml_ctx(const ml_ctx_t *C);

#endif
