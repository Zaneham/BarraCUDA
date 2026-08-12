/* MLIR frontend boundary. See mlir_fe.h. The reader normally owns _start, so
 * platform_*.c is built with PLATFORM_SKIP_ENTRY and we bring corec's heap up
 * ourselves the first time anyone asks for a context. */

#include <setjmp.h>
#include <stdlib.h>

#include <base/arena.h>
#include <base/string.h>
#include <platform/platform.h>

#include "mlir_api.h"
#include "mlir_parser.h"
#include "mlir_fe.h"

struct ml_ctx {
    Arena        *arena;
    MLIR_Context  ctx;
    MLIR_OpHandle root;
    int           dead;    /* a failed parse leaves the reader mid-descent */
};

/* One mmap, process-wide, so this happens once however many contexts open. */
static int ml_up = 0;

/* Where mlir_parse_fail lands, set only across one parse. */
static jmp_buf ml_bail;
static int     ml_catching = 0;

void
mlir_parse_fail(void)
{
    if (ml_catching)
        longjmp(ml_bail, 1);

    /* Nothing to unwind to, so the standalone behaviour is the honest one. */
    exit(1);
}

ml_ctx_t *
ml_open(size_t arena_bytes)
{
    ml_ctx_t *C;

    if (!ml_up) {
        platform_init(0, NULL, NULL);
        ml_up = 1;
    }

    C = (ml_ctx_t *)calloc(1, sizeof *C);
    if (!C)
        return NULL;

    C->arena = arena_create(arena_bytes);
    if (!C->arena) {
        free(C);
        return NULL;
    }
    MLIR_SetArenaAllocator(&C->ctx, C->arena);
    return C;
}

void
ml_close(ml_ctx_t *C)
{
    if (!C)
        return;

    /* The interning caches hold handles into this arena. Upstream assumes one
     * context per process, we do not, and a stale one is a walk through freed
     * memory on the next parse rather than anything that announces itself. */
    MLIR_ResetInternRegistry();
    arena_destroy(C->arena);
    free(C);
}

int
ml_parse(ml_ctx_t *C, const char *src, size_t len)
{
    if (!C || !src || C->dead)
        return -1;

    /* Clearing the interning tables is what makes a module self-contained.
     * Every type it names is interned afresh into this context's arena, so
     * nothing outlives its own ml_close. The cost is cross-module dedup. */
    MLIR_ResetInternRegistry();

    /* Unwinding abandons whatever the reader had half built, which is safe
     * only because it all came from our arena. The context is done after
     * that though, since the reader's statics are past saving. */
    ml_catching = 1;
    if (setjmp(ml_bail)) {
        ml_catching = 0;
        C->dead = 1;
        C->root = 0;
        return -1;
    }
    C->root = MLIR_ParseTextClassic(&C->ctx,
                                    str_from_cstr_len_view_const(src, len));
    ml_catching = 0;
    return C->root ? 0 : -1;
}

void *
ml_root(const ml_ctx_t *C)
{
    return (C && C->root) ? (void *)C->root : NULL;
}

void *
ml_ctx(const ml_ctx_t *C)
{
    return C ? (void *)&C->ctx : NULL;
}

int
ml_echo(const ml_ctx_t *C, FILE *out)
{
    string s;

    if (!C || !C->root)
        return -1;

    /* Const off only because the printer allocates into the arena. */
    s = MLIR_PrintOperationClassic((MLIR_Context *)&C->ctx, C->root);
    fprintf(out, "%.*s", (int)s.size, s.str);
    return 0;
}
