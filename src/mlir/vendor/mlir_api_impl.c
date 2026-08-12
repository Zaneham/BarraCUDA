#include <stdio.h>
#include <base/arena.h>
#include <base/string.h>
#include <base/strbuf.h>
#include <base/format.h>
#include <base/hashtable.h>
#include <base/vector.h>
#include "tokenizer.h"
#include "mlir_api.h"
#include "mlir_op_names.h"
#include "mlir_parser.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Internal concrete IR definitions

typedef enum {
    TYPE_KIND_UNKNOWN,
    TYPE_KIND_OPAQUE,
    TYPE_KIND_INTEGER,
    TYPE_KIND_FLOAT,
    TYPE_KIND_MEMREF,
    TYPE_KIND_TENSOR,
    TYPE_KIND_FUNCTION,
    TYPE_KIND_INDEX,
    TYPE_KIND_POINTER,
    TYPE_KIND_LLVM_PTR,
    TYPE_KIND_LLVM_VOID,
    TYPE_KIND_LLVM_ARRAY,
    TYPE_KIND_LLVM_STRUCT,
    TYPE_KIND_LLVM_FUNCTION
} TypeKind;

typedef enum {
    LOC_KIND_UNKNOWN,
    LOC_KIND_FILE,
    LOC_KIND_NAME,
    LOC_KIND_CALLSITE,
    LOC_KIND_FUSED,
    LOC_KIND_REF
} LocationKind;

// Internal struct definitions using handle fields

typedef struct IR_Type {
    TypeKind kind;
    union {
        struct {
            uint32_t width;
            bool is_signed;
        } integer;
        struct {
            uint32_t width;
            bool is_bfloat;
        } floating;
        struct {
            MLIR_TypeHandle element_type;
            int64_t *shape;
            uint32_t rank;
        } shaped;
        struct {
            MLIR_TypeHandle element_type;
            uint32_t address_space;
            bool has_address_space;
        } pointer;
        struct {
            MLIR_TypeHandle *inputs;
            size_t n_inputs;
            MLIR_TypeHandle *results;
            size_t n_results;
        } function;
        struct {
            MLIR_TypeHandle element;
            uint64_t count;
        } llvm_array;
        struct {
            // Anonymous structs have name.size == 0. Identified structs are
            // interned by name (see intern_llvm_struct in this file).
            string name;
            MLIR_TypeHandle *fields;
            size_t n_fields;
            bool body_set;
        } llvm_struct;
        struct {
            // LLVM function type: exactly one return type (LLVM void marker
            // when the function returns void), N input types, optional varargs.
            MLIR_TypeHandle result;
            MLIR_TypeHandle *inputs;
            size_t n_inputs;
            bool is_var_arg;
        } llvm_function;
    } data;
} IR_Type;

typedef struct IR_Attribute {
    enum {
        ATTR_KIND_INTEGER,
        ATTR_KIND_FLOAT,
        ATTR_KIND_STRING,
        ATTR_KIND_BOOL,
        ATTR_KIND_ARRAY,
        ATTR_KIND_DICT,
        ATTR_KIND_TYPE
    } kind;
    union {
        int64_t integer_value;
        double float_value;
        string string_value;
        bool bool_value;
        struct {
            MLIR_AttributeHandle *elements;
            size_t count;
        } array;
        MLIR_TypeHandle type_value;
    } data;
    string name;
    // MLIR type of the attribute's value. Required for ATTR_KIND_INTEGER
    // and ATTR_KIND_FLOAT (and will apply to ATTR_KIND_DENSE_ELEMENTS
    // when added). For other kinds it is unused and must be
    // MLIR_INVALID_HANDLE (the zero value, set by `IR_Attribute a = {0}`).
    MLIR_TypeHandle type;
} IR_Attribute;

typedef struct IR_Location {
    LocationKind kind;
    union {
        struct { string filename; int line; int column; } file;
        struct { string name; } name;
        struct { int ref_id; } ref;
    } data;
    string original_text;
} IR_Location;

// Def-use chain node. One UseNode per operand slot (regular operand or
// successor-operand) that currently points at a real value. Each value
// owns a doubly-linked list of the UseNodes that reference it, threaded
// through prev/next. Slot-to-UseNode mapping lives in IR_Op's parallel
// `operand_uses` / `successor_operand_uses` arrays so unlinking is O(1)
// when an operand is overwritten or the owning op is erased.
typedef struct UseNode {
    struct UseNode *prev;
    struct UseNode *next;
    MLIR_OpHandle owner;
    uint16_t kind;       // 0 = regular operand, 1 = successor operand
    uint16_t succ_idx;   // kind==1: which successor segment
    uint32_t op_idx;     // index within operands or successor_operands[succ_idx]
} UseNode;

// Single row of the per-successor use-list array. Wrapped in a struct
// so the outer pointer `IR_Op::successor_operand_uses` stays at two
// pointer levels (UseSlot ⇒ TY_STRUCT inside tinyC, then one `*`
// in the field type — tinyC rejects a `UseNode ***` field directly).
typedef struct UseSlot {
    UseNode **arr;
} UseSlot;

// Lazily-allocated side struct for IR_Value metadata that is unused on the
// wasm->macho lift path (which disables def-use tracking and supplies no
// register names). See IR_Value::aux.
typedef struct IR_ValueAux {
    string register_name;
    // Head of the doubly-linked list of UseNodes whose owner ops currently
    // reference this value as an operand. `n_uses` is the cached length so
    // MLIR_GetValueNumUses is O(1).
    UseNode *use_head;
    uint32_t n_uses;
} IR_ValueAux;

typedef struct IR_Value {
    MLIR_ValueKind kind;
    uint32_t result_index;
    uintptr_t def_handle;  // MLIR_OpHandle if OP_RESULT, MLIR_BlockHandle if BLOCK_ARG
    MLIR_TypeHandle type;
    MLIR_LocationHandle location;
    // Rarely-needed-on-the-hot-path value metadata: the SSA register name
    // (only set on the parser/printer roundtrip path) and the def-use list
    // (only populated when the context tracks uses — the wasm->macho pipeline
    // opts out). Stored in a side struct allocated lazily so values on the
    // macho lift path, which set neither, cost one pointer instead of 28
    // bytes. aux is non-NULL whenever the context tracks uses or a register
    // name was supplied.
    IR_ValueAux *aux;
} IR_Value;

// Rarely-used op metadata consulted only by the classic printer and the
// parser roundtrip (trailing comments, the unnumbered location def, and the
// recorded source line). The wasm->macho lift path never sets these, so we
// store them in a side struct that is allocated lazily and left NULL on the
// hot path, keeping IR_Op compact (op count dominates peak memory there).
typedef struct IR_OpAux {
    MLIR_LocationHandle unnumbered_loc_def;
    string trailing_comment;
    // Non-canonical op name (only for OP_TYPE_UNREGISTERED / custom ops whose
    // textual name differs from op_type_to_string(op_type)). NULL str means
    // the name is derivable from op_type and is not stored.
    string opname;
    int32_t source_line_start;
    // Def-use bookkeeping, populated only when the context tracks uses (the
    // wasm->macho pipeline opts out, so these stay NULL there).
    // operand_uses[i] is the UseNode linking operands[i] into its value's use
    // list; successor_operand_uses[s].arr[i] mirrors successor_operands[s][i].
    UseNode **operand_uses;
    UseSlot *successor_operand_uses;
} IR_OpAux;

typedef struct IR_Op {
    // 32-bit fields grouped to avoid padding. Counts and line numbers never
    // approach 2^32, so uint32_t is ample and keeps IR_Op compact — the op
    // count dominates peak memory on the wasm->macho pipeline.
    MLIR_OpType op_type;
    uint32_t n_operands;
    MLIR_ValueHandle *operands;
    MLIR_TypeHandle *result_types;
    MLIR_AttributeHandle *attributes;
    MLIR_RegionHandle *regions;
    MLIR_BlockHandle *successors;
    MLIR_ValueHandle **successor_operands;
    uint64_t *n_successor_operands;
    MLIR_ValueHandle *results;
    MLIR_LocationHandle location;
    IR_OpAux *aux;
    MLIR_BlockHandle parent_block;
    uint32_t n_result_types;
    uint32_t n_attributes;
    uint32_t n_regions;
    uint32_t n_successors;
    uint32_t n_results;
} IR_Op;

typedef struct IR_Block {
    MLIR_OpHandle *operations;
    MLIR_ValueHandle *arguments;
    MLIR_RegionHandle parent_region;
    uint32_t n_operations;
    uint32_t cap_operations;
    uint32_t n_arguments;
    uint32_t cap_arguments;
} IR_Block;

typedef struct IR_Region {
    MLIR_BlockHandle *blocks;
    uint64_t n_blocks;
    uint64_t cap_blocks;
} IR_Region;

static inline IR_Op *resolve_op(MLIR_OpHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Op *)(uintptr_t)h;
}

static inline IR_Region *resolve_region(MLIR_RegionHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Region *)(uintptr_t)h;
}

static inline IR_Block *resolve_block(MLIR_BlockHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Block *)(uintptr_t)h;
}

static inline IR_Value *resolve_value(MLIR_ValueHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Value *)(uintptr_t)h;
}

static inline IR_Type *resolve_type(MLIR_TypeHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Type *)(uintptr_t)h;
}

static inline IR_Attribute *resolve_attr(MLIR_AttributeHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Attribute *)(uintptr_t)h;
}

static inline IR_Location *resolve_loc(MLIR_LocationHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Location *)(uintptr_t)h;
}

// Process-wide registry of all created ops. Native MLIR has no per-context
// op list, but the lowering pass needs one for MLIR_ReplaceAllUsesOfValue
// (which has to scan every op's operand/successor-operand arrays). The
// tinyc/parser binaries each create exactly one MLIR_Context per process,
// so a single static list is enough. Memory comes from the context arena.
static MLIR_OpHandle *g_all_ops = NULL;
static size_t         g_n_all_ops = 0;
static size_t         g_cap_all_ops = 0;

// --- Def-use chain machinery -----------------------------------------------
//
// On every assignment `op->operands[i] = v` (or successor-operand slot
// equivalent) we create a UseNode and link it into v's doubly-linked
// `use_head` list. On the reverse — slot overwrite, op erase, operand
// array resize — we unlink the corresponding UseNode. This makes
// MLIR_GetValueNumUses O(1), MLIR_ReplaceAllUsesOfValue O(uses), and
// MLIR_GetValueUseOwner O(idx). The previous scan-the-module
// implementations were O(N) per call and dominated cf->scf lifting on
// non-trivial functions.

static inline void use_list_unlink(IR_Value *v, UseNode *u) {
    if (!v || !v->aux || !u) return;
    if (u->prev) u->prev->next = u->next;
    else         v->aux->use_head = u->next;
    if (u->next) u->next->prev = u->prev;
    u->prev = u->next = NULL;
    if (v->aux->n_uses) v->aux->n_uses--;
}

static inline void use_list_push_front(IR_Value *v, UseNode *u) {
    if (!v || !v->aux || !u) return;
    u->prev = NULL;
    u->next = v->aux->use_head;
    if (v->aux->use_head) v->aux->use_head->prev = u;
    v->aux->use_head = u;
    v->aux->n_uses++;
}

// Register `slot_value` as a use by `owner_op` at the indicated
// regular-operand slot (kind=0, op_idx=i, succ_idx unused). Returns the
// newly-allocated UseNode (or NULL if slot_value is invalid). The
// returned pointer is stored in IR_Op::operand_uses[i] so it can later
// be unlinked in O(1).
static UseNode *use_register_operand(MLIR_Context *ctx, MLIR_OpHandle owner_op,
                                     MLIR_ValueHandle slot_value, uint32_t op_idx) {
    if (slot_value == MLIR_INVALID_HANDLE) return NULL;
    IR_Value *v = resolve_value(slot_value);
    if (!v) return NULL;
    UseNode *u = arena_new(ctx->arena, UseNode);
    u->owner = owner_op;
    u->kind = 0;
    u->succ_idx = 0;
    u->op_idx = op_idx;
    use_list_push_front(v, u);
    return u;
}

static UseNode *use_register_succ_operand(MLIR_Context *ctx, MLIR_OpHandle owner_op,
                                          MLIR_ValueHandle slot_value,
                                          uint32_t succ_idx, uint32_t op_idx) {
    if (slot_value == MLIR_INVALID_HANDLE) return NULL;
    IR_Value *v = resolve_value(slot_value);
    if (!v) return NULL;
    UseNode *u = arena_new(ctx->arena, UseNode);
    u->owner = owner_op;
    u->kind = 1;
    u->succ_idx = (uint16_t)succ_idx;
    u->op_idx = op_idx;
    use_list_push_front(v, u);
    return u;
}

// Unlink the UseNode currently associated with `slot_value`. No-op if
// slot_value is INVALID or `u` is NULL. Safe to call repeatedly.
static inline void use_unregister(MLIR_ValueHandle slot_value, UseNode *u) {
    if (slot_value == MLIR_INVALID_HANDLE || !u) return;
    IR_Value *v = resolve_value(slot_value);
    if (!v) return;
    use_list_unlink(v, u);
}

// Allocate an IR_OpAux initialised to the values the getters treat as
// "absent" (so forcing an aux for def-use bookkeeping doesn't change the
// trailing-comment / source-line / opname / unnumbered-loc getters).
static inline IR_OpAux *op_aux_default(Arena *arena) {
    IR_OpAux *a = arena_new_array(arena, IR_OpAux, 1);
    *a = (IR_OpAux){0};
    a->source_line_start = -1;
    return a;
}

static void register_op(MLIR_Context *ctx, MLIR_OpHandle h) {
    if (!ctx || !ctx->arena) return;
    if (g_n_all_ops == g_cap_all_ops) {
        size_t nc = g_cap_all_ops ? g_cap_all_ops * 2 : 256;
        MLIR_OpHandle *p = arena_new_array(ctx->arena, MLIR_OpHandle, nc);
        if (g_all_ops && g_n_all_ops)
            memcpy(p, g_all_ops, g_n_all_ops * sizeof(MLIR_OpHandle));
        g_all_ops = p;
        g_cap_all_ops = nc;
    }
    g_all_ops[g_n_all_ops++] = h;
}

static inline MLIR_OpHandle alloc_op(MLIR_Context *ctx, IR_Op op) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Op *slot = arena_new(ctx->arena, IR_Op);
    *slot = op;
    MLIR_OpHandle h = (MLIR_OpHandle)(uintptr_t)slot;
    // g_all_ops is only consulted by MLIR_ReplaceAllUsesOfValue (def-use).
    // When tracking is disabled (e.g. the wasm->macho pipeline) skip it: it
    // saves memory and, crucially, avoids dangling g_all_ops entries when the
    // pipeline frees an intermediate module's arena between passes.
    if (!ctx->no_def_use_tracking)
        register_op(ctx, h);
    return h;
}

static inline MLIR_RegionHandle alloc_region(MLIR_Context *ctx, IR_Region r) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Region *slot = arena_new(ctx->arena, IR_Region);
    *slot = r;
    return (MLIR_RegionHandle)(uintptr_t)slot;
}

static inline MLIR_BlockHandle alloc_block(MLIR_Context *ctx, IR_Block b) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Block *slot = arena_new(ctx->arena, IR_Block);
    *slot = b;
    return (MLIR_BlockHandle)(uintptr_t)slot;
}

static inline MLIR_ValueHandle alloc_value(MLIR_Context *ctx, IR_Value v) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Value *slot = arena_new(ctx->arena, IR_Value);
    *slot = v;
    return (MLIR_ValueHandle)(uintptr_t)slot;
}

// Allocate an IR_ValueAux for a freshly-created value, or return NULL when
// none is needed. An aux is required whenever the context tracks uses (so the
// def-use list can be populated as ops reference the value) or a register name
// was supplied. On the wasm->macho lift path neither holds, so values cost
// only the inline aux pointer.
static inline IR_ValueAux *value_aux_create(MLIR_Context *ctx, string register_name) {
    if (!ctx || !ctx->arena) return NULL;
    bool tracking = !ctx->no_def_use_tracking;
    if (!tracking && register_name.size == 0) return NULL;
    IR_ValueAux *a = arena_new(ctx->arena, IR_ValueAux);
    *a = (IR_ValueAux){0};
    a->register_name = register_name;
    return a;
}

static inline MLIR_TypeHandle alloc_type(MLIR_Context *ctx, IR_Type t) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Type *slot = arena_new(ctx->arena, IR_Type);
    *slot = t;
    return (MLIR_TypeHandle)(uintptr_t)slot;
}

// Compare two IR_Type structures for value-equality. Used to intern
// types so that two semantically-equivalent types share the same
// MLIR_TypeHandle (matching upstream MLIR's IntegerType::get etc.).
// This makes `t1 == t2` a valid type-identity check across the API —
// e.g., the get_undef_value / get_switch_value caches in the
// cf->scf lifter rely on it.
static bool type_eq(const IR_Type *a, const IR_Type *b) {
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case TYPE_KIND_UNKNOWN:
        case TYPE_KIND_OPAQUE:
        case TYPE_KIND_INDEX:
        case TYPE_KIND_LLVM_PTR:
        case TYPE_KIND_LLVM_VOID:
            return true;
        case TYPE_KIND_INTEGER:
            return a->data.integer.width == b->data.integer.width &&
                   a->data.integer.is_signed == b->data.integer.is_signed;
        case TYPE_KIND_FLOAT:
            return a->data.floating.width == b->data.floating.width &&
                   a->data.floating.is_bfloat == b->data.floating.is_bfloat;
        case TYPE_KIND_TENSOR:
        case TYPE_KIND_MEMREF:
            if (a->data.shaped.element_type != b->data.shaped.element_type) return false;
            if (a->data.shaped.rank != b->data.shaped.rank) return false;
            if (a->data.shaped.rank == 0) return true;
            return memcmp(a->data.shaped.shape, b->data.shaped.shape,
                          (size_t)a->data.shaped.rank * sizeof(int64_t)) == 0;
        case TYPE_KIND_POINTER:
            return a->data.pointer.element_type == b->data.pointer.element_type &&
                   a->data.pointer.has_address_space == b->data.pointer.has_address_space &&
                   a->data.pointer.address_space == b->data.pointer.address_space;
        case TYPE_KIND_LLVM_ARRAY:
            return a->data.llvm_array.element == b->data.llvm_array.element &&
                   a->data.llvm_array.count == b->data.llvm_array.count;
        case TYPE_KIND_LLVM_FUNCTION:
            if (a->data.llvm_function.result != b->data.llvm_function.result) return false;
            if (a->data.llvm_function.is_var_arg != b->data.llvm_function.is_var_arg) return false;
            if (a->data.llvm_function.n_inputs != b->data.llvm_function.n_inputs) return false;
            if (a->data.llvm_function.n_inputs == 0) return true;
            return memcmp(a->data.llvm_function.inputs, b->data.llvm_function.inputs,
                          a->data.llvm_function.n_inputs * sizeof(MLIR_TypeHandle)) == 0;
        case TYPE_KIND_FUNCTION:
            if (a->data.function.n_inputs != b->data.function.n_inputs) return false;
            if (a->data.function.n_results != b->data.function.n_results) return false;
            if (a->data.function.n_inputs > 0 &&
                memcmp(a->data.function.inputs, b->data.function.inputs,
                       a->data.function.n_inputs * sizeof(MLIR_TypeHandle)) != 0) return false;
            if (a->data.function.n_results > 0 &&
                memcmp(a->data.function.results, b->data.function.results,
                       a->data.function.n_results * sizeof(MLIR_TypeHandle)) != 0) return false;
            return true;
        case TYPE_KIND_LLVM_STRUCT:
            // Identified (name.size > 0) handled by intern_llvm_struct.
            // We don't intern LLVM struct types through this path; the
            // dedicated intern_llvm_struct keeps name->handle identity.
            return false;
    }
    return false;
}

// Process-wide type-interning cache. Matches the pattern of
// intern_llvm_struct (g_struct_handles): stored handles point into a
// context's arena; we assume effectively single-context-per-process
// (true for both the tinyc driver and the parser).
static MLIR_TypeHandle *g_type_handles = NULL;
static size_t          g_n_types = 0;
static size_t          g_cap_types = 0;

static MLIR_TypeHandle intern_type(MLIR_Context *ctx, IR_Type t) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < g_n_types; i++) {
        IR_Type *cur = resolve_type(g_type_handles[i]);
        if (cur && type_eq(cur, &t)) return g_type_handles[i];
    }
    // Pin the interned object and the registry array to a persistent arena
    // when requested, so cached handles survive temp-arena churn in the
    // streaming backend (see MLIR_Context.type_arena).
    Arena *save = ctx->arena;
    if (ctx->type_arena) ctx->arena = ctx->type_arena;
    MLIR_TypeHandle h = alloc_type(ctx, t);
    if (g_n_types == g_cap_types) {
        size_t nc = g_cap_types ? g_cap_types * 2 : 64;
        MLIR_TypeHandle *nh = arena_new_array(ctx->arena, MLIR_TypeHandle, nc);
        if (g_n_types) {
            memcpy(nh, g_type_handles, g_n_types * sizeof(MLIR_TypeHandle));
        }
        g_type_handles = nh;
        g_cap_types = nc;
    }
    ctx->arena = save;
    g_type_handles[g_n_types++] = h;
    return h;
}

static inline MLIR_AttributeHandle alloc_attr_obj(MLIR_Context *ctx, IR_Attribute a) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Attribute *slot = arena_new(ctx->arena, IR_Attribute);
    *slot = a;
    return (MLIR_AttributeHandle)(uintptr_t)slot;
}

// Copy `s` into ctx->arena so the returned string is owned by the context's
// arena rather than aliasing caller-provided memory. Multi-pass pipelines
// (wasm->macho) free intermediate arenas between passes, so any string stored
// in an IR object that must outlive its producing pass has to live in the
// destination arena. Empty/NULL strings are returned unchanged.
static inline string arena_dup_str(MLIR_Context *ctx, string s) {
    if (!ctx || !ctx->arena || s.size == 0 || !s.str) return s;
    char *p = arena_new_array(ctx->arena, char, s.size);
    memcpy(p, s.str, s.size);
    return (string){ p, s.size };
}

static inline MLIR_LocationHandle alloc_loc(MLIR_Context *ctx, IR_Location l) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Location *slot = arena_new(ctx->arena, IR_Location);
    *slot = l;
    return (MLIR_LocationHandle)(uintptr_t)slot;
}

// API lifecycle
void MLIR_InitApi(MLIR_Context *ctx, MLIR_OpHandle root) {
    (void)ctx;
    (void)root;
}

void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena) {
    if (!ctx) return;
    ctx->arena = arena;
}

Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx) {
    return ctx ? ctx->arena : NULL;
}

// Operation creation
MLIR_OpHandle MLIR_CreateOpWithSuccessors(
    MLIR_Context *ctx,
    MLIR_OpType type,
    string opname,
    MLIR_AttributeHandle *attributes, size_t n_attributes,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_BlockHandle *successors, size_t n_successors,
    MLIR_ValueHandle **successor_operands,
    size_t *n_successor_operands,
    MLIR_LocationHandle location,
    MLIR_LocationHandle unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {

    IR_Op op = {0};
    op.op_type = type;
    Arena *arena = ctx ? ctx->arena : NULL;
    // The lowering pass (and other callers) frequently builds these arrays
    // on the stack and hands them to us; copy into the arena so they
    // survive past the caller's stack frame.
    #define DUP(dst, dn, src, sn, T) do { \
        (dn) = (uint32_t)(sn); \
        if ((sn) > 0 && arena) { \
            (dst) = arena_new_array(arena, T, (sn)); \
            memcpy((dst), (src), (sn) * sizeof(T)); \
        } else { (dst) = NULL; } \
    } while (0)
    DUP(op.attributes, op.n_attributes, attributes, n_attributes, MLIR_AttributeHandle);
    DUP(op.result_types, op.n_result_types, result_types, n_result_types, MLIR_TypeHandle);
    DUP(op.results, op.n_results, results, n_results, MLIR_ValueHandle);
    DUP(op.operands, op.n_operands, operands, n_operands, MLIR_ValueHandle);
    DUP(op.regions, op.n_regions, regions, n_regions, MLIR_RegionHandle);
    DUP(op.successors, op.n_successors, successors, n_successors, MLIR_BlockHandle);
    // The def-use chains (operand_uses / successor_operand_uses + UseNodes)
    // are only consulted by the cf->scf and lower-to-LLVM passes. Contexts
    // that opt out (e.g. the wasm->macho pipeline) skip these allocations
    // entirely, which is the dominant per-op memory cost on that path.
    bool track_uses = !(ctx && ctx->no_def_use_tracking);
    // Allocate the aux side struct when use tracking is on (it holds the
    // def-use arrays) or when any printer/parser metadata diverges from the
    // getter defaults. The wasm->macho lift passes defaults and opts out of
    // tracking, so aux stays NULL on the hot path.
    bool store_name = !str_eq(opname, op_type_to_string(type));
    if (arena && (track_uses || store_name ||
                  unnumbered_loc_def != MLIR_INVALID_HANDLE ||
                  trailing_comment.size != 0 || source_line_start != -1)) {
        IR_OpAux *aux = op_aux_default(arena);
        aux->unnumbered_loc_def = unnumbered_loc_def;
        aux->trailing_comment = trailing_comment;
        aux->opname = store_name ? opname : (string){0};
        aux->source_line_start = (int32_t)source_line_start;
        op.aux = aux;
    }
    if (n_successors > 0 && arena) {
        op.successor_operands = arena_new_array(arena, MLIR_ValueHandle *, n_successors);
        op.n_successor_operands = arena_new_array(arena, uint64_t, n_successors);
        if (track_uses)
            op.aux->successor_operand_uses = arena_new_array(arena, UseSlot, n_successors);
        for (size_t s = 0; s < n_successors; s++) {
            uint64_t cnt = n_successor_operands ? (uint64_t)n_successor_operands[s] : 0;
            op.n_successor_operands[s] = cnt;
            if (cnt > 0 && successor_operands && successor_operands[s]) {
                op.successor_operands[s] = arena_new_array(arena, MLIR_ValueHandle, cnt);
                memcpy(op.successor_operands[s], successor_operands[s], cnt * sizeof(MLIR_ValueHandle));
                if (track_uses) {
                    op.aux->successor_operand_uses[s].arr = arena_new_array(arena, UseNode *, cnt);
                    for (size_t k = 0; k < cnt; k++) op.aux->successor_operand_uses[s].arr[k] = NULL;
                }
            } else {
                op.successor_operands[s] = NULL;
                if (track_uses) op.aux->successor_operand_uses[s].arr = NULL;
            }
        }
    }
    if (n_operands > 0 && arena && track_uses) {
        op.aux->operand_uses = arena_new_array(arena, UseNode *, n_operands);
        for (size_t i = 0; i < n_operands; i++) op.aux->operand_uses[i] = NULL;
    }
    #undef DUP
    op.location = location;

    MLIR_OpHandle handle = alloc_op(ctx, op);

    // Wire up def-use chains for every operand slot we just copied. We do
    // this after alloc_op so each UseNode carries the final op handle.
    IR_Op *stored = resolve_op(handle);
    if (stored && track_uses) {
        for (size_t i = 0; i < stored->n_operands; i++) {
            stored->aux->operand_uses[i] = use_register_operand(
                ctx, handle, stored->operands[i], (uint32_t)i);
        }
        for (size_t s = 0; s < stored->n_successors; s++) {
            uint64_t cnt = stored->n_successor_operands[s];
            for (uint64_t k = 0; k < cnt; k++) {
                stored->aux->successor_operand_uses[s].arr[k] = use_register_succ_operand(
                    ctx, handle, stored->successor_operands[s][k],
                    (uint32_t)s, (uint32_t)k);
            }
        }
    }

    // Set def_handle on result values
    for (size_t i = 0; i < n_results; i++) {
        IR_Value *v = resolve_value(results[i]);
        if (v) {
            v->def_handle = handle;
            if (i < n_result_types && result_types[i] != MLIR_INVALID_HANDLE && v->type == MLIR_INVALID_HANDLE) {
                v->type = result_types[i];
            }
        }
    }

    return handle;
}

MLIR_OpHandle MLIR_CreateOp(
    MLIR_Context *ctx,
    MLIR_OpType type,
    string opname,
    MLIR_AttributeHandle *attributes, size_t n_attributes,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle location,
    MLIR_LocationHandle unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {
    return MLIR_CreateOpWithSuccessors(
        ctx, type, opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        operands, n_operands,
        regions, n_regions,
        NULL, 0, NULL, NULL,
        location, unnumbered_loc_def, trailing_comment, source_line_start);
}

void MLIR_AppendBlockOp(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_OpHandle op) {
    IR_Block *block = resolve_block(bh);
    if (!block || !ctx || !ctx->arena) return;
    Arena *arena = ctx->arena;
    if (block->n_operations >= block->cap_operations) {
        uint64_t new_cap = block->cap_operations ? block->cap_operations * 2 : 4;
        MLIR_OpHandle *new_ops = arena_new_array(arena, MLIR_OpHandle, new_cap);
        if (block->operations) memcpy(new_ops, block->operations, block->n_operations * sizeof(MLIR_OpHandle));
        block->operations = new_ops;
        block->cap_operations = (uint32_t)new_cap;
    }
    block->operations[block->n_operations] = op;
    block->n_operations++;
    IR_Op *o = resolve_op(op);
    if (o) o->parent_block = bh;
}

void MLIR_InsertBlockOpBeforeTerminator(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_OpHandle op) {
    // Legacy text backend has no terminator concept; fall back to append.
    MLIR_AppendBlockOp(ctx, bh, op);
}

void MLIR_InsertBlockOpAtIndex(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_OpHandle op, size_t idx) {
    IR_Block *block = resolve_block(bh);
    if (!block || !ctx || !ctx->arena) return;
    if (idx > block->n_operations) idx = block->n_operations;
    Arena *arena = ctx->arena;
    if (block->n_operations >= block->cap_operations) {
        uint64_t new_cap = block->cap_operations ? block->cap_operations * 2 : 4;
        MLIR_OpHandle *new_ops = arena_new_array(arena, MLIR_OpHandle, new_cap);
        if (block->operations && idx > 0)
            memcpy(new_ops, block->operations, idx * sizeof(MLIR_OpHandle));
        if (block->operations && idx < block->n_operations)
            memcpy(new_ops + idx + 1, block->operations + idx,
                   (block->n_operations - idx) * sizeof(MLIR_OpHandle));
        block->operations = new_ops;
        block->cap_operations = (uint32_t)new_cap;
    } else {
        // In-place shift right to make room at idx.
        if (idx < block->n_operations)
            memmove(block->operations + idx + 1, block->operations + idx,
                    (block->n_operations - idx) * sizeof(MLIR_OpHandle));
    }
    block->operations[idx] = op;
    block->n_operations++;
    IR_Op *o = resolve_op(op);
    if (o) o->parent_block = bh;
}

void MLIR_AppendBlockArg(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_ValueHandle arg) {
    IR_Block *block = resolve_block(bh);
    if (!block || !ctx || !ctx->arena) return;
    Arena *arena = ctx->arena;
    if (block->n_arguments >= block->cap_arguments) {
        uint64_t new_cap = block->cap_arguments ? block->cap_arguments * 2 : 4;
        MLIR_ValueHandle *new_args = arena_new_array(arena, MLIR_ValueHandle, new_cap);
        if (block->arguments) memcpy(new_args, block->arguments, block->n_arguments * sizeof(MLIR_ValueHandle));
        block->arguments = new_args;
        block->cap_arguments = (uint32_t)new_cap;
    }
    block->arguments[block->n_arguments] = arg;
    block->n_arguments++;

    // Set def_handle to block for block arguments
    IR_Value *v = resolve_value(arg);
    if (v) v->def_handle = bh;
}

void MLIR_AppendRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle rh, MLIR_BlockHandle block) {
    IR_Region *region = resolve_region(rh);
    if (!region || !ctx || !ctx->arena) return;
    Arena *arena = ctx->arena;
    if (region->n_blocks >= region->cap_blocks) {
        uint64_t new_cap = region->cap_blocks ? region->cap_blocks * 2 : 4;
        MLIR_BlockHandle *new_blocks = arena_new_array(arena, MLIR_BlockHandle, new_cap);
        if (region->blocks) memcpy(new_blocks, region->blocks, region->n_blocks * sizeof(MLIR_BlockHandle));
        region->blocks = new_blocks;
        region->cap_blocks = new_cap;
    }
    region->blocks[region->n_blocks] = block;
    region->n_blocks++;
    IR_Block *b = resolve_block(block);
    if (b) b->parent_region = rh;
}

size_t MLIR_GetRegionNumBlocks(MLIR_RegionHandle rh) {
    IR_Region *r = resolve_region(rh);
    return r ? r->n_blocks : 0;
}

MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_RegionHandle rh, size_t idx) {
    IR_Region *r = resolve_region(rh);
    if (!r || idx >= r->n_blocks) return MLIR_INVALID_HANDLE;
    return r->blocks[idx];
}

size_t MLIR_GetBlockNumOps(MLIR_BlockHandle bh) {
    IR_Block *b = resolve_block(bh);
    return b ? b->n_operations : 0;
}

MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle bh, size_t idx) {
    IR_Block *b = resolve_block(bh);
    if (!b || idx >= b->n_operations) return MLIR_INVALID_HANDLE;
    return b->operations[idx];
}

MLIR_OpType MLIR_GetOpType(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->op_type : OP_TYPE_UNREGISTERED;
}

size_t MLIR_GetOpNumRegions(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_regions : 0;
}

MLIR_RegionHandle MLIR_GetOpRegion(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_regions) return MLIR_INVALID_HANDLE;
    return op->regions[idx];
}

string MLIR_GetOpTrailingComment(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return (op && op->aux) ? op->aux->trailing_comment : str_lit("");
}

size_t MLIR_GetOpNumAttributes(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_attributes : 0;
}

MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_attributes) return MLIR_INVALID_HANDLE;
    return op->attributes[idx];
}

MLIR_AttributeHandle MLIR_GetOpAttributeByName(MLIR_OpHandle oh, const char *name) {
    IR_Op *op = resolve_op(oh);
    if (!op) return MLIR_INVALID_HANDLE;
    size_t nlen = strlen(name);
    for (size_t i = 0; i < op->n_attributes; i++) {
        string an = MLIR_GetAttributeName(op->attributes[i]);
        if (an.size == nlen && memcmp(an.str, name, nlen) == 0)
            return op->attributes[i];
    }
    return MLIR_INVALID_HANDLE;
}

size_t MLIR_GetOpNumOperands(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_operands : 0;
}

MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_operands) return MLIR_INVALID_HANDLE;
    return op->operands[idx];
}

size_t MLIR_GetOpNumResults(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_results : 0;
}

MLIR_ValueHandle MLIR_GetOpResult(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_results) return MLIR_INVALID_HANDLE;
    return op->results[idx];
}

MLIR_LocationHandle MLIR_GetOpLocation(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->location : MLIR_INVALID_HANDLE;
}

string MLIR_GetOpName(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    if (!op) return str_lit("");
    if (op->aux && op->aux->opname.str) return op->aux->opname;
    return op_type_to_string(op->op_type);
}

string MLIR_GetOpName_string(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    if (!op) return str_lit("");
    if (op->aux && op->aux->opname.str) return op->aux->opname;
    return op_type_to_string(op->op_type);
}

size_t MLIR_GetOpNumResultTypes(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_result_types : 0;
}

MLIR_TypeHandle MLIR_GetOpResult_type(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_result_types) return MLIR_INVALID_HANDLE;
    return op->result_types[idx];
}

// Native backend doesn't track successors/successor-operands separately;
// the classic parser stores _target/_ntrue/_nfalse + interleaves operands.
// Return 0/invalid so the printer falls back to its attribute-based path.
size_t MLIR_GetOpNumSuccessors(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_successors : 0;
}
MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_successors) return MLIR_INVALID_HANDLE;
    return op->successors[idx];
}
size_t MLIR_GetOpNumSuccessorOperands(MLIR_OpHandle oh, size_t s) {
    IR_Op *op = resolve_op(oh);
    if (!op || s >= op->n_successors || !op->n_successor_operands) return 0;
    return (size_t)op->n_successor_operands[s];
}
MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_OpHandle oh, size_t s, size_t i) {
    IR_Op *op = resolve_op(oh);
    if (!op || s >= op->n_successors || !op->successor_operands) return MLIR_INVALID_HANDLE;
    if (i >= (size_t)op->n_successor_operands[s]) return MLIR_INVALID_HANDLE;
    return op->successor_operands[s][i];
}

size_t MLIR_GetBlockNumArgs(MLIR_BlockHandle bh) {
    IR_Block *b = resolve_block(bh);
    return b ? b->n_arguments : 0;
}

MLIR_ValueHandle MLIR_GetBlockArg(MLIR_BlockHandle bh, size_t idx) {
    IR_Block *b = resolve_block(bh);
    if (!b || idx >= b->n_arguments) return MLIR_INVALID_HANDLE;
    return b->arguments[idx];
}

size_t MLIR_GetBlockIndex(MLIR_BlockHandle bh) {
    IR_Block *b = resolve_block(bh);
    if (!b) return (size_t)-1;
    IR_Region *r = resolve_region(b->parent_region);
    if (!r) return (size_t)-1;
    for (size_t i = 0; i < r->n_blocks; i++) {
        if (r->blocks[i] == bh) return i;
    }
    return (size_t)-1;
}

MLIR_LocationHandle MLIR_GetValueLocation(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->location : MLIR_INVALID_HANDLE;
}

// Type to string
string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle op) {
    (void)ctx; (void)op;
    return str_lit("error: MLIR_PrintOperationUpstream requires the upstream backend\n");
}

MLIR_OpHandle MLIR_ParseTextUpstream(MLIR_Context *ctx, string text) {
    (void)ctx; (void)text;
    return MLIR_INVALID_HANDLE;
}

string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle th) {
    IR_Type *type = resolve_type(th);
    if (!type) return str_lit("null");
    if (!ctx || !ctx->arena) return str_lit("null");
    Arena *arena = ctx->arena;

    switch (type->kind) {
        case TYPE_KIND_UNKNOWN:
            return str_lit("?");
        case TYPE_KIND_OPAQUE:
            return str_lit("unknown");
        case TYPE_KIND_INTEGER:
            // Signless `i{w}` regardless of `is_signed` — MLIR integers are
            // signless by default and the upstream backend always produces
            // `i{w}` from `MLIR_CreateTypeInteger`, so match it.
            return format(arena, str_lit("i{}"), (int64_t)type->data.integer.width);
        case TYPE_KIND_FLOAT:
            if (type->data.floating.is_bfloat && type->data.floating.width == 16) {
                return str_lit("bf16");
            }
            return format(arena, str_lit("f{}"), (int64_t)type->data.floating.width);
        case TYPE_KIND_TENSOR: {
            MLIR_TypeHandle elem_h = type->data.shaped.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    strbuf shape_str = strbuf_make();
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            strbuf_append(arena, &shape_str, str_lit("?x"));
                        } else {
                            strbuf_append(arena, &shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("tensor<{}{}>"), strbuf_to_string(shape_str), elem_str);
                } else {
                    return format(arena, str_lit("tensor<{}>"), elem_str);
                }
            }
            return str_lit("tensor<?>");
        }
        case TYPE_KIND_MEMREF: {
            MLIR_TypeHandle elem_h = type->data.shaped.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    strbuf shape_str = strbuf_make();
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            strbuf_append(arena, &shape_str, str_lit("?x"));
                        } else {
                            strbuf_append(arena, &shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("memref<{}{}>"), strbuf_to_string(shape_str), elem_str);
                } else {
                    return format(arena, str_lit("memref<{}>"), elem_str);
                }
            }
            return str_lit("memref<?>");
        }
        case TYPE_KIND_POINTER: {
            MLIR_TypeHandle elem_h = type->data.pointer.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.pointer.has_address_space) {
                    return format(arena, str_lit("!tt.ptr<{}, {}>"), elem_str, (int64_t)type->data.pointer.address_space);
                } else {
                    return format(arena, str_lit("!tt.ptr<{}>"), elem_str);
                }
            }
            return str_lit("!tt.ptr<?>");
        }
        case TYPE_KIND_INDEX:
            return str_lit("index");
        case TYPE_KIND_FUNCTION: {
            strbuf in_str = strbuf_make();
            for (size_t i = 0; i < type->data.function.n_inputs; i++) {
                if (i > 0) strbuf_append(arena, &in_str, str_lit(", "));
                strbuf_append(arena, &in_str,
                              MLIR_GetTypeString(ctx, type->data.function.inputs[i]));
            }
            string out_str = str_lit("");
            size_t nr = type->data.function.n_results;
            if (nr == 1) {
                out_str = MLIR_GetTypeString(ctx, type->data.function.results[0]);
            } else {
                strbuf body = strbuf_make();
                for (size_t i = 0; i < nr; i++) {
                    if (i > 0) strbuf_append(arena, &body, str_lit(", "));
                    strbuf_append(arena, &body,
                                  MLIR_GetTypeString(ctx, type->data.function.results[i]));
                }
                out_str = format(arena, str_lit("({})"), strbuf_to_string(body));
            }
            return format(arena, str_lit("({}) -> {}"), strbuf_to_string(in_str), out_str);
        }
        case TYPE_KIND_LLVM_PTR:
            return str_lit("!llvm.ptr");
        case TYPE_KIND_LLVM_VOID:
            return str_lit("!llvm.void");
        case TYPE_KIND_LLVM_ARRAY: {
            string elem = MLIR_GetTypeString(ctx, type->data.llvm_array.element);
            return format(arena, str_lit("!llvm.array<{} x {}>"),
                          (int64_t)type->data.llvm_array.count, elem);
        }
        case TYPE_KIND_LLVM_STRUCT: {
            // Anonymous: !llvm.struct<(T1, T2)>; identified: !llvm.struct<"name", (T1, T2)>.
            // Within nested struct fields MLIR omits the outer "!llvm." prefix —
            // but the translator's print_llvm_type_text accepts both, so we
            // always emit the canonical form here for simplicity.
            strbuf body = strbuf_make();
            strbuf_append(arena, &body, str_lit("("));
            for (size_t i = 0; i < type->data.llvm_struct.n_fields; i++) {
                if (i > 0) strbuf_append(arena, &body, str_lit(", "));
                strbuf_append(arena, &body,
                              MLIR_GetTypeString(ctx, type->data.llvm_struct.fields[i]));
            }
            strbuf_append(arena, &body, str_lit(")"));
            if (type->data.llvm_struct.name.size > 0) {
                return format(arena, str_lit("!llvm.struct<\"{}\", {}>"),
                              type->data.llvm_struct.name, strbuf_to_string(body));
            }
            return format(arena, str_lit("!llvm.struct<{}>"), strbuf_to_string(body));
        }
        case TYPE_KIND_LLVM_FUNCTION: {
            string ret = MLIR_GetTypeString(ctx, type->data.llvm_function.result);
            strbuf body = strbuf_make();
            for (size_t i = 0; i < type->data.llvm_function.n_inputs; i++) {
                if (i > 0) strbuf_append(arena, &body, str_lit(", "));
                strbuf_append(arena, &body,
                              MLIR_GetTypeString(ctx, type->data.llvm_function.inputs[i]));
            }
            if (type->data.llvm_function.is_var_arg) {
                if (type->data.llvm_function.n_inputs) {
                    strbuf_append(arena, &body, str_lit(", "));
                }
                strbuf_append(arena, &body, str_lit("..."));
            }
            return format(arena, str_lit("!llvm.func<{} ({})>"), ret, strbuf_to_string(body));
        }
        default:
            return str_lit("unknown");
    }
}

// Type creation
MLIR_TypeHandle MLIR_CreateTypeInteger(MLIR_Context *ctx, uint32_t width, bool is_signed) {
    // MLIR integer types are signless by default. The public API takes an
    // `is_signed` flag for forward-compat, but the upstream backend ignores
    // it (always constructs a signless `IntegerType`). Match that so both
    // backends produce the same canonical type identity for `i{w}` and the
    // wasm pipeline / lowering passes can compare types via handle equality.
    (void)is_signed;
    IR_Type t = {0};
    t.kind = TYPE_KIND_INTEGER;
    t.data.integer.width = width;
    t.data.integer.is_signed = false;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *ctx, uint32_t width, bool is_bfloat) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_FLOAT;
    t.data.floating.width = width;
    t.data.floating.is_bfloat = is_bfloat;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_INDEX;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_UNKNOWN;
    return intern_type(ctx, t);
}

static void copy_shape_to_arena(MLIR_Context *ctx, IR_Type *type, const int64_t *shape, size_t rank) {
    type->data.shaped.rank = (uint32_t)rank;
    if (rank > 0 && shape) {
        if (!ctx || !ctx->arena) {
            type->data.shaped.shape = NULL;
            type->data.shaped.rank = 0;
            return;
        }
        type->data.shaped.shape = arena_new_array(ctx->arena, int64_t, rank);
        for (size_t i = 0; i < rank; i++) {
            type->data.shaped.shape[i] = shape[i];
        }
    } else {
        type->data.shaped.shape = NULL;
    }
}

MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_TENSOR;
    t.data.shaped.element_type = element_type;
    copy_shape_to_arena(ctx, &t, shape, rank);
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_MEMREF;
    t.data.shaped.element_type = element_type;
    copy_shape_to_arena(ctx, &t, shape, rank);
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_POINTER;
    t.data.pointer.element_type = element_type;
    t.data.pointer.has_address_space = has_address_space;
    t.data.pointer.address_space = address_space;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_PTR;
    return intern_type(ctx, t);
}

// Identified LLVM structs are interned by name within a context: the same
// printed reference (`!llvm.struct<"foo", ...>`) must resolve to the same
// type handle across all uses. We use a process-wide static table keyed
// by the string contents (cheap given the small N in tinyc programs).
static MLIR_TypeHandle *g_struct_handles = NULL;
static string         *g_struct_names   = NULL;
static size_t          g_n_structs = 0;
static size_t          g_cap_structs = 0;

static MLIR_TypeHandle intern_llvm_struct(MLIR_Context *ctx, string name) {
    for (size_t i = 0; i < g_n_structs; i++) {
        if (g_struct_names[i].size == name.size &&
            memcmp(g_struct_names[i].str, name.str, name.size) == 0) {
            return g_struct_handles[i];
        }
    }
    // Pin to the persistent type arena when requested (see intern_type).
    Arena *save = ctx->arena;
    if (ctx->type_arena) ctx->arena = ctx->type_arena;
    if (g_n_structs == g_cap_structs) {
        size_t nc = g_cap_structs ? g_cap_structs * 2 : 32;
        MLIR_TypeHandle *nh = arena_new_array(ctx->arena, MLIR_TypeHandle, nc);
        string *nn = arena_new_array(ctx->arena, string, nc);
        if (g_n_structs) {
            memcpy(nh, g_struct_handles, g_n_structs * sizeof(MLIR_TypeHandle));
            memcpy(nn, g_struct_names, g_n_structs * sizeof(string));
        }
        g_struct_handles = nh; g_struct_names = nn; g_cap_structs = nc;
    }
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_STRUCT;
    t.data.llvm_struct.name = name;
    MLIR_TypeHandle h = alloc_type(ctx, t);
    ctx->arena = save;
    g_struct_handles[g_n_structs] = h;
    g_struct_names[g_n_structs] = name;
    g_n_structs++;
    return h;
}

// Drop all process-wide intern/registry state so it no longer references any
// arena. Used by multi-pass pipelines (e.g. wasm->macho) right before they
// switch ctx->arena to a fresh arena and free the previous one: any cached
// handles pointed into the about-to-be-freed arena, and the dedup caches must
// start empty so the next pass re-interns its types/structs into the new
// arena (making that pass's module self-contained). Safe to call only when no
// live consumer still depends on the old def-use registry.
void MLIR_ResetInternRegistry(void) {
    g_all_ops = NULL;      g_n_all_ops = 0;   g_cap_all_ops = 0;
    g_type_handles = NULL; g_n_types = 0;     g_cap_types = 0;
    g_struct_handles = NULL; g_struct_names = NULL;
    g_n_structs = 0;       g_cap_structs = 0;
}

MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *ctx, string name) {
    return intern_llvm_struct(ctx, name);
}

void MLIR_SetTypeLLVMStructBody(MLIR_Context *ctx, MLIR_TypeHandle struct_ty,
                                 const MLIR_TypeHandle *fields, size_t n_fields) {
    IR_Type *t = resolve_type(struct_ty);
    if (!t || t->kind != TYPE_KIND_LLVM_STRUCT) return;
    if (t->data.llvm_struct.body_set) return; // upstream silently ignores re-set
    if (n_fields > 0) {
        MLIR_TypeHandle *buf = arena_new_array(ctx->arena, MLIR_TypeHandle, n_fields);
        memcpy(buf, fields, n_fields * sizeof(MLIR_TypeHandle));
        t->data.llvm_struct.fields = buf;
    }
    t->data.llvm_struct.n_fields = n_fields;
    t->data.llvm_struct.body_set = true;
}

MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *ctx, MLIR_TypeHandle elem, uint64_t count) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_ARRAY;
    t.data.llvm_array.element = elem;
    t.data.llvm_array.count = count;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMFunction(MLIR_Context *ctx,
                                             MLIR_TypeHandle result,
                                             const MLIR_TypeHandle *inputs,
                                             size_t n_inputs,
                                             bool is_var_arg) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_FUNCTION;
    t.data.llvm_function.result = result;
    t.data.llvm_function.is_var_arg = is_var_arg;
    if (n_inputs > 0) {
        MLIR_TypeHandle *buf = arena_new_array(ctx->arena, MLIR_TypeHandle, n_inputs);
        memcpy(buf, inputs, n_inputs * sizeof(MLIR_TypeHandle));
        t.data.llvm_function.inputs = buf;
        t.data.llvm_function.n_inputs = n_inputs;
    }
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMVoid(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_VOID;
    return intern_type(ctx, t);
}

// LLVM-dialect global helpers. Construct `llvm.mlir.global` ops with the
// attributes the LLVM-IR translator looks for (sym_name, global_type,
// linkage, constant, value). Both helpers return the op so the caller
// can append it to the module body. MLIR_CreateLLVMGlobal optionally
// hands back the entry block of the initializer region for region-init
// globals.
static MLIR_AttributeHandle make_llvm_linkage_attr(MLIR_Context *ctx, string linkage_kind) {
    // Mirror the upstream printer: "#llvm.linkage<<kind>>".
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = str_lit("linkage");
    strbuf body = strbuf_make();
    strbuf_append(ctx->arena, &body, str_lit("#llvm.linkage<"));
    strbuf_append(ctx->arena, &body, linkage_kind);
    strbuf_append(ctx->arena, &body, str_lit(">"));
    a.data.string_value = strbuf_to_string(body);
    return alloc_attr_obj(ctx, a);
}

MLIR_OpHandle MLIR_CreateLLVMGlobalString(MLIR_Context *ctx, string sym_name,
                                          string bytes, MLIR_LocationHandle loc) {
    MLIR_TypeHandle i8 = MLIR_CreateTypeInteger(ctx, 8, false);
    MLIR_TypeHandle arr = MLIR_CreateTypeLLVMArray(ctx, i8, bytes.size);
    MLIR_AttributeHandle attrs[5];
    attrs[0] = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), sym_name);
    attrs[1] = MLIR_CreateAttributeType(ctx, str_lit("global_type"), arr);
    attrs[2] = make_llvm_linkage_attr(ctx, str_lit("private"));
    attrs[3] = MLIR_CreateAttributeBool(ctx, str_lit("constant"), true);
    attrs[4] = MLIR_CreateAttributeString(ctx, str_lit("value"), bytes);
    return MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.mlir.global"),
                         attrs, 5, NULL, 0, NULL, 0, NULL, 0,
                         NULL, 0, loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

// Aggregate-array global with a raw-bytes initializer:
//   llvm.mlir.global internal @<sym>("<bytes>") : !llvm.array<N x iX>
// Used for `static const T arr[N] = { v1, v2, ... };` at file scope.
// The `bytes` buffer holds the packed little-endian element bytes
// (size must equal N * sizeof(elem)).
// Backends interpret the STRING `value` attribute as raw initializer
// bytes (see lower_global in mlir_llvm_to_wasmssa.c and emit_global in
// mlir_translate_to_llvm_ir.c).
MLIR_OpHandle MLIR_CreateLLVMGlobalArrayInit(MLIR_Context *ctx, string sym_name,
                                              MLIR_TypeHandle array_ty,
                                              bool is_constant,
                                              string bytes,
                                              MLIR_LocationHandle loc) {
    MLIR_AttributeHandle attrs[5];
    size_t na = 0;
    attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), sym_name);
    attrs[na++] = MLIR_CreateAttributeType(ctx, str_lit("global_type"), array_ty);
    attrs[na++] = make_llvm_linkage_attr(ctx, str_lit("internal"));
    if (is_constant) {
        attrs[na++] = MLIR_CreateAttributeBool(ctx, str_lit("constant"), true);
    }
    attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("value"), bytes);
    return MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.mlir.global"),
                         attrs, na, NULL, 0, NULL, 0, NULL, 0,
                         NULL, 0, loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

MLIR_OpHandle MLIR_CreateLLVMGlobal(MLIR_Context *ctx, string sym_name,
                                    MLIR_TypeHandle elem_ty, bool is_constant,
                                    int init_kind, int64_t init_int, double init_float,
                                    MLIR_BlockHandle *out_init_block,
                                    MLIR_LocationHandle loc) {
    // init_kind: 0=integer, 1=float, 2=region, 3=zero-init/no-init, 4=external
    MLIR_AttributeHandle attrs[6];
    size_t na = 0;
    attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), sym_name);
    attrs[na++] = MLIR_CreateAttributeType(ctx, str_lit("global_type"), elem_ty);
    string lk = (init_kind == 4) ? str_lit("external") : str_lit("internal");
    attrs[na++] = make_llvm_linkage_attr(ctx, lk);
    if (is_constant) attrs[na++] = MLIR_CreateAttributeBool(ctx, str_lit("constant"), true);
    if (init_kind == 0) {
        attrs[na++] = MLIR_CreateAttributeInteger(ctx, str_lit("value"), init_int, elem_ty);
    } else if (init_kind == 1) {
        attrs[na++] = MLIR_CreateAttributeFloat(ctx, str_lit("value"), init_float, elem_ty);
    }

    // Region for region-initialized globals (init_kind == 2): create one
    // empty block so the caller can populate it.
    MLIR_RegionHandle regs[1];
    size_t nr = 0;
    if (init_kind == 2) {
        MLIR_RegionHandle r = MLIR_CreateRegion(ctx);
        MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, r, entry);
        if (out_init_block) *out_init_block = entry;
        regs[0] = r; nr = 1;
    } else if (out_init_block) {
        *out_init_block = MLIR_INVALID_HANDLE;
    }
    return MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.mlir.global"),
                         attrs, na, NULL, 0, NULL, 0, NULL, 0,
                         nr ? regs : NULL, nr, loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *ctx,
                                         const MLIR_TypeHandle *inputs, size_t n_inputs,
                                         const MLIR_TypeHandle *results, size_t n_results) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_FUNCTION;
    if (ctx && ctx->arena) {
        if (n_inputs > 0 && inputs) {
            t.data.function.inputs = arena_new_array(ctx->arena, MLIR_TypeHandle, n_inputs);
            for (size_t i = 0; i < n_inputs; i++) t.data.function.inputs[i] = inputs[i];
            t.data.function.n_inputs = n_inputs;
        }
        if (n_results > 0 && results) {
            t.data.function.results = arena_new_array(ctx->arena, MLIR_TypeHandle, n_results);
            for (size_t i = 0; i < n_results; i++) t.data.function.results[i] = results[i];
            t.data.function.n_results = n_results;
        }
    }
    return intern_type(ctx, t);
}

bool MLIR_IsTypeFunction(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && (t->kind == TYPE_KIND_FUNCTION || t->kind == TYPE_KIND_LLVM_FUNCTION);
}

size_t MLIR_GetTypeFunctionNumInputs(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return 0;
    if (t->kind == TYPE_KIND_FUNCTION) return t->data.function.n_inputs;
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) return t->data.llvm_function.n_inputs;
    return 0;
}

MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_TypeHandle th, size_t idx) {
    IR_Type *t = resolve_type(th);
    if (!t) return MLIR_INVALID_HANDLE;
    if (t->kind == TYPE_KIND_FUNCTION) {
        if (idx >= t->data.function.n_inputs || !t->data.function.inputs) return MLIR_INVALID_HANDLE;
        return t->data.function.inputs[idx];
    }
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) {
        if (idx >= t->data.llvm_function.n_inputs || !t->data.llvm_function.inputs) return MLIR_INVALID_HANDLE;
        return t->data.llvm_function.inputs[idx];
    }
    return MLIR_INVALID_HANDLE;
}

size_t MLIR_GetTypeFunctionNumResults(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return 0;
    if (t->kind == TYPE_KIND_FUNCTION) return t->data.function.n_results;
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) {
        // LLVMFunctionType always has exactly one return type; void is
        // represented as a separate LLVMVoid type. Surface 0 results in
        // the void case so callers can use the same "0 results means void"
        // convention as upstream's MLIR_GetTypeFunctionNumResults.
        IR_Type *r = resolve_type(t->data.llvm_function.result);
        return (r && r->kind == TYPE_KIND_LLVM_VOID) ? 0 : 1;
    }
    return 0;
}

MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_TypeHandle th, size_t idx) {
    IR_Type *t = resolve_type(th);
    if (!t) return MLIR_INVALID_HANDLE;
    if (t->kind == TYPE_KIND_FUNCTION) {
        if (idx >= t->data.function.n_results || !t->data.function.results) return MLIR_INVALID_HANDLE;
        return t->data.function.results[idx];
    }
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) {
        if (idx != 0) return MLIR_INVALID_HANDLE;
        IR_Type *r = resolve_type(t->data.llvm_function.result);
        if (r && r->kind == TYPE_KIND_LLVM_VOID) return MLIR_INVALID_HANDLE;
        return t->data.llvm_function.result;
    }
    return MLIR_INVALID_HANDLE;
}

bool MLIR_GetTypeFunctionIsVarArg(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return false;
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) return t->data.llvm_function.is_var_arg;
    return false;
}

MLIR_TypeHandle MLIR_GetTypeShapedElement(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return MLIR_INVALID_HANDLE;
    if (t->kind == TYPE_KIND_TENSOR || t->kind == TYPE_KIND_MEMREF) {
        return t->data.shaped.element_type;
    }
    return MLIR_INVALID_HANDLE;
}

bool MLIR_IsTypeLLVMStruct(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_LLVM_STRUCT;
}
size_t MLIR_GetTypeLLVMStructNumFields(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_STRUCT) return 0;
    return t->data.llvm_struct.n_fields;
}
MLIR_TypeHandle MLIR_GetTypeLLVMStructField(MLIR_TypeHandle th, size_t idx) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_STRUCT) return MLIR_INVALID_HANDLE;
    if (idx >= t->data.llvm_struct.n_fields) return MLIR_INVALID_HANDLE;
    return t->data.llvm_struct.fields[idx];
}
bool MLIR_IsTypeLLVMArray(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_LLVM_ARRAY;
}
MLIR_TypeHandle MLIR_GetTypeLLVMArrayElement(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_ARRAY) return MLIR_INVALID_HANDLE;
    return t->data.llvm_array.element;
}
uint64_t MLIR_GetTypeLLVMArrayNumElements(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_ARRAY) return 0;
    return t->data.llvm_array.count;
}

void MLIR_SetTypeIntegerProperties(MLIR_TypeHandle th, uint32_t width, bool is_signed) {
    // `is_signed` is intentionally ignored — see MLIR_CreateTypeInteger.
    (void)is_signed;
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_INTEGER;
    t->data.integer.width = width;
    t->data.integer.is_signed = false;
}

void MLIR_SetTypeFloatProperties(MLIR_TypeHandle th, uint32_t width, bool is_bfloat) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_FLOAT;
    t->data.floating.width = width;
    t->data.floating.is_bfloat = is_bfloat;
}

// Attribute creation
MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *ctx, string name, int64_t value, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_INTEGER;
    a.name = name;
    a.data.integer_value = value;
    a.type = type;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *ctx, string name, string value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = arena_dup_str(ctx, name);
    a.data.string_value = arena_dup_str(ctx, value);
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *ctx, string name, double value, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_FLOAT;
    a.name = name;
    a.data.float_value = value;
    a.type = type;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *ctx, string name, bool value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_BOOL;
    a.name = name;
    a.data.bool_value = value;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_ARRAY;
    a.name = name;
    a.data.array.elements = elements;
    a.data.array.count = count;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_DICT;
    a.name = name;
    a.data.array.elements = elements;
    a.data.array.count = count;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *ctx, string name, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_TYPE;
    a.name = name;
    a.data.type_value = type;
    return alloc_attr_obj(ctx, a);
}

// Native backend stores SymbolRef as a plain string attribute. The
// upstream printer renders symbol references as `@name`, so we store
// the same form here — keeps MLIR_GetAttributeAsString consistent
// across backends (the LLVM-IR translator depends on this for callee
// emission).
MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *ctx, string name, string value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = name;
    string with_at = format(ctx->arena, str_lit("@{}"), value);
    a.data.string_value = with_at;
    return alloc_attr_obj(ctx, a);
}

// DenseI32 array attribute. The native LLVM-IR translator reads this
// via MLIR_GetAttributeAsString and parses the printed form, so we
// stash a string matching upstream's printer ("array<i32: v0, v1, ...>").
MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *ctx, string name,
                                                       const int32_t *values, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = name;
    Arena *arena = ctx->arena;
    strbuf s = strbuf_make();
    strbuf_append(arena, &s, str_lit("array<i32"));
    for (size_t i = 0; i < count; i++) {
        strbuf_append(arena, &s, i == 0 ? str_lit(": ") : str_lit(", "));
        strbuf_append(arena, &s, format(arena, str_lit("{}"), (int64_t)values[i]));
    }
    strbuf_append(arena, &s, str_lit(">"));
    a.data.string_value = strbuf_to_string(s);
    return alloc_attr_obj(ctx, a);
}

// DenseI64 array attribute. Same string-encoded shape as DenseI32 but
// with `i64` element type. Used by `scf.index_switch`'s `cases` attr.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI64Array(MLIR_Context *ctx, string name,
                                                       const int64_t *values, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = name;
    Arena *arena = ctx->arena;
    strbuf s = strbuf_make();
    strbuf_append(arena, &s, str_lit("array<i64"));
    for (size_t i = 0; i < count; i++) {
        strbuf_append(arena, &s, i == 0 ? str_lit(": ") : str_lit(", "));
        strbuf_append(arena, &s, format(arena, str_lit("{}"), values[i]));
    }
    strbuf_append(arena, &s, str_lit(">"));
    a.data.string_value = strbuf_to_string(s);
    return alloc_attr_obj(ctx, a);
}

// LLVM linkage attribute. Native backend stores the printable form
// (e.g. "#llvm.linkage<internal>") in a string attr — this matches the
// upstream backend's MLIR_GetAttributeAsString result, so the native
// LLVM-IR translator can recognize it the same way for both backends.
MLIR_AttributeHandle MLIR_CreateAttributeLLVMLinkageInternal(MLIR_Context *ctx, string name) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = name;
    a.data.string_value = str_lit("#llvm.linkage<internal>");
    return alloc_attr_obj(ctx, a);
}

// Value creation
MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *ctx, string register_name, uint32_t result_index, MLIR_TypeHandle type, MLIR_LocationHandle location) {
    IR_Value v = {0};
    v.kind = BLOCK_ARG;
    v.aux = value_aux_create(ctx, register_name);
    v.result_index = result_index;
    v.type = type;
    v.location = location;
    v.def_handle = MLIR_INVALID_HANDLE;
    return alloc_value(ctx, v);
}

MLIR_ValueHandle MLIR_CreateValueOpResult(MLIR_Context *ctx, MLIR_OpHandle def, uint32_t result_index, MLIR_TypeHandle type, string register_name, MLIR_LocationHandle location) {
    IR_Value v = {0};
    v.kind = OP_RESULT;
    v.def_handle = def;
    v.result_index = result_index;
    v.type = type;
    v.aux = value_aux_create(ctx, register_name);
    v.location = location;
    return alloc_value(ctx, v);
}

// Block and Region creation
MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *ctx) {
    IR_Block b = {0};
    return alloc_block(ctx, b);
}

MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *ctx) {
    IR_Region r = {0};
    return alloc_region(ctx, r);
}

void MLIR_AppendOpAttribute(MLIR_Context *ctx, MLIR_OpHandle oh, MLIR_AttributeHandle attr) {
    IR_Op *op = resolve_op(oh);
    if (!op || !ctx || !ctx->arena) return;
    size_t new_count = op->n_attributes + 1;
    Arena *arena = ctx->arena;
    MLIR_AttributeHandle *new_attrs = arena_new_array(arena, MLIR_AttributeHandle, new_count);
    if (op->attributes) {
        memcpy(new_attrs, op->attributes, op->n_attributes * sizeof(MLIR_AttributeHandle));
    }
    new_attrs[new_count - 1] = attr;
    op->attributes = new_attrs;
    op->n_attributes = (uint32_t)new_count;
}

int64_t MLIR_GetOpSourceLineStart(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return (op && op->aux) ? op->aux->source_line_start : -1;
}

MLIR_LocationHandle MLIR_GetOpUnnumberedLocationDef(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return (op && op->aux) ? op->aux->unnumbered_loc_def : MLIR_INVALID_HANDLE;
}

// Attribute accessors
MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return MLIR_ATTR_KIND_INTEGER;
    switch (attr->kind) {
        case ATTR_KIND_INTEGER: return MLIR_ATTR_KIND_INTEGER;
        case ATTR_KIND_FLOAT:   return MLIR_ATTR_KIND_FLOAT;
        case ATTR_KIND_STRING:  return MLIR_ATTR_KIND_STRING;
        case ATTR_KIND_BOOL:    return MLIR_ATTR_KIND_BOOL;
        case ATTR_KIND_ARRAY:   return MLIR_ATTR_KIND_ARRAY;
        case ATTR_KIND_DICT:    return MLIR_ATTR_KIND_DICT;
        case ATTR_KIND_TYPE:    return MLIR_ATTR_KIND_TYPE;
        default:                return MLIR_ATTR_KIND_DICT;
    }
}

string MLIR_GetAttributeName(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->name : str_lit("");
}

// Native parser never produces "other" attrs; this is just for API parity.
string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    (void)ctx;
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return str_lit("");
    // The native lowering/translation passes only consult this for a few
    // attributes whose printed form is well-defined (LLVM linkage, plain
    // strings). For everything else we return empty — callers must use
    // the typed accessors.
    if (attr->kind == ATTR_KIND_STRING) return attr->data.string_value;
    return str_lit("");
}

int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.integer_value : 0;
}

string MLIR_GetAttributeString(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.string_value : str_lit("");
}

double MLIR_GetAttributeFloat(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.float_value : 0.0;
}

MLIR_TypeHandle MLIR_GetAttributeType(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return MLIR_INVALID_HANDLE;
    if (attr->kind != ATTR_KIND_INTEGER && attr->kind != ATTR_KIND_FLOAT)
        return MLIR_INVALID_HANDLE;
    return attr->type;
}

bool MLIR_GetAttributeBool(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.bool_value : false;
}

size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.array.count : 0;
}

MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle ah, size_t idx) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || idx >= attr->data.array.count) return MLIR_INVALID_HANDLE;
    return attr->data.array.elements[idx];
}

size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.array.count : 0;
}

MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle ah, size_t idx) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || idx >= attr->data.array.count) return MLIR_INVALID_HANDLE;
    return attr->data.array.elements[idx];
}

MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || attr->kind != ATTR_KIND_TYPE) return MLIR_INVALID_HANDLE;
    return attr->data.type_value;
}

// Value accessors
MLIR_ValueKind MLIR_GetValueKind(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->kind : BLOCK_ARG;
}

MLIR_TypeHandle MLIR_GetValueType(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->type : MLIR_INVALID_HANDLE;
}

string MLIR_GetValueRegisterName(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return (v && v->aux) ? v->aux->register_name : str_lit("");
}

uint32_t MLIR_GetValueResultIndex(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->result_index : 0;
}

MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    if (!v || v->kind != OP_RESULT) return MLIR_INVALID_HANDLE;
    return (MLIR_OpHandle)v->def_handle;
}

string MLIR_MLIR_OpTypeToString(MLIR_OpType type) {
    return op_type_to_string(type);
}

// Location accessors
MLIR_LocationKind MLIR_GetLocationKind(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    if (!loc) return MLIR_LOC_UNKNOWN;
    switch (loc->kind) {
        case LOC_KIND_FILE: return MLIR_LOC_FILE;
        case LOC_KIND_NAME: return MLIR_LOC_NAME;
        case LOC_KIND_CALLSITE: return MLIR_LOC_CALLSITE;
        case LOC_KIND_FUSED: return MLIR_LOC_FUSED;
        case LOC_KIND_REF: return MLIR_LOC_REF;
        case LOC_KIND_UNKNOWN: default: return MLIR_LOC_UNKNOWN;
    }
}

string MLIR_GetLocationOriginalText(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->original_text : str_lit("");
}

string MLIR_GetLocationFileFilename(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.filename : str_lit("");
}

int MLIR_GetLocationFileLine(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.line : 0;
}

int MLIR_GetLocationFileColumn(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.column : 0;
}

string MLIR_GetLocationName(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.name.name : str_lit("");
}

int MLIR_GetLocationRefId(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.ref.ref_id : 0;
}

MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *ctx, string original_text) {
    IR_Location l = {0};
    l.kind = LOC_KIND_UNKNOWN;
    l.original_text = original_text;
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *ctx, string filename, int line, int column) {
    IR_Location l = {0};
    l.kind = LOC_KIND_FILE;
    l.data.file.filename = filename;
    l.data.file.line = line;
    l.data.file.column = column;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc({}:{}:{})"), filename, (int64_t)line, (int64_t)column);
    }
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *ctx, string name) {
    IR_Location l = {0};
    l.kind = LOC_KIND_NAME;
    l.data.name.name = name;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc(\"{}\")"), name);
    }
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *ctx, int ref_id) {
    IR_Location l = {0};
    l.kind = LOC_KIND_REF;
    l.data.ref.ref_id = ref_id;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc(#loc{})"), (int64_t)ref_id);
    }
    return alloc_loc(ctx, l);
}

MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name) {
    (void)name;
    IR_Type t = {0};
    t.kind = TYPE_KIND_OPAQUE;
    return intern_type(ctx, t);
}

void MLIR_SetTypeTensorProperties(MLIR_TypeHandle th, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_TENSOR;
    t->data.shaped.element_type = element_type;
    t->data.shaped.shape = (int64_t*)shape;
    t->data.shaped.rank = (uint32_t)rank;
}

void MLIR_SetTypeMemrefProperties(MLIR_TypeHandle th, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_MEMREF;
    t->data.shaped.element_type = element_type;
    t->data.shaped.shape = (int64_t*)shape;
    t->data.shaped.rank = (uint32_t)rank;
}

void MLIR_SetTypePointerProperties(MLIR_TypeHandle th, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_POINTER;
    t->data.pointer.element_type = element_type;
    t->data.pointer.has_address_space = has_address_space;
    t->data.pointer.address_space = address_space;
}

// Type introspection
bool MLIR_IsTypeInteger(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_INTEGER;
}

bool MLIR_IsTypeFloat(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_FLOAT;
}

bool MLIR_IsTypeTensor(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_TENSOR;
}

bool MLIR_IsTypeMemref(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_MEMREF;
}

bool MLIR_IsTypePointer(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_POINTER;
}

bool MLIR_IsTypeIndex(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_INDEX;
}

bool MLIR_IsTypeUnknown(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_UNKNOWN;
}

bool MLIR_IsTypeOpaque(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_OPAQUE;
}

// IR mutation primitives — native implementations used by the Stage B/C
// lowering+translation passes (mlir_lower_to_llvm.c and
// mlir_translate_to_llvm_ir.c). They mirror the upstream-backend behaviour
// just well enough for the tinyc test suite.

void MLIR_ReplaceAllUsesOfValue(MLIR_Context *ctx,
                                MLIR_ValueHandle old_value,
                                MLIR_ValueHandle new_value) {
    (void)ctx;
    if (old_value == MLIR_INVALID_HANDLE || old_value == new_value) return;
    IR_Value *ov = resolve_value(old_value);
    if (!ov || !ov->aux || !ov->aux->use_head) return;
    IR_Value *nv = (new_value == MLIR_INVALID_HANDLE) ? NULL : resolve_value(new_value);
    // Walk the use list, repointing each operand slot to new_value.
    UseNode *u = ov->aux->use_head;
    while (u) {
        UseNode *next = u->next;
        IR_Op *o = resolve_op(u->owner);
        if (o) {
            if (u->kind == 0 && u->op_idx < o->n_operands) {
                o->operands[u->op_idx] = new_value;
            } else if (u->kind == 1 && u->succ_idx < o->n_successors &&
                       o->successor_operands &&
                       o->successor_operands[u->succ_idx] &&
                       u->op_idx < o->n_successor_operands[u->succ_idx]) {
                o->successor_operands[u->succ_idx][u->op_idx] = new_value;
            }
        }
        u = next;
    }
    // Splice the whole list from ov onto nv (or just drop it if nv==NULL).
    if (nv && nv->aux) {
        UseNode *head = ov->aux->use_head;
        // Append ov's chain to nv's chain — find ov's tail, then link.
        UseNode *tail = head;
        while (tail->next) tail = tail->next;
        tail->next = nv->aux->use_head;
        if (nv->aux->use_head) nv->aux->use_head->prev = tail;
        nv->aux->use_head = head;
        nv->aux->n_uses += ov->aux->n_uses;
    } else {
        // Drop chain; UseNodes leak in the arena but that's fine (arena
        // is freed at context end).
        for (UseNode *x = ov->aux->use_head; x; x = x->next) {
            x->prev = x->next = NULL;
        }
    }
    ov->aux->use_head = NULL;
    ov->aux->n_uses = 0;
}

void MLIR_EraseOp(MLIR_Context *ctx, MLIR_OpHandle op) {
    (void)ctx;
    IR_Op *o = resolve_op(op);
    if (!o) return;
    // Tear down def-use entries so callers no longer see this op in any
    // value's use list. Leave the IR_Op shell in place — other code may
    // still hold the handle and call MLIR_GetOpParentBlock on it.
    if (o->aux && o->aux->operand_uses) {
        for (uint64_t i = 0; i < o->n_operands; i++) {
            use_unregister(o->operands[i], o->aux->operand_uses[i]);
            o->aux->operand_uses[i] = NULL;
        }
    }
    if (o->aux && o->aux->successor_operand_uses) {
        for (uint64_t s = 0; s < o->n_successors; s++) {
            if (!o->aux->successor_operand_uses[s].arr) continue;
            uint64_t cnt = o->n_successor_operands ? o->n_successor_operands[s] : 0;
            for (uint64_t k = 0; k < cnt; k++) {
                MLIR_ValueHandle v = o->successor_operands && o->successor_operands[s]
                    ? o->successor_operands[s][k] : MLIR_INVALID_HANDLE;
                use_unregister(v, o->aux->successor_operand_uses[s].arr[k]);
                o->aux->successor_operand_uses[s].arr[k] = NULL;
            }
        }
    }
    IR_Block *b = resolve_block(o->parent_block);
    if (b) {
        size_t w = 0;
        for (size_t i = 0; i < b->n_operations; i++) {
            if (b->operations[i] != op) b->operations[w++] = b->operations[i];
        }
        b->n_operations = (uint32_t)w;
    }
    o->parent_block = MLIR_INVALID_HANDLE;
}

void MLIR_SetOpRegion(MLIR_Context *ctx, MLIR_OpHandle op, size_t idx,
                      MLIR_RegionHandle region) {
    (void)ctx;
    IR_Op *o = resolve_op(op);
    if (!o || idx >= o->n_regions) return;
    o->regions[idx] = region;
}

MLIR_RegionHandle MLIR_TakeOpRegion(MLIR_Context *ctx, MLIR_OpHandle op,
                                    size_t idx) {
    IR_Op *o = resolve_op(op);
    if (!o || idx >= o->n_regions) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle taken = o->regions[idx];
    // Replace with a fresh empty region and reparent any blocks that
    // referenced the old region.
    IR_Region empty = {0};
    o->regions[idx] = alloc_region(ctx, empty);
    return taken;
}

MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_OpHandle op) {
    IR_Op *o = resolve_op(op);
    return o ? o->parent_block : MLIR_INVALID_HANDLE;
}

size_t MLIR_GetBlockOpIndex(MLIR_BlockHandle block, MLIR_OpHandle op) {
    IR_Block *b = resolve_block(block);
    if (!b) return (size_t)-1;
    for (size_t i = 0; i < b->n_operations; i++) {
        if (b->operations[i] == op) return i;
    }
    return (size_t)-1;
}

MLIR_RegionHandle MLIR_GetBlockParentRegion(MLIR_BlockHandle block) {
    IR_Block *b = resolve_block(block);
    return b ? b->parent_region : MLIR_INVALID_HANDLE;
}

void MLIR_MoveOpToBlockEnd(MLIR_Context *ctx, MLIR_OpHandle op,
                           MLIR_BlockHandle dest) {
    if (op == MLIR_INVALID_HANDLE || dest == MLIR_INVALID_HANDLE) return;
    IR_Op *o = resolve_op(op);
    if (!o) return;
    // Detach from current parent.
    IR_Block *cur = resolve_block(o->parent_block);
    if (cur) {
        size_t w = 0;
        for (size_t i = 0; i < cur->n_operations; i++) {
            if (cur->operations[i] != op) cur->operations[w++] = cur->operations[i];
        }
        cur->n_operations = (uint32_t)w;
    }
    o->parent_block = MLIR_INVALID_HANDLE;
    MLIR_AppendBlockOp(ctx, dest, op);
}

void MLIR_MoveBlockToRegionEnd(MLIR_Context *ctx, MLIR_BlockHandle block,
                               MLIR_RegionHandle dest) {
    if (block == MLIR_INVALID_HANDLE || dest == MLIR_INVALID_HANDLE) return;
    IR_Block *b = resolve_block(block);
    if (!b) return;
    IR_Region *cur = resolve_region(b->parent_region);
    if (cur) {
        size_t w = 0;
        for (size_t i = 0; i < cur->n_blocks; i++) {
            if (cur->blocks[i] != block) cur->blocks[w++] = cur->blocks[i];
        }
        cur->n_blocks = w;
    }
    b->parent_region = MLIR_INVALID_HANDLE;
    MLIR_AppendRegionBlock(ctx, dest, block);
}

void MLIR_SetOpOperand(MLIR_Context *ctx, MLIR_OpHandle op,
                       size_t idx, MLIR_ValueHandle value) {
    (void)ctx;
    IR_Op *o = resolve_op(op);
    if (!o) return;
    // The contract (matching upstream's MLIR_SetOpOperand on MLIR core,
    // where `op->setOperand(idx, ...)` indexes the *flat* operand range)
    // is that `idx` indexes into the combined sequence:
    //   [0 .. n_operands)               -> regular operands
    //   [n_operands ..)                 -> successor operands, segmented
    //                                       by successor in declaration
    //                                       order.
    // `MLIR_GetValueUseOwner` returns these same synthetic flat indices
    // for kind=1 (successor-operand) uses; callers like `replace_use`
    // in the cf->scf lift round-trip a use back through SetOpOperand
    // expecting it to update either segment.
    if (idx < o->n_operands) {
        MLIR_ValueHandle old = o->operands[idx];
        if (old == value) return;
        if (o->aux && o->aux->operand_uses) {
            use_unregister(old, o->aux->operand_uses[idx]);
            o->aux->operand_uses[idx] = NULL;
        }
        o->operands[idx] = value;
        if (o->aux && o->aux->operand_uses) {
            o->aux->operand_uses[idx] = use_register_operand(ctx, op, value, (uint32_t)idx);
        }
        return;
    }
    // Map the post-operand index into the successor-operand segments.
    size_t rem = idx - o->n_operands;
    for (uint64_t s = 0; s < o->n_successors; ++s) {
        uint64_t cnt = o->n_successor_operands ? o->n_successor_operands[s] : 0;
        if (rem < cnt) {
            MLIR_ValueHandle *seg = o->successor_operands
                ? o->successor_operands[s] : NULL;
            if (!seg) return;
            MLIR_ValueHandle old = seg[rem];
            if (old == value) return;
            if (o->aux && o->aux->successor_operand_uses && o->aux->successor_operand_uses[s].arr) {
                use_unregister(old, o->aux->successor_operand_uses[s].arr[rem]);
                o->aux->successor_operand_uses[s].arr[rem] = NULL;
            }
            seg[rem] = value;
            if (o->aux && o->aux->successor_operand_uses && o->aux->successor_operand_uses[s].arr) {
                o->aux->successor_operand_uses[s].arr[rem] = use_register_succ_operand(
                    ctx, op, value, (uint32_t)s, (uint32_t)rem);
            }
            return;
        }
        rem -= cnt;
    }
    // idx out of range: silently no-op (matches the original guard).
}

void MLIR_SetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle op,
                         size_t succ_idx, MLIR_BlockHandle block) {
    (void)ctx;
    IR_Op *o = resolve_op(op);
    if (!o || succ_idx >= o->n_successors) return;
    o->successors[succ_idx] = block;
}

void MLIR_SetOpSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle op,
                                 size_t succ_idx,
                                 const MLIR_ValueHandle *values, size_t n) {
    IR_Op *o = resolve_op(op);
    if (!o || succ_idx >= o->n_successors) return;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    if (!o->successor_operands) {
        o->successor_operands = arena_new_array(arena, MLIR_ValueHandle *, o->n_successors);
        for (size_t s = 0; s < o->n_successors; s++) o->successor_operands[s] = NULL;
    }
    if (!o->n_successor_operands) {
        o->n_successor_operands = arena_new_array(arena, uint64_t, o->n_successors);
        for (size_t s = 0; s < o->n_successors; s++) o->n_successor_operands[s] = 0;
    }
    if (!o->aux) o->aux = op_aux_default(arena);
    if (!o->aux->successor_operand_uses) {
        o->aux->successor_operand_uses = arena_new_array(arena, UseSlot, o->n_successors);
        for (size_t s = 0; s < o->n_successors; s++) o->aux->successor_operand_uses[s].arr = NULL;
    }
    // Unregister existing uses in this segment before replacing.
    uint64_t old_n = o->n_successor_operands[succ_idx];
    for (uint64_t k = 0; k < old_n; k++) {
        MLIR_ValueHandle v = o->successor_operands[succ_idx]
            ? o->successor_operands[succ_idx][k] : MLIR_INVALID_HANDLE;
        UseNode *u = o->aux->successor_operand_uses[succ_idx].arr
            ? o->aux->successor_operand_uses[succ_idx].arr[k] : NULL;
        use_unregister(v, u);
    }
    if (n > 0) {
        MLIR_ValueHandle *arr = arena_new_array(arena, MLIR_ValueHandle, n);
        if (values) memcpy(arr, values, n * sizeof(MLIR_ValueHandle));
        o->successor_operands[succ_idx] = arr;
        o->aux->successor_operand_uses[succ_idx].arr = arena_new_array(arena, UseNode *, n);
        for (size_t k = 0; k < n; k++) {
            o->aux->successor_operand_uses[succ_idx].arr[k] = use_register_succ_operand(
                ctx, op, arr[k], (uint32_t)succ_idx, (uint32_t)k);
        }
    } else {
        o->successor_operands[succ_idx] = NULL;
        o->aux->successor_operand_uses[succ_idx].arr = NULL;
    }
    o->n_successor_operands[succ_idx] = (uint64_t)n;
}

void MLIR_EraseBlock(MLIR_Context *ctx, MLIR_BlockHandle block) {
    (void)ctx;
    IR_Block *b = resolve_block(block);
    if (!b) return;
    IR_Region *r = resolve_region(b->parent_region);
    if (r) {
        size_t w = 0;
        for (size_t i = 0; i < r->n_blocks; i++) {
            if (r->blocks[i] != block) r->blocks[w++] = r->blocks[i];
        }
        r->n_blocks = w;
    }
    b->parent_region = MLIR_INVALID_HANDLE;
}

void MLIR_InsertRegionBlockAfter(MLIR_Context *ctx, MLIR_RegionHandle region,
                                 MLIR_BlockHandle block, MLIR_BlockHandle after) {
    if (region == MLIR_INVALID_HANDLE || block == MLIR_INVALID_HANDLE) return;
    IR_Block *b = resolve_block(block);
    if (!b) return;
    // Detach from current region if any.
    IR_Region *cur = resolve_region(b->parent_region);
    if (cur) {
        size_t w = 0;
        for (size_t i = 0; i < cur->n_blocks; i++) {
            if (cur->blocks[i] != block) cur->blocks[w++] = cur->blocks[i];
        }
        cur->n_blocks = w;
    }
    b->parent_region = MLIR_INVALID_HANDLE;

    IR_Region *r = resolve_region(region);
    if (!r) return;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    // Find insertion index. after == MLIR_INVALID_HANDLE means insert at front.
    size_t insert_at = 0;
    if (after != MLIR_INVALID_HANDLE) {
        insert_at = r->n_blocks; // default: append if `after` not found
        for (size_t i = 0; i < r->n_blocks; i++) {
            if (r->blocks[i] == after) { insert_at = i + 1; break; }
        }
    }
    if (r->n_blocks >= r->cap_blocks) {
        uint64_t new_cap = r->cap_blocks ? r->cap_blocks * 2 : 4;
        MLIR_BlockHandle *nb = arena_new_array(arena, MLIR_BlockHandle, new_cap);
        if (insert_at > 0) memcpy(nb, r->blocks, insert_at * sizeof(MLIR_BlockHandle));
        if (insert_at < r->n_blocks)
            memcpy(nb + insert_at + 1, r->blocks + insert_at,
                   (r->n_blocks - insert_at) * sizeof(MLIR_BlockHandle));
        r->blocks = nb;
        r->cap_blocks = new_cap;
    } else {
        if (insert_at < r->n_blocks)
            memmove(r->blocks + insert_at + 1, r->blocks + insert_at,
                    (r->n_blocks - insert_at) * sizeof(MLIR_BlockHandle));
    }
    r->blocks[insert_at] = block;
    r->n_blocks++;
    b->parent_region = region;
}

void MLIR_InsertRegionBlockBefore(MLIR_Context *ctx, MLIR_RegionHandle region,
                                  MLIR_BlockHandle block, MLIR_BlockHandle before) {
    if (region == MLIR_INVALID_HANDLE || block == MLIR_INVALID_HANDLE) return;
    IR_Block *b = resolve_block(block);
    if (!b) return;
    IR_Region *cur = resolve_region(b->parent_region);
    if (cur) {
        size_t w = 0;
        for (size_t i = 0; i < cur->n_blocks; i++) {
            if (cur->blocks[i] != block) cur->blocks[w++] = cur->blocks[i];
        }
        cur->n_blocks = w;
    }
    b->parent_region = MLIR_INVALID_HANDLE;

    IR_Region *r = resolve_region(region);
    if (!r) return;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    size_t insert_at = r->n_blocks; // append if not found / invalid
    if (before != MLIR_INVALID_HANDLE) {
        for (size_t i = 0; i < r->n_blocks; i++) {
            if (r->blocks[i] == before) { insert_at = i; break; }
        }
    }
    if (r->n_blocks >= r->cap_blocks) {
        uint64_t new_cap = r->cap_blocks ? r->cap_blocks * 2 : 4;
        MLIR_BlockHandle *nb = arena_new_array(arena, MLIR_BlockHandle, new_cap);
        if (insert_at > 0) memcpy(nb, r->blocks, insert_at * sizeof(MLIR_BlockHandle));
        if (insert_at < r->n_blocks)
            memcpy(nb + insert_at + 1, r->blocks + insert_at,
                   (r->n_blocks - insert_at) * sizeof(MLIR_BlockHandle));
        r->blocks = nb;
        r->cap_blocks = new_cap;
    } else {
        if (insert_at < r->n_blocks)
            memmove(r->blocks + insert_at + 1, r->blocks + insert_at,
                    (r->n_blocks - insert_at) * sizeof(MLIR_BlockHandle));
    }
    r->blocks[insert_at] = block;
    r->n_blocks++;
    b->parent_region = region;
}

MLIR_OpHandle MLIR_GetBlockTerminator(MLIR_BlockHandle block) {
    IR_Block *b = resolve_block(block);
    if (!b || b->n_operations == 0) return MLIR_INVALID_HANDLE;
    return b->operations[b->n_operations - 1];
}

MLIR_OpHandle MLIR_GetBlockParentOp(MLIR_BlockHandle block) {
    IR_Block *b = resolve_block(block);
    if (!b) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle r = b->parent_region;
    if (r == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    // Scan global op list to find which op owns this region.
    for (size_t i = 0; i < g_n_all_ops; i++) {
        IR_Op *op = resolve_op(g_all_ops[i]);
        if (!op) continue;
        for (size_t j = 0; j < op->n_regions; j++) {
            if (op->regions[j] == r) return g_all_ops[i];
        }
    }
    return MLIR_INVALID_HANDLE;
}

bool MLIR_BlockIsEntry(MLIR_BlockHandle block) {
    IR_Block *b = resolve_block(block);
    if (!b) return false;
    IR_Region *r = resolve_region(b->parent_region);
    if (!r || r->n_blocks == 0) return false;
    return r->blocks[0] == block;
}

// Helper: count and enumerate predecessors of `block`. Walks all blocks in
// the parent region; each terminator successor slot equal to `block` is one
// predecessor entry. The (predecessor block, successor slot) pair is unique.
static size_t native_collect_preds(MLIR_BlockHandle block,
                                    MLIR_BlockHandle *out_preds,
                                    size_t *out_succ_idxs,
                                    size_t cap) {
    IR_Block *b = resolve_block(block);
    if (!b) return 0;
    IR_Region *r = resolve_region(b->parent_region);
    if (!r) return 0;
    size_t n = 0;
    for (size_t i = 0; i < r->n_blocks; i++) {
        IR_Block *pb = resolve_block(r->blocks[i]);
        if (!pb || pb->n_operations == 0) continue;
        IR_Op *term = resolve_op(pb->operations[pb->n_operations - 1]);
        if (!term) continue;
        for (size_t s = 0; s < term->n_successors; s++) {
            if (term->successors[s] == block) {
                if (out_preds && n < cap) out_preds[n] = r->blocks[i];
                if (out_succ_idxs && n < cap) out_succ_idxs[n] = s;
                n++;
            }
        }
    }
    return n;
}

size_t MLIR_GetBlockNumPredecessors(MLIR_BlockHandle block) {
    return native_collect_preds(block, NULL, NULL, 0);
}

MLIR_BlockHandle MLIR_GetBlockPredecessor(MLIR_BlockHandle block, size_t idx,
                                          size_t *out_succ_idx) {
    IR_Block *b = resolve_block(block);
    if (!b) return MLIR_INVALID_HANDLE;
    IR_Region *r = resolve_region(b->parent_region);
    if (!r) return MLIR_INVALID_HANDLE;
    size_t n = 0;
    for (size_t i = 0; i < r->n_blocks; i++) {
        IR_Block *pb = resolve_block(r->blocks[i]);
        if (!pb || pb->n_operations == 0) continue;
        IR_Op *term = resolve_op(pb->operations[pb->n_operations - 1]);
        if (!term) continue;
        for (size_t s = 0; s < term->n_successors; s++) {
            if (term->successors[s] == block) {
                if (n == idx) {
                    if (out_succ_idx) *out_succ_idx = s;
                    return r->blocks[i];
                }
                n++;
            }
        }
    }
    if (out_succ_idx) *out_succ_idx = SIZE_MAX;
    return MLIR_INVALID_HANDLE;
}

MLIR_ValueHandle MLIR_AddBlockArgument(MLIR_Context *ctx, MLIR_BlockHandle block,
                                       MLIR_TypeHandle type,
                                       MLIR_LocationHandle loc) {
    IR_Block *b = resolve_block(block);
    if (!b) return MLIR_INVALID_HANDLE;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    // Create the BlockArg value.
    IR_Value *v = arena_new(arena, IR_Value);
    *v = (IR_Value){0};
    v->kind = BLOCK_ARG;
    v->def_handle = (uintptr_t)block;
    v->result_index = (uint32_t)b->n_arguments;
    v->type = type;
    v->aux = value_aux_create(ctx, (string){0});
    v->location = loc;
    MLIR_ValueHandle vh = (MLIR_ValueHandle)(uintptr_t)v;
    // Append to block->arguments, growing geometrically so repeated
    // AddBlockArgument calls (e.g. when lift-cf-to-scf builds an
    // EdgeMultiplexer with many entry blocks each contributing their
    // args) don't cost O(N^2) memory in the arena.
    if (b->n_arguments >= b->cap_arguments) {
        uint64_t new_cap = b->cap_arguments ? b->cap_arguments * 2 : 4;
        MLIR_ValueHandle *na = arena_new_array(arena, MLIR_ValueHandle, new_cap);
        if (b->n_arguments > 0)
            memcpy(na, b->arguments, b->n_arguments * sizeof(MLIR_ValueHandle));
        b->arguments = na;
        b->cap_arguments = (uint32_t)new_cap;
    }
    b->arguments[b->n_arguments] = vh;
    b->n_arguments++;
    return vh;
}

void MLIR_EraseBlockArguments(MLIR_Context *ctx, MLIR_BlockHandle block,
                              size_t start, size_t count) {
    (void)ctx;
    IR_Block *b = resolve_block(block);
    if (!b || count == 0) return;
    if (start >= b->n_arguments) return;
    if (start + count > b->n_arguments) count = b->n_arguments - start;
    // Shift remaining arguments down.
    for (size_t i = start; i + count < b->n_arguments; i++)
        b->arguments[i] = b->arguments[i + count];
    b->n_arguments -= (uint32_t)count;
    // Re-number result_index of remaining args.
    for (size_t i = start; i < b->n_arguments; i++) {
        IR_Value *v = resolve_value(b->arguments[i]);
        if (v) v->result_index = (uint32_t)i;
    }
}

MLIR_BlockHandle MLIR_GetValueParentBlock(MLIR_ValueHandle value) {
    IR_Value *v = resolve_value(value);
    if (!v) return MLIR_INVALID_HANDLE;
    if (v->kind == BLOCK_ARG)
        return (MLIR_BlockHandle)v->def_handle;
    if (v->kind == OP_RESULT) {
        IR_Op *op = resolve_op((MLIR_OpHandle)v->def_handle);
        if (!op) return MLIR_INVALID_HANDLE;
        return op->parent_block;
    }
    return MLIR_INVALID_HANDLE;
}

size_t MLIR_GetValueNumUses(MLIR_Context *ctx, MLIR_ValueHandle value) {
    (void)ctx;
    if (value == MLIR_INVALID_HANDLE) return 0;
    IR_Value *v = resolve_value(value);
    return (v && v->aux) ? v->aux->n_uses : 0;
}

MLIR_OpHandle MLIR_GetValueUseOwner(MLIR_Context *ctx, MLIR_ValueHandle value,
                                    size_t idx, size_t *out_operand_idx) {
    (void)ctx;
    if (out_operand_idx) *out_operand_idx = SIZE_MAX;
    if (value == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    IR_Value *v = resolve_value(value);
    if (!v) return MLIR_INVALID_HANDLE;
    // Walk the use list to position idx. Use list order is "most recently
    // linked first" because use_list_push_front prepends. Callers don't
    // depend on a specific iteration order (snapshot_uses copies into an
    // owned array, then iterates regardless).
    UseNode *u = v->aux ? v->aux->use_head : NULL;
    size_t n = 0;
    while (u && n < idx) { u = u->next; n++; }
    if (!u) return MLIR_INVALID_HANDLE;
    IR_Op *o = resolve_op(u->owner);
    if (!o) return MLIR_INVALID_HANDLE;
    if (out_operand_idx) {
        if (u->kind == 0) {
            *out_operand_idx = u->op_idx;
        } else {
            // Successor-operand uses report a synthetic operand_idx of
            // n_operands + sum-of-previous-segments + k, matching the
            // unified-operand view exposed by the upstream backend.
            size_t off = o->n_operands;
            for (uint16_t s = 0; s < u->succ_idx && s < o->n_successors; s++) {
                off += o->n_successor_operands ? o->n_successor_operands[s] : 0;
            }
            *out_operand_idx = off + u->op_idx;
        }
    }
    return u->owner;
}

void MLIR_SetOpOperands(MLIR_Context *ctx, MLIR_OpHandle op,
                        const MLIR_ValueHandle *values, size_t n) {
    IR_Op *o = resolve_op(op);
    if (!o) return;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    // Unregister all existing operand uses.
    if (o->aux && o->aux->operand_uses) {
        for (uint64_t i = 0; i < o->n_operands; i++) {
            use_unregister(o->operands[i], o->aux->operand_uses[i]);
        }
    }
    MLIR_ValueHandle *nv = n ? arena_new_array(arena, MLIR_ValueHandle, n) : NULL;
    for (size_t i = 0; i < n; i++) nv[i] = values ? values[i] : MLIR_INVALID_HANDLE;
    o->operands = nv;
    o->n_operands = (uint32_t)n;
    if (!o->aux) o->aux = op_aux_default(arena);
    o->aux->operand_uses = n ? arena_new_array(arena, UseNode *, n) : NULL;
    for (size_t i = 0; i < n; i++) {
        o->aux->operand_uses[i] = use_register_operand(ctx, op, nv[i], (uint32_t)i);
    }
}

void MLIR_AppendOpSuccessorOperand(MLIR_Context *ctx, MLIR_OpHandle op,
                                   size_t succ_idx, MLIR_ValueHandle value) {
    IR_Op *o = resolve_op(op);
    if (!o || succ_idx >= o->n_successors) return;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    size_t old_n = o->n_successor_operands[succ_idx];
    MLIR_ValueHandle *na = arena_new_array(arena, MLIR_ValueHandle, old_n + 1);
    if (old_n > 0)
        memcpy(na, o->successor_operands[succ_idx],
               old_n * sizeof(MLIR_ValueHandle));
    na[old_n] = value;
    // Resize the parallel UseNode-pointer array. Existing UseNodes stay
    // valid (they're individually arena-allocated; only the pointer
    // array changes).
    if (!o->aux) o->aux = op_aux_default(arena);
    if (!o->aux->successor_operand_uses) {
        o->aux->successor_operand_uses = arena_new_array(arena, UseSlot, o->n_successors);
        for (size_t s = 0; s < o->n_successors; s++) o->aux->successor_operand_uses[s].arr = NULL;
    }
    UseNode **nu = arena_new_array(arena, UseNode *, old_n + 1);
    if (old_n > 0 && o->aux->successor_operand_uses[succ_idx].arr)
        memcpy(nu, o->aux->successor_operand_uses[succ_idx].arr, old_n * sizeof(UseNode *));
    nu[old_n] = use_register_succ_operand(ctx, op, value,
                                          (uint32_t)succ_idx, (uint32_t)old_n);
    o->successor_operands[succ_idx] = na;
    o->aux->successor_operand_uses[succ_idx].arr = nu;
    o->n_successor_operands[succ_idx] = old_n + 1;
}

void MLIR_SpliceBlockOps(MLIR_Context *ctx, MLIR_BlockHandle dst,
                         MLIR_BlockHandle src) {
    IR_Block *d = resolve_block(dst);
    IR_Block *s = resolve_block(src);
    if (!d || !s || s->n_operations == 0) return;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    uint64_t new_n = d->n_operations + s->n_operations;
    // Grow geometrically (matching AppendBlockOp) so repeated splices
    // into the same dst block don't cost O(N^2) memory.
    if (new_n > d->cap_operations) {
        uint64_t new_cap = d->cap_operations ? d->cap_operations : 4;
        while (new_cap < new_n) new_cap *= 2;
        MLIR_OpHandle *no = arena_new_array(arena, MLIR_OpHandle, new_cap);
        if (d->n_operations > 0)
            memcpy(no, d->operations, d->n_operations * sizeof(MLIR_OpHandle));
        d->operations = no;
        d->cap_operations = (uint32_t)new_cap;
    }
    memcpy(d->operations + d->n_operations, s->operations,
           s->n_operations * sizeof(MLIR_OpHandle));
    // Reparent moved ops.
    for (size_t i = 0; i < s->n_operations; i++) {
        IR_Op *op = resolve_op(s->operations[i]);
        if (op) op->parent_block = dst;
    }
    d->n_operations = (uint32_t)new_n;
    s->operations = NULL;
    s->n_operations = 0;
    s->cap_operations = 0;
}

// Native cf->scf lift: dispatch to the agnostic C port. The port handles
// the patterns it understands and returns false otherwise; for native we
// do not have an upstream fallback so callers (wasm lowering) will reject
// any leftover cf ops at the wasmssa-lower stage.
#include "mlir_lift_cf_to_scf.h"
bool MLIR_LiftCfToScf(MLIR_Context *ctx, MLIR_OpHandle module) {
    return MLIR_LiftCfToScfNative(ctx, module);
}

// The lowering / LLVM-IR-translation / wasm-translation entry points
// declared in mlir_api.h are implemented in dedicated agnostic
// translation units (mlir_lower_to_llvm.c, mlir_translate_to_llvm_ir.c,
// mlir_translate_to_wasm.c). They are linked into hosted builds
// alongside this file. There is no parser-only "native" build, so we
// no longer need fallback stubs here.

#ifdef __cplusplus
}
#endif
