#include "bir.h"
#include <string.h>
#include <stdio.h>

/* ---- Name Tables ---- */

/* Every opcode gets a name. Even the ones that probably shouldn't exist. */
static const char *op_names[BIR_OP_COUNT] = {
    [BIR_ADD]           = "add",
    [BIR_SUB]           = "sub",
    [BIR_MUL]           = "mul",
    [BIR_UMULHI]        = "umulhi",
    [BIR_POPCOUNT]      = "popcount",
    [BIR_CTZ]           = "ctz",
    [BIR_CLZ]           = "clz",
    [BIR_BREV]          = "brev",
    [BIR_SDIV]          = "sdiv",
    [BIR_UDIV]          = "udiv",
    [BIR_SREM]          = "srem",
    [BIR_UREM]          = "urem",
    [BIR_FADD]          = "fadd",
    [BIR_FSUB]          = "fsub",
    [BIR_FMUL]          = "fmul",
    [BIR_FDIV]          = "fdiv",
    [BIR_FREM]          = "frem",

    [BIR_AND]           = "and",
    [BIR_OR]            = "or",
    [BIR_XOR]           = "xor",
    [BIR_SHL]           = "shl",
    [BIR_LSHR]          = "lshr",
    [BIR_ASHR]          = "ashr",

    [BIR_ICMP]          = "icmp",
    [BIR_FCMP]          = "fcmp",

    [BIR_TRUNC]         = "trunc",
    [BIR_ZEXT]          = "zext",
    [BIR_SEXT]          = "sext",
    [BIR_FPTRUNC]       = "fptrunc",
    [BIR_FPEXT]         = "fpext",
    [BIR_FPTOSI]        = "fptosi",
    [BIR_FPTOUI]        = "fptoui",
    [BIR_SITOFP]        = "sitofp",
    [BIR_UITOFP]        = "uitofp",
    [BIR_PTRTOINT]      = "ptrtoint",
    [BIR_INTTOPTR]      = "inttoptr",
    [BIR_BITCAST]       = "bitcast",

    [BIR_ALLOCA]        = "alloca",
    [BIR_SHARED_ALLOC]  = "shared_alloc",
    [BIR_GLOBAL_REF]    = "global_ref",
    [BIR_LOAD]          = "load",
    [BIR_STORE]         = "store",
    [BIR_GEP]           = "gep",

    [BIR_BR]            = "br",
    [BIR_BR_COND]       = "br_cond",
    [BIR_SWITCH]        = "switch",
    [BIR_RET]           = "ret",
    [BIR_UNREACHABLE]   = "unreachable",

    [BIR_PHI]           = "phi",
    [BIR_PARAM]         = "param",

    [BIR_THREAD_ID]     = "thread_id",
    [BIR_BLOCK_ID]      = "block_id",
    [BIR_BLOCK_DIM]     = "block_dim",
    [BIR_GRID_DIM]      = "grid_dim",

    [BIR_BARRIER]       = "barrier",
    [BIR_BARRIER_GROUP] = "barrier_group",

    [BIR_ATOMIC_ADD]    = "atomic_add",
    [BIR_ATOMIC_SUB]    = "atomic_sub",
    [BIR_ATOMIC_AND]    = "atomic_and",
    [BIR_ATOMIC_OR]     = "atomic_or",
    [BIR_ATOMIC_XOR]    = "atomic_xor",
    [BIR_ATOMIC_MIN]    = "atomic_min",
    [BIR_ATOMIC_MAX]    = "atomic_max",
    [BIR_ATOMIC_XCHG]   = "atomic_xchg",
    [BIR_ATOMIC_CAS]    = "atomic_cas",
    [BIR_ATOMIC_LOAD]   = "atomic_load",
    [BIR_ATOMIC_STORE]  = "atomic_store",

    [BIR_SHFL]          = "shfl",
    [BIR_SHFL_UP]       = "shfl_up",
    [BIR_SHFL_DOWN]     = "shfl_down",
    [BIR_SHFL_XOR]      = "shfl_xor",
    [BIR_BALLOT]        = "ballot",
    [BIR_VOTE_ANY]      = "vote_any",
    [BIR_VOTE_ALL]      = "vote_all",

    [BIR_SQRT]          = "sqrt",
    [BIR_RSQ]           = "rsq",
    [BIR_RCP]           = "rcp",
    [BIR_EXP2]          = "exp2",
    [BIR_LOG2]          = "log2",
    [BIR_SIN]           = "sin",
    [BIR_COS]           = "cos",
    [BIR_FABS]          = "fabs",
    [BIR_FLOOR]         = "floor",
    [BIR_CEIL]          = "ceil",
    [BIR_FTRUNC]        = "ftrunc",
    [BIR_RNDNE]         = "rndne",
    [BIR_FMAX]          = "fmax",
    [BIR_FMIN]          = "fmin",

    [BIR_MFMA]          = "mfma",
    [BIR_MMA]           = "mma",
    [BIR_MFRG]          = "mfrg",

    [BIR_CALL]          = "call",
    [BIR_SELECT]        = "select",
    [BIR_INLINE_ASM]    = "inline_asm",
};

static const char *cmp_names[BIR_CMP_COUNT] = {
    [BIR_ICMP_EQ]  = "eq",  [BIR_ICMP_NE]  = "ne",
    [BIR_ICMP_SLT] = "slt", [BIR_ICMP_SLE] = "sle",
    [BIR_ICMP_SGT] = "sgt", [BIR_ICMP_SGE] = "sge",
    [BIR_ICMP_ULT] = "ult", [BIR_ICMP_ULE] = "ule",
    [BIR_ICMP_UGT] = "ugt", [BIR_ICMP_UGE] = "uge",

    [BIR_FCMP_OEQ] = "oeq", [BIR_FCMP_ONE] = "one",
    [BIR_FCMP_OLT] = "olt", [BIR_FCMP_OLE] = "ole",
    [BIR_FCMP_OGT] = "ogt", [BIR_FCMP_OGE] = "oge",
    [BIR_FCMP_UEQ] = "ueq", [BIR_FCMP_UNE] = "une",
    [BIR_FCMP_ULT] = "ult", [BIR_FCMP_ULE] = "ule",
    [BIR_FCMP_UGT] = "ugt", [BIR_FCMP_UGE] = "uge",
    [BIR_FCMP_ORD] = "ord", [BIR_FCMP_UNO] = "uno",
};

static const char *addrspace_names[BIR_AS_COUNT] = {
    [BIR_AS_PRIVATE]  = "private",
    [BIR_AS_SHARED]   = "shared",
    [BIR_AS_GLOBAL]   = "global",
    [BIR_AS_CONSTANT] = "constant",
    [BIR_AS_GENERIC]  = "generic",
};

static const char *type_kind_names[BIR_TYPE_KIND_COUNT] = {
    [BIR_TYPE_VOID]   = "void",
    [BIR_TYPE_INT]    = "int",
    [BIR_TYPE_FLOAT]  = "float",
    [BIR_TYPE_BFLOAT] = "bfloat",
    [BIR_TYPE_PTR]    = "ptr",
    [BIR_TYPE_VECTOR] = "vector",
    [BIR_TYPE_STRUCT] = "struct",
    [BIR_TYPE_ARRAY]  = "array",
    [BIR_TYPE_FUNC]   = "func",
};

static const char *order_names[BIR_ORDER_COUNT] = {
    [BIR_ORDER_RELAXED] = "relaxed",
    [BIR_ORDER_ACQUIRE] = "acquire",
    [BIR_ORDER_RELEASE] = "release",
    [BIR_ORDER_ACQ_REL] = "acq_rel",
    [BIR_ORDER_SEQ_CST] = "seq_cst",
};

const char *bir_op_name(int op)
{
    if (op >= 0 && op < BIR_OP_COUNT && op_names[op])
        return op_names[op];
    return "???";
}

const char *bir_type_kind_name(int kind)
{
    if (kind >= 0 && kind < BIR_TYPE_KIND_COUNT)
        return type_kind_names[kind];
    return "???";
}

const char *bir_cmp_name(int pred)
{
    if (pred >= 0 && pred < BIR_CMP_COUNT)
        return cmp_names[pred];
    return "???";
}

const char *bir_addrspace_name(int as)
{
    if (as >= 0 && as < BIR_AS_COUNT)
        return addrspace_names[as];
    return "???";
}

const char *bir_order_name(int ord)
{
    if (ord >= 0 && ord < BIR_ORDER_COUNT)
        return order_names[ord];
    return "???";
}

/* ---- Pool overflow ---- */

/* Fixed order, so the report reads the same whichever pool filled first. */
static const struct { uint32_t bit; const char *name; uint32_t cap; }
pool_tab[] = {
    { BIR_P_TYPES,    "type",           BIR_MAX_TYPES       },
    { BIR_P_TFIELDS,  "type field",     BIR_MAX_TYPE_FIELDS },
    { BIR_P_STRINGS,  "string table",   BIR_MAX_STRINGS     },
    { BIR_P_CONSTS,   "constant",       BIR_MAX_CONSTS      },
    { BIR_P_INSTS,    "instruction",    BIR_MAX_INSTS       },
    { BIR_P_BLOCKS,   "block",          BIR_MAX_BLOCKS      },
    { BIR_P_FUNCS,    "function",       BIR_MAX_FUNCS       },
    { BIR_P_GLOBALS,  "global",         BIR_MAX_GLOBALS     },
    { BIR_P_EXTRAOPS, "extra operand",  BIR_MAX_EXTRA_OPS   },
    { BIR_P_PHIS,     "mem2reg phi",    0u                  },
};

void bir_pfull(bir_module_t *M, uint32_t bit)
{
    if (M != NULL) M->pool_full |= bit;
}

int bir_pchk(const bir_module_t *M, const char *phase)
{
    if (M == NULL || M->pool_full == 0u) return BC_OK;

    for (uint32_t i = 0; i < sizeof(pool_tab) / sizeof(pool_tab[0]); i++) {
        if (!(M->pool_full & pool_tab[i].bit)) continue;
        if (pool_tab[i].cap != 0u)
            fprintf(stderr, "E120: BIR %s pool exhausted during %s "
                    "(capacity %u). Raise the matching BIR_MAX_* and "
                    "rebuild.\n", pool_tab[i].name, phase, pool_tab[i].cap);
        else
            fprintf(stderr, "E120: BIR %s pool exhausted during %s.\n",
                    pool_tab[i].name, phase);
    }
    return BC_ERR_OVERFLOW;
}

/* ---- Module Init ---- */

void bir_module_init(bir_module_t *M)
{
    memset(M, 0, sizeof(*M));
    /* Reserve type 0 as void.  ptr_inner() returns types[t].inner,
       and callers treat 0 as "no element type".  If some other type
       (e.g. f32) happened to land at index 0, the sentinel check
       would misfire.  Pinning void at 0 prevents the collision. */
    bir_type_void(M);
}

/* ---- Type Interning ---- */

/* Simple types: compared field-by-field. No indirection needed. */
static int type_eq_simple(const bir_type_t *a, const bir_type_t *b)
{
    return a->kind == b->kind
        && a->addrspace == b->addrspace
        && a->width == b->width
        && a->inner == b->inner
        && a->count == b->count
        && a->num_fields == b->num_fields;
}

static uint32_t intern_type(bir_module_t *M, const bir_type_t *t)
{
    uint32_t guard = M->num_types;
    for (uint32_t i = 0; i < M->num_types && guard > 0; i++, guard--) {
        if (type_eq_simple(&M->types[i], t))
            return i;
    }
    if (M->num_types >= BIR_MAX_TYPES) {
        bir_pfull(M, BIR_P_TYPES);
        return 0;
    }
    uint32_t idx = M->num_types++;
    M->types[idx] = *t;
    return idx;
}

/* Compound types (struct, func): must compare actual field type indices. */
static uint32_t intern_compound(bir_module_t *M, uint8_t kind,
                                uint32_t inner, const uint32_t *fields,
                                int nfields)
{
    uint32_t guard = M->num_types;
    for (uint32_t i = 0; i < M->num_types && guard > 0; i++, guard--) {
        bir_type_t *t = &M->types[i];
        if (t->kind != kind || t->inner != inner
            || t->num_fields != (uint16_t)nfields)
            continue;
        int match = 1;
        for (int j = 0; j < nfields && match; j++) {
            if (M->type_fields[t->count + (uint32_t)j] != fields[j])
                match = 0;
        }
        if (match) return i;
    }
    if (M->num_type_fields + (uint32_t)nfields > BIR_MAX_TYPE_FIELDS) {
        bir_pfull(M, BIR_P_TFIELDS);
        return 0;
    }
    if (M->num_types >= BIR_MAX_TYPES) {
        bir_pfull(M, BIR_P_TYPES);
        return 0;
    }

    uint32_t start = M->num_type_fields;
    for (int i = 0; i < nfields; i++)
        M->type_fields[M->num_type_fields++] = fields[i];

    uint32_t idx = M->num_types++;
    bir_type_t *nt = &M->types[idx];
    memset(nt, 0, sizeof(*nt));
    nt->kind = kind;
    nt->inner = inner;
    nt->count = start;
    nt->num_fields = (uint16_t)nfields;
    return idx;
}

uint32_t bir_type_void(bir_module_t *M)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_VOID;
    return intern_type(M, &t);
}

uint32_t bir_type_int(bir_module_t *M, int width_bits)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_INT;
    t.width = (uint16_t)width_bits;
    return intern_type(M, &t);
}

uint32_t bir_type_float(bir_module_t *M, int width_bits)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_FLOAT;
    t.width = (uint16_t)width_bits;
    return intern_type(M, &t);
}

uint32_t bir_type_bfloat(bir_module_t *M)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_BFLOAT;
    t.width = 16;
    return intern_type(M, &t);
}

uint32_t bir_type_ptr(bir_module_t *M, uint32_t pointee, int addrspace)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_PTR;
    t.addrspace = (uint8_t)addrspace;
    t.inner = pointee;
    return intern_type(M, &t);
}

uint32_t bir_type_array(bir_module_t *M, uint32_t elem, uint32_t count)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_ARRAY;
    t.inner = elem;
    t.count = count;
    return intern_type(M, &t);
}

uint32_t bir_type_vector(bir_module_t *M, uint32_t elem, uint32_t count)
{
    bir_type_t t;
    memset(&t, 0, sizeof(t));
    t.kind = BIR_TYPE_VECTOR;
    t.inner = elem;
    t.width = (uint16_t)count;
    return intern_type(M, &t);
}

uint32_t bir_type_struct(bir_module_t *M, const uint32_t *fields, int nfields)
{
    return intern_compound(M, BIR_TYPE_STRUCT, 0, fields, nfields);
}

uint32_t bir_type_func(bir_module_t *M, uint32_t ret,
                       const uint32_t *params, int nparams)
{
    return intern_compound(M, BIR_TYPE_FUNC, ret, params, nparams);
}

/* ---- Type Sizes ---- */

static uint32_t bsz_al(uint32_t sz, uint32_t psz)
{
    uint32_t a = 1;
    if (sz > psz) sz = psz;
    while (a * 2u <= sz) a *= 2u;
    return a;
}

static uint32_t bsz_up(uint32_t x, uint32_t a)
{
    return (x + a - 1u) & ~(a - 1u);
}

uint32_t bir_bsz(const bir_module_t *M, uint32_t ty, uint32_t psz)
{
    struct { uint32_t ty, mul, fld, sum, alg; } fr[BIR_BSZ_DEEP];
    uint32_t sp = 1, guard = 4u * BIR_MAX_TYPE_FIELDS;

    fr[0].ty = ty; fr[0].mul = 1; fr[0].fld = 0; fr[0].sum = 0; fr[0].alg = 1;

    while (guard--) {
        uint32_t i = sp - 1, w, va, val;
        const bir_type_t *T;

        if (fr[i].ty >= M->num_types) return 0;
        T = &M->types[fr[i].ty];

        if (T->kind == BIR_TYPE_ARRAY || T->kind == BIR_TYPE_VECTOR) {
            uint32_t n = (T->kind == BIR_TYPE_ARRAY) ? T->count
                                                     : (uint32_t)T->width;
            if (n && fr[i].mul > 0xFFFFFFFFu / n) return 0;
            fr[i].mul *= n; fr[i].ty = T->inner; continue;
        }

        if (T->kind == BIR_TYPE_STRUCT) {
            if (fr[i].fld < (uint32_t)T->num_fields) {
                uint32_t f = T->count + fr[i].fld++;
                if (f >= M->num_type_fields || sp >= BIR_BSZ_DEEP) return 0;
                fr[sp].ty = M->type_fields[f];
                fr[sp].mul = 1; fr[sp].fld = 0; fr[sp].sum = 0; fr[sp].alg = 1;
                sp++;
                continue;
            }
            w  = bsz_up(fr[i].sum, fr[i].alg);
            va = fr[i].alg;
        } else {
            switch (T->kind) {
            case BIR_TYPE_INT:
            case BIR_TYPE_FLOAT:
            case BIR_TYPE_BFLOAT: w = ((uint32_t)T->width + 7u) / 8u; break;
            case BIR_TYPE_PTR:    w = psz; break;
            default:              return 0;
            }
            va = bsz_al(w, psz);
        }
        if (!w) return 0;

        if (fr[i].mul > 0xFFFFFFFFu / w) return 0;
        val = fr[i].mul * w;
        if (--sp == 0) return val;

        if (va > fr[sp - 1].alg) fr[sp - 1].alg = va;
        if (bsz_up(fr[sp - 1].sum, va) > 0xFFFFFFFFu - val) return 0;
        fr[sp - 1].sum = bsz_up(fr[sp - 1].sum, va) + val;
    }
    return 0;
}

uint32_t bir_gsz(const bir_module_t *M, uint32_t ty, uint32_t psz)
{
    if (ty < M->num_types && M->types[ty].kind == BIR_TYPE_PTR)
        return bir_bsz(M, M->types[ty].inner, psz);
    return bir_bsz(M, ty, psz);
}

/* ---- String Table ---- */

uint32_t bir_add_string(bir_module_t *M, const char *s, uint32_t len)
{
    /* Offset 0 is a live string, not a sentinel. */
    if (M->string_len + len + 1 > BIR_MAX_STRINGS) {
        bir_pfull(M, BIR_P_STRINGS);
        return 0;
    }
    uint32_t offset = M->string_len;
    memcpy(&M->strings[offset], s, len);
    M->strings[offset + len] = '\0';
    M->string_len += len + 1;
    return offset;
}

/* ---- Constants ---- */

/* Nothing is pinned at const 0 the way void is at type 0, so a refusal
   here is indistinguishable from a real index. Hence the bit. */

uint32_t bir_const_int(bir_module_t *M, uint32_t type, int64_t val)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_INT
            && M->consts[i].type == type
            && M->consts[i].d.ival == val)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_INT;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.ival = val;
    return idx;
}

/* A constant whose value is a sequence of bytes interned in the
 * module's strings table. Used for string literals; could be used
 * for other byte-array constants later. The offset and length name
 * the slice of M->strings; we do not deduplicate the bytes because
 * deduplicating literals can be surprising (two source-distinct
 * literals collapsing into one pointer comparison) and we are not
 * doing the work to confirm that surprise is welcome. */

uint32_t bir_const_bytes(bir_module_t *M, uint32_t type,
                         uint32_t off, uint32_t len)
{
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_BYTES;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.bytes.off = off;
    M->consts[idx].d.bytes.len = len;
    return idx;
}

int bir_global_is_bytes(const bir_module_t *M, uint32_t gi)
{
    if (gi >= M->num_globals) return 0;
    uint32_t init = M->globals[gi].initializer;
    if (init == BIR_VAL_NONE) return 0;
    if (!BIR_VAL_IS_CONST(init)) return 0;
    uint32_t ci = BIR_VAL_INDEX(init);
    if (ci >= M->num_consts) return 0;
    return M->consts[ci].kind == BIR_CONST_BYTES;
}

int bir_mang(const char *name, uint16_t tu, char *out, int size)
{
    int n = snprintf(out, (size_t)size, "%s__%u", name, (unsigned)tu);
    if (n < 0 || n >= size) { out[0] = '\0'; return 1; }
    return 0;
}

uint32_t bir_fsym(const bir_module_t *M, const char *name, uint16_t tu,
                  int nargs)
{
    char mng[BIR_SYM_MAX];

    for (int pass = 0; pass < 2; pass++) {
        const char *want = name;
        uint16_t wtu = BIR_TU_EXT;

        if (pass == 0) {
            if (tu == BIR_TU_EXT) continue;
            if (bir_mang(name, tu, mng, (int)sizeof mng) != 0) continue;
            want = mng;
            wtu = tu;
        }
        for (uint32_t i = 0; i < M->num_funcs; i++) {
            const bir_func_t *F = &M->funcs[i];
            if (F->tu != wtu || F->name >= M->string_len) continue;
            if (strcmp(&M->strings[F->name], want) != 0) continue;
            if (nargs < 0 || F->num_params == (uint16_t)nargs) return i;
        }
    }
    return BIR_SYM_NONE;
}

uint32_t bir_gsym(const bir_module_t *M, const char *name, uint16_t tu)
{
    char mng[BIR_SYM_MAX];

    if (tu != BIR_TU_EXT && bir_mang(name, tu, mng, (int)sizeof mng) == 0) {
        for (uint32_t i = 0; i < M->num_globals; i++) {
            const bir_global_t *G = &M->globals[i];
            if (G->tu != tu || G->name >= M->string_len) continue;
            if (strcmp(&M->strings[G->name], mng) == 0) return i;
        }
    }
    for (uint32_t i = 0; i < M->num_globals; i++) {
        const bir_global_t *G = &M->globals[i];
        if (G->tu != BIR_TU_EXT || G->name >= M->string_len) continue;
        if (strcmp(&M->strings[G->name], name) == 0) return i;
    }
    return BIR_SYM_NONE;
}

uint32_t bir_const_float(bir_module_t *M, uint32_t type, double val)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_FLOAT
            && M->consts[i].type == type
            && M->consts[i].d.fval == val)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_FLOAT;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.fval = val;
    return idx;
}

uint32_t bir_const_null(bir_module_t *M, uint32_t type)
{
    uint32_t guard = M->num_consts;
    for (uint32_t i = 0; i < M->num_consts && guard > 0; i++, guard--) {
        if (M->consts[i].kind == BIR_CONST_NULL && M->consts[i].type == type)
            return i;
    }
    if (M->num_consts >= BIR_MAX_CONSTS) {
        bir_pfull(M, BIR_P_CONSTS);
        return 0;
    }
    uint32_t idx = M->num_consts++;
    M->consts[idx].kind = BIR_CONST_NULL;
    memset(M->consts[idx].pad, 0, sizeof(M->consts[idx].pad));
    M->consts[idx].type = type;
    M->consts[idx].d.ival = 0;
    return idx;
}
