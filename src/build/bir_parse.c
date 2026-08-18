/* bir_parse.c -- see bir_parse.h. Coverage is partial; an unknown opcode is
 * an error rather than a silent skip. Extend by round-tripping real kernels. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "bir_parse.h"

/* ---- Lexer ---- */

/* One length for every identifier buffer, or copies truncate. */
#define NAME_MAX_LEN 128

typedef enum {
    T_EOF, T_IDENT,     /* bare word, including dotted like block_id.x */
    T_VAL,              /* %N */
    T_GLOBAL,           /* @name */
    T_INT,              /* decimal, possibly negative */
    T_FLOAT,            /* what %g writes: 2.5, 0.125, 1e+06 */
    T_LBRACE, T_RBRACE, T_LPAREN, T_RPAREN, T_LT, T_GT,
    T_LBRACK, T_RBRACK,
    T_COMMA, T_COLON, T_EQUALS
} tok_kind;

typedef struct {
    tok_kind kind;
    char     text[NAME_MAX_LEN];
    /* long long, because long is 32 bits on Windows and strtol quietly
       saturates a hash constant like 0x846ca68b to INT32_MAX. */
    long long num;
    double    fnum;
    uint32_t  line;
} tok_t;

typedef struct {
    const char *p;
    const char *name;
    uint32_t    line;
    tok_t       cur;
    int         err;
} lex_t;

static void lx_err(lex_t *L, const char *msg)
{
    if (L->err) return;             /* keep the first, it is the real one */
    L->err = 1;
    fprintf(stderr, "%s:%u: %s\n", L->name, L->line, msg);
}


static void skip_ws(lex_t *L)
{
    for (;;) {
        while (*L->p == ' ' || *L->p == '\t' || *L->p == '\r') L->p++;
        if (*L->p == '\n') { L->line++; L->p++; continue; }
        if (*L->p == ';') { while (*L->p && *L->p != '\n') L->p++; continue; }
        return;
    }
}

static int ident_ch(char c)
{
    return isalnum((unsigned char)c) || c == '_' || c == '.';
}

static void lx_next(lex_t *L)
{
    skip_ws(L);
    tok_t *t = &L->cur;
    memset(t, 0, sizeof *t);
    t->line = L->line;

    char c = *L->p;
    if (c == '\0') { t->kind = T_EOF; return; }

    switch (c) {
    case '{': L->p++; t->kind = T_LBRACE; return;
    case '}': L->p++; t->kind = T_RBRACE; return;
    case '(': L->p++; t->kind = T_LPAREN; return;
    case ')': L->p++; t->kind = T_RPAREN; return;
    case '<': L->p++; t->kind = T_LT;     return;
    case '>': L->p++; t->kind = T_GT;     return;
    case ',': L->p++; t->kind = T_COMMA;  return;
    case ':': L->p++; t->kind = T_COLON;  return;
    case '[': L->p++; t->kind = T_LBRACK; return;
    case ']': L->p++; t->kind = T_RBRACK; return;
    case '=': L->p++; t->kind = T_EQUALS; return;
    default: break;
    }

    if (c == '%' || c == '@') {
        L->p++;
        size_t n = 0;
        while (ident_ch(*L->p) && n < sizeof t->text - 1) t->text[n++] = *L->p++;
        t->text[n] = '\0';
        if (c == '%') {
            t->kind = T_VAL;
            t->num = strtoll(t->text, NULL, 10);
        } else {
            t->kind = T_GLOBAL;
        }
        return;
    }

    if (isdigit((unsigned char)c) || (c == '-' && isdigit((unsigned char)L->p[1]))) {
        /* strtod reaches further than strtol exactly when the text is a
         * decimal or exponent form, which is how the two are told apart. */
        char *ie = NULL, *fe = NULL;
        long long iv = strtoll(L->p, &ie, 10);
        double    fv = strtod(L->p, &fe);
        if (fe > ie) { t->kind = T_FLOAT; t->fnum = fv; L->p = fe; }
        else         { t->kind = T_INT;   t->num  = iv; L->p = ie; }
        return;
    }

    if (ident_ch(c)) {
        size_t n = 0;
        while (ident_ch(*L->p) && n < sizeof t->text - 1) t->text[n++] = *L->p++;
        t->text[n] = '\0';
        t->kind = T_IDENT;
        return;
    }

    lx_err(L, "unexpected character");
    L->p++;
}

static int is_ident(lex_t *L, const char *s)
{
    return L->cur.kind == T_IDENT && strcmp(L->cur.text, s) == 0;
}

static int eat_ident(lex_t *L, const char *s)
{
    if (!is_ident(L, s)) return 0;
    lx_next(L);
    return 1;
}

static int expect(lex_t *L, tok_kind k, const char *what)
{
    if (L->cur.kind != k) { lx_err(L, what); return 0; }
    lx_next(L);
    return 1;
}

/* Text numbers values per function; the builder returns absolute indices.
 * SSA means no forward references, so a flat array filled in order is enough. */

#define VMAP_MAX 65536

typedef struct {
    lex_t   L;
    bb_t   *B;
    bb_val  vmap[VMAP_MAX];
    uint32_t nv;

    /* Reserved up front: a branch names blocks that appear later. */
    char     blk_name[512][NAME_MAX_LEN];
    uint32_t blk_idx[512];
    uint32_t nblk;

    uint32_t depth;         /* ptr<> nesting, so the type walk cannot recurse away */
} P;

/* Deeper than anything a real module holds, and shallow enough that the stack
 * survives a file that nests on purpose. */
#define TYPE_MAX_DEPTH 16

static void vmap_set(P *p, long long local, bb_val v)
{
    if (local < 0 || local >= VMAP_MAX) { lx_err(&p->L, "value number out of range"); return; }
    if ((uint32_t)local >= p->nv) p->nv = (uint32_t)local + 1;
    p->vmap[local] = v;
}

static bb_val vmap_get(P *p, long long local)
{
    if (local < 0 || (uint32_t)local >= p->nv) {
        lx_err(&p->L, "reference to an undefined value");
        return BB_NONE;
    }
    return p->vmap[local];
}

static uint32_t blk_lookup(P *p, const char *name)
{
    for (uint32_t i = 0; i < p->nblk; i++)
        if (strcmp(p->blk_name[i], name) == 0) return p->blk_idx[i];
    lx_err(&p->L, "branch to an unknown block");
    return 0;
}

/* ---- Types ---- */

/* int/float widths come from the name: i32, f64, and so on. */
static uint32_t parse_type(P *p)
{
    lex_t *L = &p->L;

    if (is_ident(L, "void")) { lx_next(L); return bb_void(p->B); }

    /* [N x T] and <N x T>, which is what a shared_alloc points at. */
    if (L->cur.kind == T_LBRACK || L->cur.kind == T_LT) {
        int vec = (L->cur.kind == T_LT);
        lx_next(L);
        if (L->cur.kind != T_INT) { lx_err(L, "expected a count"); return 0; }
        uint32_t n = (uint32_t)L->cur.num;
        lx_next(L);
        if (!eat_ident(L, "x")) { lx_err(L, "expected 'x' in an array type"); return 0; }
        if (p->depth >= TYPE_MAX_DEPTH) { lx_err(L, "type nested too deeply"); return 0; }
        p->depth++;
        uint32_t e = parse_type(p);
        p->depth--;
        if (!expect(L, vec ? T_GT : T_RBRACK, "expected the closing bracket")) return 0;
        return vec ? bb_vec(p->B, e, n) : bb_arr(p->B, e, n);
    }

    if (is_ident(L, "ptr")) {
        lx_next(L);
        if (!expect(L, T_LT, "expected '<' after ptr")) return 0;
        int as = BB_AS_GENERIC;
        if      (eat_ident(L, "private"))  as = BB_AS_PRIVATE;
        else if (eat_ident(L, "shared"))   as = BB_AS_SHARED;
        else if (eat_ident(L, "global"))   as = BB_AS_GLOBAL;
        else if (eat_ident(L, "constant")) as = BB_AS_CONST;
        else if (eat_ident(L, "generic"))  as = BB_AS_GENERIC;
        else { lx_err(L, "unknown address space"); return 0; }
        if (!expect(L, T_COMMA, "expected ',' in ptr type")) return 0;
        if (p->depth >= TYPE_MAX_DEPTH) {
            lx_err(L, "pointer type nested too deeply");
            return 0;
        }
        p->depth++;
        uint32_t inner = parse_type(p);
        p->depth--;
        if (!expect(L, T_GT, "expected '>' closing ptr type")) return 0;
        return bb_ptr(p->B, inner, as);
    }

    if (L->cur.kind == T_IDENT) {
        char c = L->cur.text[0];
        int bits = atoi(L->cur.text + 1);
        if ((c == 'i' || c == 'f') && bits > 0) {
            lx_next(L);
            return (c == 'i') ? bb_int(p->B, bits) : bb_flt(p->B, bits);
        }
    }

    lx_err(L, "expected a type");
    return 0;
}

/* ---- Instructions ---- */

/* Names as bir_cmp_name prints them, so the two stay in step by inspection. */
static int cmp_pred(const char *s)
{
    static const struct { const char *n; int v; } tab[] = {
        { "eq",  BB_EQ  }, { "ne",  BB_NE  },
        { "slt", BB_SLT }, { "sle", BB_SLE },
        { "sgt", BB_SGT }, { "sge", BB_SGE },
        { "ult", BB_ULT }, { "ule", BB_ULE },
        { "ugt", BB_UGT }, { "uge", BB_UGE },
        { "oeq", BB_OEQ }, { "one", BB_ONE },
        { "olt", BB_OLT }, { "ole", BB_OLE },
        { "ogt", BB_OGT }, { "oge", BB_OGE },
        { NULL, 0 }
    };
    for (int i = 0; tab[i].n != NULL; i++)
        if (strcmp(s, tab[i].n) == 0) return tab[i].v;
    return -1;
}

/* thread_id.x and friends carry the dimension in the opcode name. */
static int dim_suffix(const char *s)
{
    const char *dot = strrchr(s, '.');
    if (dot == NULL) return 0;
    switch (dot[1]) { case 'x': return 0; case 'y': return 1; case 'z': return 2; }
    return -1;
}

/* ty is what the immediate would be typed as; ignored for %N operands. */
static bb_val read_operand(P *p, uint32_t ty)
{
    lex_t *L = &p->L;

    if (L->cur.kind == T_VAL) {
        long long n = L->cur.num;
        lx_next(L);
        return vmap_get(p, n);
    }

    if (L->cur.kind == T_INT || L->cur.kind == T_FLOAT) {
        int isf = (L->cur.kind == T_FLOAT);
        double fv = isf ? L->cur.fnum : (double)L->cur.num;
        long long iv = isf ? (long long)L->cur.fnum : L->cur.num;
        lx_next(L);
        /* %g drops the point on a whole float, so the type decides, not the text. */
        return bb_isflt(p->B, ty) ? bb_cf(p->B, ty, fv) : bb_ci(p->B, ty, iv);
    }

    lx_err(L, "expected a value operand");
    return BB_NONE;
}

/* "; line N" sits after the operands but must be set before emitting, so
 * scan ahead for it without disturbing the token stream. */
static void take_line_comment(P *p)
{
    const char *s = p->L.p;
    while (*s && *s != '\n') {
        if (s[0] == ';') {
            while (*s == ';' || *s == ' ' || *s == '\t') s++;
            if (!strncmp(s, "line ", 5)) bb_line(p->B, (uint32_t)atoi(s + 5));
            return;
        }
        s++;
    }
}

/* Returns 0 on success. `dest` is the %N being defined, or -1 for void. */
static int parse_inst(P *p, long long dest)
{
    lex_t *L = &p->L;
    if (L->cur.kind != T_IDENT) { lx_err(L, "expected an opcode"); return -1; }

    take_line_comment(p);

    char op[128];
    snprintf(op, sizeof op, "%s", L->cur.text);
    lx_next(L);

    bb_val r = BB_NONE;

    /* Thread model. No operands, dimension in the name. */
    if (!strncmp(op, "thread_id", 9) || !strncmp(op, "block_id", 8) ||
        !strncmp(op, "block_dim", 9) || !strncmp(op, "grid_dim", 8)) {
        int d = dim_suffix(op);
        if (d < 0) { lx_err(L, "bad dimension suffix"); return -1; }
        if      (!strncmp(op, "thread_id", 9)) r = bb_tid (p->B, d);
        else if (!strncmp(op, "block_id",  8)) r = bb_bid (p->B, d);
        else if (!strncmp(op, "block_dim", 9)) r = bb_bdim(p->B, d);
        else                                   r = bb_gdim(p->B, d);
        goto done;
    }

    /* Binary: OP TYPE %a, %b */
    {
        static const struct { const char *n; int kind; } bins[] = {
            {"add",BB_ADD},{"sub",BB_SUB},{"mul",BB_MUL},
            {"sdiv",BB_SDIV},{"udiv",BB_UDIV},{"srem",BB_SREM},{"urem",BB_UREM},
            {"fadd",BB_FADD},{"fsub",BB_FSUB},{"fmul",BB_FMUL},
            {"fdiv",BB_FDIV},{"frem",BB_FREM},
            {"and",BB_AND},{"or",BB_OR},{"xor",BB_XOR},
            {"shl",BB_SHL},{"lshr",BB_LSHR},{"ashr",BB_ASHR},
            {NULL,0}
        };
        for (int i = 0; bins[i].n; i++) {
            if (strcmp(op, bins[i].n)) continue;
            uint32_t t = parse_type(p);
            bb_val a = read_operand(p, t);
            if (!expect(L, T_COMMA, "expected ',' between operands")) return -1;
            bb_val b = read_operand(p, t);
            r = bb_op(p->B, bins[i].kind, t, a, b);
            goto done;
        }
    }

    /* Intrinsics print no type, so the operand carries it: sqrt %a, fmin %a %b */
    {
        static const struct { const char *n; int fn; int arity; } fns[] = {
            {"sqrt",BB_SQRT,1},{"rsq",BB_RSQ,1},{"rcp",BB_RCP,1},
            {"exp2",BB_EXP2,1},{"log2",BB_LOG2,1},{"sin",BB_SIN,1},{"cos",BB_COS,1},
            {"fabs",BB_FABS,1},{"floor",BB_FLOOR,1},{"ceil",BB_CEIL,1},
            {"ftrunc",BB_FTRUNC,1},{"rndne",BB_RNDNE,1},
            {"fmax",BB_FMAX,2},{"fmin",BB_FMIN,2},
            {NULL,0,0}
        };
        for (int i = 0; fns[i].n; i++) {
            if (strcmp(op, fns[i].n)) continue;
            uint32_t ft = bb_flt(p->B, 32);
            bb_val a = read_operand(p, ft);
            uint32_t t = bb_tyof(p->B, a);
            if (t == 0) t = ft;
            if (fns[i].arity == 1) {
                r = bb_fn1(p->B, fns[i].fn, t, a);
            } else {
                if (L->cur.kind == T_COMMA) lx_next(L);   /* printed without one */
                bb_val b = read_operand(p, t);
                r = bb_fn2(p->B, fns[i].fn, t, a, b);
            }
            goto done;
        }
    }

    /* Conversions: OP SRCTYPE %v to DSTTYPE */
    {
        static const struct { const char *n; int c; } cvts[] = {
            {"trunc",BB_TRUNC},{"zext",BB_ZEXT},{"sext",BB_SEXT},
            {"fptrunc",BB_FPTRUNC},{"fpext",BB_FPEXT},
            {"fptosi",BB_FPTOSI},{"fptoui",BB_FPTOUI},
            {"sitofp",BB_SITOFP},{"uitofp",BB_UITOFP},{"bitcast",BB_BITCAST},
            {NULL,0}
        };
        for (int i = 0; cvts[i].n; i++) {
            if (strcmp(op, cvts[i].n)) continue;
            uint32_t src = parse_type(p);
            bb_val a = read_operand(p, src);
            if (!eat_ident(L, "to")) { lx_err(L, "expected 'to' in a conversion"); return -1; }
            uint32_t dst = parse_type(p);
            r = bb_cvt(p->B, cvts[i].c, dst, a);
            goto done;
        }
    }

    /* select TYPE %cond, %true, %false */
    if (!strcmp(op, "select")) {
        uint32_t t = parse_type(p);
        bb_val cnd = read_operand(p, bb_int(p->B, 1));
        if (!expect(L, T_COMMA, "expected ',' after the condition")) return -1;
        bb_val tv = read_operand(p, t);
        if (!expect(L, T_COMMA, "expected ',' between the arms")) return -1;
        bb_val fv = read_operand(p, t);
        r = bb_sel(p->B, t, cnd, tv, fv);
        goto done;
    }

    /* icmp/fcmp PRED TYPE %a, %b */
    if (!strcmp(op, "icmp") || !strcmp(op, "fcmp")) {
        int isf = (op[0] == 'f');
        if (L->cur.kind != T_IDENT) { lx_err(L, "expected a predicate"); return -1; }
        int pred = cmp_pred(L->cur.text);
        if (pred < 0) { lx_err(L, "unknown comparison predicate"); return -1; }
        lx_next(L);
        uint32_t ct = parse_type(p);      /* operand type, implied by the values */
        bb_val a = read_operand(p, ct);
        if (!expect(L, T_COMMA, "expected ',' between operands")) return -1;
        bb_val b = read_operand(p, ct);
        r = isf ? bb_fcmp(p->B, pred, a, b) : bb_icmp(p->B, pred, a, b);
        goto done;
    }

    /* gep TYPE, %base, %idx */
    if (!strcmp(op, "gep")) {
        uint32_t t = parse_type(p);
        if (!expect(L, T_COMMA, "expected ',' after gep type")) return -1;
        bb_val base = read_operand(p, t);
        if (!expect(L, T_COMMA, "expected ',' before gep index")) return -1;
        bb_val idx = read_operand(p, bb_int(p->B, 32));
        r = bb_gep(p->B, t, base, idx);
        goto done;
    }

    if (!strcmp(op, "barrier")) { bb_barr(p->B); goto done; }

    if (!strcmp(op, "shared_alloc")) {
        r = bb_shal(p->B, parse_type(p));
        goto done;
    }

    /* alloca TYPE, where TYPE is the pointer it yields */
    if (!strcmp(op, "alloca")) {
        r = bb_alca(p->B, parse_type(p));
        goto done;
    }

    /* load TYPE, %ptr */
    if (!strcmp(op, "load")) {
        uint32_t t = parse_type(p);
        if (!expect(L, T_COMMA, "expected ',' after load type")) return -1;
        bb_val a = read_operand(p, t);
        r = bb_load(p->B, t, a);
        goto done;
    }

    /* store TYPE %val, %ptr */
    if (!strcmp(op, "store")) {
        uint32_t st = parse_type(p);
        bb_val v = read_operand(p, st);
        if (!expect(L, T_COMMA, "expected ',' before store address")) return -1;
        bb_val a = read_operand(p, st);
        bb_store(p->B, v, a);
        goto done;
    }

    /* br LABEL */
    if (!strcmp(op, "br")) {
        if (L->cur.kind != T_IDENT) { lx_err(L, "expected a block label"); return -1; }
        uint32_t b = blk_lookup(p, L->cur.text);
        lx_next(L);
        bb_br(p->B, b);
        goto done;
    }

    /* br_cond %c, TRUE, FALSE [merge M] */
    if (!strcmp(op, "br_cond")) {
        bb_val c = read_operand(p, bb_int(p->B, 1));
        if (!expect(L, T_COMMA, "expected ',' after condition")) return -1;
        if (L->cur.kind != T_IDENT) { lx_err(L, "expected a block label"); return -1; }
        uint32_t bt = blk_lookup(p, L->cur.text); lx_next(L);
        if (!expect(L, T_COMMA, "expected ',' between labels")) return -1;
        if (L->cur.kind != T_IDENT) { lx_err(L, "expected a block label"); return -1; }
        uint32_t bf = blk_lookup(p, L->cur.text); lx_next(L);
        uint32_t bm = bf;
        if (eat_ident(L, "merge")) {
            if (L->cur.kind != T_IDENT) { lx_err(L, "expected a merge label"); return -1; }
            bm = blk_lookup(p, L->cur.text); lx_next(L);
        }
        bb_brif(p->B, c, bt, bf, bm);
        goto done;
    }

    if (!strcmp(op, "ret")) {
        if (eat_ident(L, "void")) { bb_ret(p->B); goto done; }
        uint32_t rt = parse_type(p);
        bb_val rv = read_operand(p, rt);
        bb_retv(p->B, rt, rv);
        goto done;
    }

    /* call TYPE @name(%a, %b) */
    if (!strcmp(op, "call")) {
        uint32_t rt = parse_type(p);
        if (L->cur.kind != T_GLOBAL) { lx_err(L, "expected @name after call"); return -1; }
        int fidx = bb_ffind(p->B, L->cur.text);
        if (fidx < 0) { lx_err(L, "call to an unknown function"); return -1; }
        lx_next(L);
        if (!expect(L, T_LPAREN, "expected '(' after the callee")) return -1;
        bb_val argv[BB_MAX_ARGS];
        int argc = 0;
        while (L->cur.kind != T_RPAREN && L->cur.kind != T_EOF && !L->err) {
            if (argc >= BB_MAX_ARGS) { lx_err(L, "too many call arguments"); return -1; }
            argv[argc++] = read_operand(p, rt);
            if (L->cur.kind == T_COMMA) lx_next(L);
        }
        if (!expect(L, T_RPAREN, "expected ')' closing the arguments")) return -1;
        r = bb_call(p->B, rt, fidx, argv, argc);
        goto done;
    }

    lx_err(L, "unsupported opcode");
    return -1;

done:
    if (dest >= 0) vmap_set(p, dest, r);
    return p->L.err ? -1 : 0;
}

/* ---- Blocks and functions ---- */

/* One pass for "name:" at line start, reserving every block. */
static void prescan_blocks(P *p, const char *body)
{
    const char *s = body;
    int at_line_start = 1;
    while (*s && *s != '}') {
        if (at_line_start) {
            while (*s == ' ' || *s == '\t') s++;
            const char *st = s;
            while (ident_ch(*s)) s++;
            if (s > st && *s == ':' && p->nblk < 512) {
                size_t n = (size_t)(s - st);
                if (n > NAME_MAX_LEN - 1) n = NAME_MAX_LEN - 1;
                memcpy(p->blk_name[p->nblk], st, n);
                p->blk_name[p->nblk][n] = '\0';
                p->blk_idx[p->nblk] = bb_blk(p->B, p->blk_name[p->nblk]);
                p->nblk++;
            }
        }
        at_line_start = (*s == '\n');
        if (*s) s++;
    }
}

static int parse_func(P *p, const char *body_start)
{
    lex_t *L = &p->L;

    if (L->cur.kind != T_GLOBAL) { lx_err(L, "expected @name after func"); return -1; }
    char name[NAME_MAX_LEN];
    snprintf(name, sizeof name, "%s", L->cur.text);
    lx_next(L);

    if (!expect(L, T_LPAREN, "expected '(' after function name")) return -1;

    uint32_t ptypes[32];
    long long pnums[32];
    int np = 0;
    while (L->cur.kind != T_RPAREN && L->cur.kind != T_EOF && !L->err) {
        if (np >= 32) { lx_err(L, "too many parameters"); return -1; }
        ptypes[np] = parse_type(p);
        if (L->cur.kind != T_VAL) { lx_err(L, "expected %N after parameter type"); return -1; }
        pnums[np] = L->cur.num;
        lx_next(L);
        np++;
        if (L->cur.kind == T_COMMA) lx_next(L);
    }
    if (!expect(L, T_RPAREN, "expected ')' closing parameters")) return -1;

    int kernel = 0;
    while (L->cur.kind == T_IDENT) {
        if (!strcmp(L->cur.text, "__global__")) kernel = 1;
        else if (strcmp(L->cur.text, "__device__") && strcmp(L->cur.text, "__host__")) break;
        lx_next(L);
    }

    if (bb_func(p->B, name, bb_void(p->B), ptypes, np, kernel) != 0) {
        lx_err(L, "could not open function");
        return -1;
    }

    if (!expect(L, T_LBRACE, "expected '{' opening function body")) return -1;

    p->nblk = 0;
    prescan_blocks(p, body_start);
    if (p->nblk == 0) { lx_err(L, "function has no blocks"); return -1; }

    /* Params are instructions but print only in the signature. */
    bb_open(p->B, p->blk_idx[0]);
    for (int i = 0; i < np; i++)
        vmap_set(p, pnums[i], bb_param(p->B, i, ptypes[i]));

    /* Block 0 is already open for the params. */
    int first = 1;

    while (L->cur.kind != T_RBRACE && L->cur.kind != T_EOF && !L->err) {
        /* A label starts a block. */
        if (L->cur.kind == T_IDENT) {
            const char *after = L->p;
            while (*after == ' ' || *after == '\t') after++;
            if (*after == ':') {
                char lbl[NAME_MAX_LEN];
                snprintf(lbl, sizeof lbl, "%s", L->cur.text);
                lx_next(L);
                lx_next(L);                     /* the colon */
                uint32_t bi = blk_lookup(p, lbl);
                if (first) first = 0;
                else { bb_close(p->B); bb_open(p->B, bi); }
                continue;
            }
        }

        long long dest = -1;
        if (L->cur.kind == T_VAL) {
            dest = L->cur.num;
            lx_next(L);
            if (!expect(L, T_EQUALS, "expected '=' after result value")) return -1;
        }
        if (parse_inst(p, dest) != 0) return -1;
        first = 0;
    }

    bb_close(p->B);
    bb_fend(p->B);

    if (!expect(L, T_RBRACE, "expected '}' closing function")) return -1;
    return L->err ? -1 : 0;
}

/* ---- Entry ---- */

bb_t *bir_parse(const char *text, const char *name)
{
    P *p = calloc(1, sizeof *p);
    if (p == NULL) return NULL;

    p->B = bb_new();
    if (p->B == NULL) { free(p); return NULL; }

    p->L.p    = text;
    p->L.name = (name != NULL) ? name : "<bir>";
    p->L.line = 1;
    lx_next(&p->L);

    while (p->L.cur.kind != T_EOF && !p->L.err) {
        if (eat_ident(&p->L, "func")) {

            const char *brace = strchr(p->L.p, '{');
            if (brace == NULL) { lx_err(&p->L, "function has no body"); break; }
            if (parse_func(p, brace + 1) != 0) break;
            continue;
        }
        lx_err(&p->L, "expected 'func' at top level");
    }

    int bad = p->L.err;
    bb_t *B = p->B;
    free(p);

    if (bad) { bb_free(B); return NULL; }
    return B;
}

bb_t *bir_parse_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }

    char *buf = malloc((size_t)n + 1);
    if (buf == NULL) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);

    bb_t *B = bir_parse(buf, path);
    free(buf);
    return B;
}
