/* ttypes.c -- BIR type system tests
 * Absorbs the noble spirit of test_bir.c into the harness.
 * test_bir.c served honourably. It has earned its retirement. */

#include "tharns.h"
#include "bir.h"

/* ---- Shared State ---- */

static bir_module_t *M;

static void typ_setup(void)
{
    if (!M) M = (bir_module_t *)malloc(sizeof(bir_module_t));
    bir_module_init(M);
}

/* ---- types: interning ---- */

static void typ01(void)
{
    typ_setup();

    uint32_t v1 = bir_type_void(M);
    uint32_t v2 = bir_type_void(M);
    CHEQ(v1, v2);

    uint32_t i32a = bir_type_int(M, 32);
    uint32_t i32b = bir_type_int(M, 32);
    CHEQ(i32a, i32b);

    uint32_t i64 = bir_type_int(M, 64);
    CHNE(i64, i32a);

    uint32_t f32a = bir_type_float(M, 32);
    uint32_t f32b = bir_type_float(M, 32);
    CHEQ(f32a, f32b);

    uint32_t f64 = bir_type_float(M, 64);
    CHNE(f64, f32a);

    uint32_t pg  = bir_type_ptr(M, f32a, BIR_AS_GLOBAL);
    uint32_t pg2 = bir_type_ptr(M, f32a, BIR_AS_GLOBAL);
    CHEQ(pg, pg2);

    uint32_t ps = bir_type_ptr(M, f32a, BIR_AS_SHARED);
    CHNE(ps, pg);

    uint32_t arr  = bir_type_array(M, f32a, 256);
    uint32_t arr2 = bir_type_array(M, f32a, 256);
    CHEQ(arr, arr2);

    uint32_t arr3 = bir_type_array(M, f32a, 512);
    CHNE(arr3, arr);

    uint32_t vec4  = bir_type_vector(M, f32a, 4);
    uint32_t vec4b = bir_type_vector(M, f32a, 4);
    CHEQ(vec4, vec4b);

    uint32_t fields1[] = { i32a, f32a };
    uint32_t s1 = bir_type_struct(M, fields1, 2);
    uint32_t s2 = bir_type_struct(M, fields1, 2);
    CHEQ(s1, s2);

    uint32_t fields2[] = { f32a, i32a };
    uint32_t s3 = bir_type_struct(M, fields2, 2);
    CHNE(s3, s1);

    uint32_t params[] = { i32a, pg };
    uint32_t fn1 = bir_type_func(M, v1, params, 2);
    uint32_t fn2 = bir_type_func(M, v1, params, 2);
    CHEQ(fn1, fn2);

    PASS();
}
TH_REG("typ", 1, "interning returns one id per type", typ01)

/* ---- types: constants ---- */

static void typ02(void)
{
    typ_setup();

    uint32_t i32 = bir_type_int(M, 32);
    uint32_t f64 = bir_type_float(M, 64);

    uint32_t c0a = bir_const_int(M, i32, 0);
    uint32_t c0b = bir_const_int(M, i32, 0);
    CHEQ(c0a, c0b);

    uint32_t c42 = bir_const_int(M, i32, 42);
    CHNE(c42, c0a);

    uint32_t cf  = bir_const_float(M, f64, 3.14);
    uint32_t cf2 = bir_const_float(M, f64, 3.14);
    CHEQ(cf, cf2);

    uint32_t ptr_t = bir_type_ptr(M, i32, BIR_AS_GLOBAL);
    uint32_t cn  = bir_const_null(M, ptr_t);
    uint32_t cn2 = bir_const_null(M, ptr_t);
    CHEQ(cn, cn2);

    /* Value reference encoding */
    uint32_t ref = BIR_MAKE_CONST(c42);
    CHECK(BIR_VAL_IS_CONST(ref));
    CHEQ(BIR_VAL_INDEX(ref), c42);

    uint32_t val_ref = BIR_MAKE_VAL(100);
    CHECK(!BIR_VAL_IS_CONST(val_ref));
    CHEQ(BIR_VAL_INDEX(val_ref), 100u);

    PASS();
}
TH_REG("typ", 2, "constant types", typ02)

/* ---- types: strings ---- */

static void typ03(void)
{
    typ_setup();

    uint32_t s1 = bir_add_string(M, "entry", 5);
    CHEQ(s1, 0u);
    CHSTR(&M->strings[s1], "entry");

    uint32_t s2 = bir_add_string(M, "loop", 4);
    CHEQ(s2, 6u);
    CHSTR(&M->strings[s2], "loop");

    PASS();
}
TH_REG("typ", 3, "string table interning", typ03)

/* ---- types: name tables ---- */

static void typ04(void)
{
    CHSTR(bir_op_name(BIR_ADD), "add");
    CHSTR(bir_op_name(BIR_BARRIER), "barrier");
    CHSTR(bir_op_name(BIR_THREAD_ID), "thread_id");
    CHSTR(bir_op_name(BIR_INLINE_ASM), "inline_asm");
    CHSTR(bir_op_name(9999), "???");

    CHSTR(bir_cmp_name(BIR_ICMP_EQ), "eq");
    CHSTR(bir_cmp_name(BIR_FCMP_OLT), "olt");

    CHSTR(bir_addrspace_name(BIR_AS_GLOBAL), "global");
    CHSTR(bir_addrspace_name(BIR_AS_SHARED), "shared");

    CHSTR(bir_type_kind_name(BIR_TYPE_PTR), "ptr");

    CHSTR(bir_order_name(BIR_ORDER_SEQ_CST), "seq_cst");

    PASS();
}
TH_REG("typ", 4, "name tables", typ04)

/* ---- types: printer doesn't segfault ---- */

static void typ05(void)
{
    typ_setup();

    uint32_t t_void = bir_type_void(M);
    uint32_t t_i32  = bir_type_int(M, 32);
    uint32_t t_f32  = bir_type_float(M, 32);
    uint32_t t_pg   = bir_type_ptr(M, t_f32, BIR_AS_GLOBAL);

    uint32_t fparams[] = { t_pg, t_pg, t_pg, t_i32 };
    uint32_t t_fn = bir_type_func(M, t_void, fparams, 4);

    uint32_t s_name = bir_add_string(M, "testKernel", 10);

    /* One function, one block, one instruction */
    uint32_t fi = M->num_funcs++;
    bir_func_t *F = &M->funcs[fi];
    memset(F, 0, sizeof(*F));
    F->name = s_name;
    F->type = t_fn;
    F->cuda_flags = 0x0001; /* CUDA_GLOBAL */
    F->num_params = 4;
    F->first_block = M->num_blocks;
    F->num_blocks = 1;

    uint32_t s_entry = bir_add_string(M, "entry", 5);
    uint32_t bi = M->num_blocks++;
    M->blocks[bi].name = s_entry;
    M->blocks[bi].first_inst = M->num_insts;

    /* ret void */
    bir_inst_t *I = &M->insts[M->num_insts++];
    memset(I, 0, sizeof(*I));
    I->op = BIR_RET;
    I->type = t_void;
    M->blocks[bi].num_insts = 1;
    F->total_insts = 1;

    /* Print to /dev/null -- we just want no crash */
    FILE *devnull = fopen("/dev/null", "w");
    if (!devnull) devnull = fopen("NUL", "w");
    if (devnull) {
        bir_print_module(M, devnull);
        fclose(devnull);
    }

    /* Suppress unused */
    (void)t_fn; (void)s_entry;

    PASS();
}
TH_REG("typ", 5, "the printer does not segfault", typ05)
