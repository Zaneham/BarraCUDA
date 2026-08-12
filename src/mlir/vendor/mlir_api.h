// Public MLIR C API declarations without exposing internal data structures.
// Implementations can back these APIs with different MLIR representations.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/string.h>
#include <base/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Handle types - opaque integer handles replacing raw pointers
// -----------------------------------------------------------------------------

typedef uintptr_t MLIR_OpHandle;
typedef uintptr_t MLIR_RegionHandle;
typedef uintptr_t MLIR_BlockHandle;
typedef uintptr_t MLIR_ValueHandle;
typedef uintptr_t MLIR_TypeHandle;
typedef uintptr_t MLIR_AttributeHandle;
typedef uintptr_t MLIR_LocationHandle;

#define MLIR_INVALID_HANDLE 0

typedef struct MLIR_Context {
    Arena *arena;
    // When true, MLIR_CreateOp* skips building the per-operand def-use
    // chains (UseNode allocations + the parallel operand_uses /
    // successor_operand_uses arrays). The def-use machinery is only needed
    // by the cf->scf and lower-to-LLVM passes; the wasm -> wasmstack ->
    // wasmssa -> llvm -> aarch64 -> macho pipeline never queries uses, so
    // disabling it there removes a large per-op memory overhead. Defaults
    // to false (tracking on) so all existing callers are unaffected.
    bool no_def_use_tracking;
    // When non-NULL, the process-wide type/struct intern registry allocates
    // its IR_Type objects AND its handle arrays into this arena instead of
    // `arena`. The streaming llvm->aarch64->macho backend lowers each function
    // into a throwaway temp arena (swapping `arena`) but must keep interned
    // types — which are cached in the global registry across functions — in a
    // persistent arena so cached handles never dangle after a temp arena is
    // released. Defaults to NULL (= use `arena`, no behaviour change).
    Arena *type_arena;
} MLIR_Context;

typedef struct MLIR_LocationMap {
    void *impl;
} MLIR_LocationMap;

size_t MLIR_GetLocationMapSize(const MLIR_LocationMap *location_map);
size_t MLIR_CollectLocationMap(const MLIR_LocationMap *location_map, string *out_keys, MLIR_LocationHandle *out_locs, size_t max);

// Vector helpers used by the parser implementation
DEFINE_VECTOR_FOR_TYPE(MLIR_OpHandle, VecOp)
DEFINE_VECTOR_FOR_TYPE(MLIR_ValueHandle, VecValue)
DEFINE_VECTOR_FOR_TYPE(MLIR_BlockHandle, VecBlock)
DEFINE_VECTOR_FOR_TYPE(MLIR_AttributeHandle, VecAttribute)

// -----------------------------------------------------------------------------
// Operation kinds
// -----------------------------------------------------------------------------

typedef enum {
    // Core ops
    OP_TYPE_UNREGISTERED = 0,
    OP_TYPE_MODULE,

    // Arithmetic dialect
    OP_TYPE_ARITH_ADDI,
    OP_TYPE_ARITH_SUBI,
    OP_TYPE_ARITH_MULI,
    OP_TYPE_ARITH_DIVI,
    OP_TYPE_ARITH_ADDF,
    OP_TYPE_ARITH_SUBF,
    OP_TYPE_ARITH_MULF,
    OP_TYPE_ARITH_DIVF,
    OP_TYPE_ARITH_CONSTANT,
    OP_TYPE_ARITH_CMPI,
    OP_TYPE_ARITH_CMPF,
    OP_TYPE_ARITH_SELECT,
    OP_TYPE_ARITH_BITCAST,
    OP_TYPE_ARITH_SITOFP,
    OP_TYPE_ARITH_FPTOSI,
    OP_TYPE_ARITH_INDEX_CAST,
    OP_TYPE_ARITH_EXTSI,
    OP_TYPE_ARITH_TRUNCI,
    OP_TYPE_ARITH_EXTF,
    OP_TYPE_ARITH_TRUNCF,
    OP_TYPE_ARITH_EXTUI,
    OP_TYPE_ARITH_MAXF,
    OP_TYPE_ARITH_DIVSI,
    OP_TYPE_ARITH_REMSI,
    OP_TYPE_ARITH_DIVUI,
    OP_TYPE_ARITH_REMUI,
    OP_TYPE_ARITH_SHRUI,
    OP_TYPE_ARITH_ORI,
    OP_TYPE_ARITH_MINSI,
    OP_TYPE_ARITH_ANDI,

    // Math dialect
    OP_TYPE_MATH_EXP,
    OP_TYPE_MATH_LOG,

    // Memory dialect
    OP_TYPE_MEMREF_LOAD,
    OP_TYPE_MEMREF_STORE,
    OP_TYPE_MEMREF_ALLOC,
    OP_TYPE_MEMREF_DEALLOC,

    // Control flow
    OP_TYPE_CF_BR,
    OP_TYPE_CF_COND_BR,
    OP_TYPE_CF_SWITCH,

    // Function dialect
    OP_TYPE_FUNC_FUNC,
    OP_TYPE_FUNC_RETURN,
    OP_TYPE_FUNC_CALL,
    OP_TYPE_FUNC_CALL_INDIRECT,
    OP_TYPE_FUNC_CONSTANT,
    OP_TYPE_UNREALIZED_CONVERSION_CAST,

    // SCF dialect
    OP_TYPE_SCF_FOR,
    OP_TYPE_SCF_WHILE,
    OP_TYPE_SCF_IF,
    OP_TYPE_SCF_YIELD,
    OP_TYPE_SCF_CONDITION,
    OP_TYPE_SCF_INDEX_SWITCH,

    // Triton dialect
    OP_TYPE_TT_GET_PROGRAM_ID,
    OP_TYPE_TT_LOAD,
    OP_TYPE_TT_STORE,
    OP_TYPE_TT_MAKE_RANGE,
    OP_TYPE_TT_SPLAT,
    OP_TYPE_TT_ADDPTR,
    OP_TYPE_TT_RETURN,
    OP_TYPE_TT_FUNC,
    OP_TYPE_TT_CALL,
    OP_TYPE_TT_REDUCE,
    OP_TYPE_TT_BROADCAST,
    OP_TYPE_TT_EXPAND_DIMS,
    OP_TYPE_TT_DOT,
    OP_TYPE_TT_PURE_EXTERN_ELEMENTWISE,

    // GPU dialect
    OP_TYPE_GPU_LAUNCH,

    // Affine dialect
    OP_TYPE_AFFINE_FOR,
    OP_TYPE_AFFINE_LOAD,

    // Vector dialect
    OP_TYPE_VECTOR_PRINT,

    // Standard dialect
    OP_TYPE_STD_CONSTANT,
    OP_TYPE_STD_RETURN,

    // Tensor dialect
    OP_TYPE_TENSOR_EXTRACT,
    OP_TYPE_TENSOR_SPLAT,
    OP_TYPE_TENSOR_COLLAPSE_SHAPE,

    // Linalg dialect
    OP_TYPE_LINALG_FILL,
    OP_TYPE_LINALG_COPY,

    // Index dialect
    OP_TYPE_INDEX_CONSTANT,

    // LLVM dialect
    OP_TYPE_LLVM_MLIR_UNDEF,
    OP_TYPE_LLVM_ALLOCA,
    OP_TYPE_LLVM_LOAD,
    OP_TYPE_LLVM_STORE,
    OP_TYPE_LLVM_GEP,
    OP_TYPE_LLVM_MLIR_ZERO,
    OP_TYPE_LLVM_MLIR_CONSTANT,
    OP_TYPE_LLVM_ICMP,
    OP_TYPE_LLVM_MLIR_ADDRESSOF,
    OP_TYPE_LLVM_MLIR_GLOBAL,
    OP_TYPE_LLVM_RETURN,
    OP_TYPE_LLVM_PTRTOINT,
    OP_TYPE_LLVM_FUNC,
    OP_TYPE_LLVM_CALL,
    OP_TYPE_LLVM_SEXT,
    OP_TYPE_LLVM_ADD,
    OP_TYPE_LLVM_SUB,
    OP_TYPE_LLVM_MUL,
    OP_TYPE_LLVM_SDIV,
    OP_TYPE_LLVM_UDIV,
    OP_TYPE_LLVM_SREM,
    OP_TYPE_LLVM_UREM,
    OP_TYPE_LLVM_AND,
    OP_TYPE_LLVM_OR,
    OP_TYPE_LLVM_XOR,
    OP_TYPE_LLVM_SHL,
    OP_TYPE_LLVM_LSHR,
    OP_TYPE_LLVM_ASHR,
    OP_TYPE_LLVM_TRUNC,
    OP_TYPE_LLVM_ZEXT,
    OP_TYPE_ARITH_XORI,
    OP_TYPE_ARITH_SHLI,
    OP_TYPE_ARITH_SHRSI,

    // Return operations
    OP_TYPE_RETURN,
    OP_TYPE_TT_REDUCE_RETURN,

    // -------------------------------------------------------------------------
    // wasmssa dialect — high-level SSA-form WebAssembly ops. Produced by
    // the LLVM-dialect -> wasmssa lowering pass and consumed by the
    // wasmssa -> wasmstack stackification pass. All values are SSA;
    // there are no explicit local.get / local.set / local.tee here.
    //
    // Op operand-order convention (non-commutative ops):
    //   sub:   %r = wasmssa.sub %lhs, %rhs   ; emits "lhs rhs i32.sub"
    //   load:  %r = wasmssa.load %addr       ; offset/align as attrs
    //   store: wasmssa.store %addr, %val
    // -------------------------------------------------------------------------
    OP_TYPE_WASMSSA_FUNC,
    OP_TYPE_WASMSSA_IMPORT_FUNC,
    OP_TYPE_WASMSSA_IMPORT_GLOBAL,
    OP_TYPE_WASMSSA_CONST,
    OP_TYPE_WASMSSA_ADD,
    OP_TYPE_WASMSSA_SUB,
    OP_TYPE_WASMSSA_BINOP,
    OP_TYPE_WASMSSA_UNOP,
    OP_TYPE_WASMSSA_LOAD,
    OP_TYPE_WASMSSA_STORE,
    OP_TYPE_WASMSSA_GLOBAL_GET,
    OP_TYPE_WASMSSA_GLOBAL_SET,
    OP_TYPE_WASMSSA_EXTEND_I32_S,
    OP_TYPE_WASMSSA_RETURN,
    OP_TYPE_WASMSSA_CALL,
    // Region-bearing structured control flow. Cross-CF value passing is
    // modelled via MLIR block arguments on the region's entry block (for
    // loops) and via the variadic operands of block_return / br (for all
    // three op kinds). Results of the region-bearing op carry the values
    // that flow out via fall-through or `br` to its label.
    //
    //   %r:R* = wasmssa.block ()           : (R*)  { ... }
    //   %r:R* = wasmssa.loop  (init:T*)    : (R*)  { ^entry(a:T*): ... }
    //   %r:R* = wasmssa.if    (%cond:i32)  : (R*)  { then } else { else }
    //
    // Region terminators: block_return | br | br_if (non-terminator) |
    //                     unreachable | return.
    OP_TYPE_WASMSSA_BLOCK,
    OP_TYPE_WASMSSA_LOOP,
    OP_TYPE_WASMSSA_IF,
    // Region terminator carrying the values that flow out of the enclosing
    // wasmssa.block / wasmssa.loop / wasmssa.if via fall-through.
    OP_TYPE_WASMSSA_BLOCK_RETURN,
    // Terminator emitting wasm `unreachable` (0x00). Used after wasmssa.loop
    // when the loop never falls through (every back-edge `br 0` or outward
    // `br N` exits the loop's own end).
    OP_TYPE_WASMSSA_UNREACHABLE,
    // Branch carrying variadic values matching the target label's signature
    // (depth-relative). Terminator.
    OP_TYPE_WASMSSA_BR,
    // Conditional branch carrying only the condition; no value operands.
    // For value-carrying conditional branches we use `wasmssa.if %cond { br N (vals) }`.
    OP_TYPE_WASMSSA_BR_IF,
    OP_TYPE_WASMSSA_SELECT,
    OP_TYPE_WASMSSA_EQZ,
    OP_TYPE_WASMSSA_ADDRESSOF,
    // Function-pointer support: FUNC_ADDR pushes the table-index of a
    // named function (lowered via R_WASM_TABLE_INDEX_SLEB); CALL_INDIRECT
    // pops args + table-index and dispatches via wasm `call_indirect`.
    OP_TYPE_WASMSSA_FUNC_ADDR,
    OP_TYPE_WASMSSA_CALL_INDIRECT,
    // Wasm function-locals access. Each function has a fixed set of
    // typed locals (params + declared locals). LOCAL_GET reads slot
    // `local_idx`, LOCAL_SET writes it. Locals are NEVER address-taken
    // in wasm semantics, so they live in a native (non-linmem) stack
    // frame that the llvm->aarch64 backend allocates. The lifter used
    // to spill them into linmem cells (8 bytes each), which inflated
    // the wasm stack by ~20x and forced unrealistic stack-size grows.
    //   attrs: valtype (i32 byte: i32|i64|f32|f64), local_idx (i32).
    OP_TYPE_WASMSSA_LOCAL_GET,
    OP_TYPE_WASMSSA_LOCAL_SET,
    // WebAssembly memory instructions: `memory.size` (0x3F 0x00) and
    // `memory.grow` (0x40 0x00). MEMORY_SIZE has no operands and an i32
    // result (size in pages). MEMORY_GROW takes one i32 operand (pages
    // to grow by) and returns the previous size in pages (or -1 on
    // failure) as i32. Both carry a trailing single-byte immediate
    // identifying the memory index (always 0 for the default memory).
    OP_TYPE_WASMSSA_MEMORY_SIZE,
    OP_TYPE_WASMSSA_MEMORY_GROW,

    // -------------------------------------------------------------------------
    // wasmstack dialect — low-level stack-machine WebAssembly ops. 1:1
    // with the wasm bytecode opcodes. Produced by the wasmssa -> wasmstack
    // stackification pass and consumed by the binary emitter.
    // -------------------------------------------------------------------------
    OP_TYPE_WASMSTACK_FUNC,
    OP_TYPE_WASMSTACK_IMPORT_FUNC,
    OP_TYPE_WASMSTACK_IMPORT_GLOBAL,
    OP_TYPE_WASMSTACK_LOCAL_GET,
    OP_TYPE_WASMSTACK_LOCAL_SET,
    OP_TYPE_WASMSTACK_LOCAL_TEE,
    OP_TYPE_WASMSTACK_CONST,
    OP_TYPE_WASMSTACK_ADD,
    OP_TYPE_WASMSTACK_SUB,
    OP_TYPE_WASMSTACK_BINOP,
    OP_TYPE_WASMSTACK_UNOP,
    OP_TYPE_WASMSTACK_LOAD,
    OP_TYPE_WASMSTACK_STORE,
    OP_TYPE_WASMSTACK_GLOBAL_GET,
    OP_TYPE_WASMSTACK_GLOBAL_SET,
    OP_TYPE_WASMSTACK_EXTEND_I32_S,
    OP_TYPE_WASMSTACK_RETURN,
    OP_TYPE_WASMSTACK_CALL,
    // Structured-CF + select.
    OP_TYPE_WASMSTACK_BLOCK,
    OP_TYPE_WASMSTACK_LOOP,
    OP_TYPE_WASMSTACK_IF,
    OP_TYPE_WASMSTACK_ELSE,
    OP_TYPE_WASMSTACK_END,
    OP_TYPE_WASMSTACK_BR,
    OP_TYPE_WASMSTACK_BR_IF,
    // Terminator emitting wasm `unreachable` (0x00).
    OP_TYPE_WASMSTACK_UNREACHABLE,
    OP_TYPE_WASMSTACK_SELECT,
    OP_TYPE_WASMSTACK_EQZ,
    OP_TYPE_WASMSTACK_ADDRESSOF,
    OP_TYPE_WASMSTACK_FUNC_ADDR,
    OP_TYPE_WASMSTACK_CALL_INDIRECT,
    // 1:1 wasmstack mirrors of OP_TYPE_WASMSSA_MEMORY_{SIZE,GROW}.
    OP_TYPE_WASMSTACK_MEMORY_SIZE,
    OP_TYPE_WASMSTACK_MEMORY_GROW,
    // `drop` (wasm 0x1a): pop one stack value, no result. Only used by
    // the wasm -> wasmstack lifter; the C-frontend wasmssa pipeline
    // never emits this op.
    OP_TYPE_WASMSTACK_DROP,
    // `br_table`: emitted by clang for switch dispatch in linked
    // runtime objects. Carries a comma-separated list of target depths
    // plus a default depth.
    OP_TYPE_WASMSTACK_BR_TABLE,
    // Module-level data segment lifted from a wasm DATA section. The
    // wasmstack -> wasmssa pass converts this into a wasmssa.import_global
    // sized appropriately.
    OP_TYPE_WASMSTACK_DATA_SEGMENT,
    // Module-level global declaration with an initial value. Lifted
    // from the wasm GLOBAL section. The backend allocates a static
    // i64 slot per global; only the initial value matters.
    OP_TYPE_WASMSTACK_GLOBAL_DECL,

    // Module-level table entry lifted from a wasm ELEM section. Carries
    // (slot, target) — i.e. table_slot_index N now contains a reference
    // to function `target`. The wasmstack -> wasmssa pass collects these
    // and emits a synthetic wasmssa.func "_tinyc_fnptr_init" containing
    // one wasmssa.func_addr per entry, so the lifter pre-pass sees them.
    OP_TYPE_WASMSTACK_FUNC_ADDR_DECL,

    // -------------------------------------------------------------------------
    // aarch64 dialect — 1:1 with the AArch64 instruction encoding. The
    // `aarch64 → Mach-O` backend is a "dumb" byte emitter; all isel /
    // register-allocation knowledge lives in the `llvm → aarch64` lowering.
    // First-light slice: just enough to run `int main() { return 42; }`
    // and have its return value become the process exit code via a
    // direct `svc #0x80` Mach syscall (no `proc_exit` shim required).
    // -------------------------------------------------------------------------
    OP_TYPE_AARCH64_FUNC,
    OP_TYPE_AARCH64_MOVZ,   // movz Wd|Xd, #imm16, LSL #(hw*16)
    OP_TYPE_AARCH64_MOVK,   // movk Wd|Xd, #imm16, LSL #(hw*16)
    OP_TYPE_AARCH64_MOV_X,  // mov Xd, Xn  (register move; X-form)
    OP_TYPE_AARCH64_BL,     // bl <symbol>  (branch-and-link, PC-relative)
    OP_TYPE_AARCH64_BLR,    // blr Xn  (indirect branch-and-link via register)
    OP_TYPE_AARCH64_SVC,    // svc #imm16
    OP_TYPE_AARCH64_RET,    // ret (== ret x30)

    // Arithmetic + memory + stack-frame ops added in the arith slice.
    OP_TYPE_AARCH64_ADD_IMM,  // add Wd|Xd, Wn|Xn, #imm12 (LSL 0)
    OP_TYPE_AARCH64_SUB_IMM,  // sub Wd|Xd, Wn|Xn, #imm12 (LSL 0)
    OP_TYPE_AARCH64_ADD_REG,  // add Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_SUB_REG,  // sub Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_MUL,      // mul Wd|Xd, Wn|Xn, Wm|Xm  (== madd ..., xzr)
    OP_TYPE_AARCH64_SDIV,     // sdiv Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_UDIV,     // udiv Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_MSUB,     // msub Wd, Wn, Wm, Wa  (used for srem/urem)
    OP_TYPE_AARCH64_AND_REG,  // and  Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_AND_IMM,  // and  Wd|Xd, Wn|Xn, #(1<<w)-1  (low-w-bits mask)
    OP_TYPE_AARCH64_ORR_REG,  // orr  Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_EOR_REG,  // eor  Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_LSL_REG,  // lslv Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_LSR_REG,  // lsrv Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_ASR_REG,  // asrv Wd|Xd, Wn|Xn, Wm|Xm
    OP_TYPE_AARCH64_LSL_IMM,  // lsl Wd|Xd, Wn|Xn, #shift  (UBFM alias)
    OP_TYPE_AARCH64_LSR_IMM,  // lsr Wd|Xd, Wn|Xn, #shift  (UBFM alias)
    OP_TYPE_AARCH64_ASR_IMM,  // asr Wd|Xd, Wn|Xn, #shift  (SBFM alias)
    OP_TYPE_AARCH64_SXTW,     // sxtw Xd, Wn  (alias for SBFM)
    OP_TYPE_AARCH64_SXTB,     // sxtb Wd|Xd, Wn  (sign-extend low byte)
    OP_TYPE_AARCH64_SXTH,     // sxth Wd|Xd, Wn  (sign-extend low half)
    OP_TYPE_AARCH64_UXTW,     // mov  Wd, Wn  (== orr Wd, WZR, Wn ; zero-extends)
    OP_TYPE_AARCH64_LDR_W,    // ldr Wd, [Xn, #imm] (i32 unsigned-offset)
    OP_TYPE_AARCH64_STR_W,    // str Wd, [Xn, #imm] (i32 unsigned-offset)
    OP_TYPE_AARCH64_LDR_X,    // ldr Xd, [Xn, #imm] (i64 unsigned-offset)
    OP_TYPE_AARCH64_STR_X,    // str Xd, [Xn, #imm] (i64 unsigned-offset)
    OP_TYPE_AARCH64_STRB_IMM, // strb Wt, [Xn, #imm12] (1-byte unscaled)
    OP_TYPE_AARCH64_LDRB_IMM, // ldrb Wt, [Xn, #imm12] (1-byte zero-extend)
    OP_TYPE_AARCH64_LDR_W_REG,  // ldr  Wt, [Xn, Xm, LSL #0]
    OP_TYPE_AARCH64_STR_W_REG,  // str  Wt, [Xn, Xm, LSL #0]
    OP_TYPE_AARCH64_LDR_X_REG,  // ldr  Xt, [Xn, Xm, LSL #0]
    OP_TYPE_AARCH64_STR_X_REG,  // str  Xt, [Xn, Xm, LSL #0]
    OP_TYPE_AARCH64_LDRB_REG,   // ldrb Wt, [Xn, Xm, LSL #0]
    OP_TYPE_AARCH64_STRB_REG,   // strb Wt, [Xn, Xm, LSL #0]
    // Pseudo: ADRP+ADD pair that resolves to the runtime base address
    // of one of the predeclared __DATA regions (vmctx, globals, linmem).
    // The macho backend patches the encoded PC-relative immediates
    // after layout. Body of the macho-backend tracks this list of
    // bases as part of the data-segment layout.
    OP_TYPE_AARCH64_ADRP_DATA,    // adrp Xd, page_of(<region>)
    OP_TYPE_AARCH64_ADD_DATA_LO,  // add  Xd, Xn, #lo12(<region>)
    // Function prologue/epilogue (modeled as an op pair so the dumb
    // backend can emit the exact instruction sequence; details of
    // sp/fp/lr handling baked into the encoder).
    OP_TYPE_AARCH64_PROLOGUE,
    OP_TYPE_AARCH64_EPILOGUE,
    // Comparison + condition-set. Used to materialise i32 booleans for
    // `llvm.icmp` and eqz results.
    OP_TYPE_AARCH64_CMP_REG,    // cmp Wn, Wm  (== subs Wzr, Wn, Wm)
    OP_TYPE_AARCH64_CMP_IMM,    // cmp Wn, #imm12  (== subs Wzr, Wn, #imm12)
    OP_TYPE_AARCH64_CSET,       // cset Wd, COND (== csinc Wd, Wzr, Wzr, invert(COND))
    // Conditional select. `csel Wd, Wn, Wm, COND`. Used for llvm.select.
    OP_TYPE_AARCH64_CSEL,
    // Control-flow ops. `target` attribute is the symbolic label name.
    // The macho backend tracks all `aarch64.label` positions inside a
    // function and patches the branch immediates after layout.
    OP_TYPE_AARCH64_B,          // b <label>     (PC-relative imm26)
    OP_TYPE_AARCH64_B_COND,     // b.<cond> <label> (PC-relative imm19)
    OP_TYPE_AARCH64_CBZ,        // cbz Wn, <label>
    OP_TYPE_AARCH64_CBNZ,       // cbnz Wn, <label>
    OP_TYPE_AARCH64_LABEL,      // pseudo: marks a position; emits 0 bytes
    OP_TYPE_AARCH64_BRK,        // brk #imm16 (trap)
    // Floating-point ops. Operand and result registers in attributes are
    // *V register* numbers (0..31) for the FP slots and GP register
    // numbers for the GP slots; the two register files are disjoint, so
    // a number like `0` means V0 in an FP slot and W0/X0 in a GP slot.
    //
    // FMOV between a GP and an FP register; `dir_to_v` chooses direction
    // (true: GP→V, false: V→GP). `sf` selects W↔S (false) or X↔D (true).
    //   attrs: dir_to_v (bool), sf (bool), rd (i32), rn (i32).
    OP_TYPE_AARCH64_FMOV_GP_V,
    // FADD/FSUB/FMUL/FDIV between two V regs into a V reg.
    //   attrs: kind (string: fadd|fsub|fmul|fdiv), fwidth (32|64),
    //          rd (i32), rn (i32), rm (i32).
    OP_TYPE_AARCH64_FP_BINOP,
    // FNEG/FABS/FSQRT on a V reg → V reg.
    //   attrs: kind (string), fwidth, rd, rn.
    OP_TYPE_AARCH64_FP_UNOP,
    // FCMP Sn/Dn, Sm/Dm — sets NZCV (no rd).
    //   attrs: fwidth (32|64), rn (i32), rm (i32).
    OP_TYPE_AARCH64_FCMP,
    // FP conversion family. `kind` is one of:
    //   "f2f"   — FP-to-FP precision change (FCVT Dd,Sn / FCVT Sd,Dn).
    //             rd and rn are both V registers.
    //   "f2i"   — FP-to-integer (FCVTZS / FCVTZU). `sign` true => FCVTZS,
    //             false => FCVTZU. rd is a GP register, rn is V.
    //   "i2f"   — Integer-to-FP (SCVTF / UCVTF). rd is V, rn is GP.
    // src_w / dst_w in {32, 64} pick S/W vs D/X.
    //   attrs: kind (string), src_w (i32), dst_w (i32), sign (bool),
    //          rd (i32), rn (i32).
    OP_TYPE_AARCH64_FP_CVT,
    // Module-level: a slice of bytes to overlay onto the linmem __DATA
    // section at the given offset. The macho backend gathers all of
    // these into a single file-backed section that precedes the
    // zero-fill portion of linmem.
    //   attrs: sym_name (string, debug), offset (i32), init_data (string).
    OP_TYPE_AARCH64_DATA_INIT,

    OP_TYPE_COUNT
} MLIR_OpType;

string MLIR_MLIR_OpTypeToString(MLIR_OpType type);

// -----------------------------------------------------------------------------
// API lifecycle
// -----------------------------------------------------------------------------

void MLIR_InitApi(MLIR_Context *ctx, MLIR_OpHandle root);
void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena);
Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx);
// Drops process-wide intern/registry state (type & struct dedup caches and the
// def-use op registry). Multi-pass pipelines call this right before switching
// to a fresh arena and freeing the previous one. See implementation comment.
void MLIR_ResetInternRegistry(void);

// -----------------------------------------------------------------------------
// Operation API
// -----------------------------------------------------------------------------

// Creation & structural mutation
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
    int64_t source_line_start);
// Like MLIR_CreateOp, but with explicit successor blocks and per-successor
// operand arrays. Used by branch ops (cf.br, cf.cond_br, ...). The successor
// blocks must be valid block handles (forward-referenced blocks are fine as
// long as MLIR_CreateBlock has been called for them).
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
    int64_t source_line_start);
void MLIR_AppendOpAttribute(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttributeHandle attr);

// Print an operation. Three named entry points instead of an enum-dispatched
// MLIR_PrintOperation: each lives in its own translation unit so the linker
// pulls in only what each binary actually calls. The Upstream variant
// requires the upstream backend (the native backend's stub returns an
// "error: ..." string).
string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle op);
string MLIR_PrintOperationClassic(MLIR_Context *ctx, MLIR_OpHandle op);
string MLIR_PrintOperationGeneric(MLIR_Context *ctx, MLIR_OpHandle op);

// Parse a top-level MLIR module from text. Returns MLIR_INVALID_HANDLE on
// failure (or, for Upstream on the native backend, always).
MLIR_OpHandle MLIR_ParseTextUpstream(MLIR_Context *ctx, string text);
MLIR_OpHandle MLIR_ParseTextClassic(MLIR_Context *ctx, string text);

// Lowering / translation entry points.
//
// Two parallel families:
//
//   - Backend-agnostic functions (MLIR_LowerToLLVMDialect{,ForWasm},
//     MLIR_TranslateModuleTo{LLVMIR,Wasm}) are implemented in dedicated
//     translation units (mlir_lower_to_llvm.c,
//     mlir_translate_to_llvm_ir.c, mlir_translate_to_wasm.c) that use
//     only this very header — no upstream MLIR types. The same .o files
//     link into both the native and upstream tinyc builds, so these
//     functions are always available and produce identical output
//     regardless of which core API implementation is linked in.
//
//   - *Upstream siblings (MLIR_LowerToLLVMDialect{,ForWasm}Upstream,
//     MLIR_TranslateModuleTo{LLVMIR,Wasm}Upstream, plus
//     MLIR_PrintOperationUpstream and MLIR_ParseTextUpstream above) are
//     implemented only by mlir_api_impl_upstream.cpp. They run upstream
//     MLIR's pass pipeline / translator / LLVM target machine and are
//     therefore unavailable in the native-only build (calls fail at
//     link time). They exist so callers can run upstream side-by-side
//     with the in-tree implementation for reference, performance, or
//     correctness comparisons.
//
// All entry points print diagnostics to stderr on failure.

// Lower a `builtin.module` op to the LLVM dialect by running the
// in-tree conversion (scf->cf, then arith/memref/cf/func/vector ->
// llvm). Mutates `module` in place. Returns true on success.
bool MLIR_LowerToLLVMDialect(MLIR_Context *ctx, MLIR_OpHandle module);

// Same as MLIR_LowerToLLVMDialect, but tailored to feed the in-tree
// LLVM-dialect-MLIR -> WASM emitter (mlir_translate_to_wasm.c).
// Instead of running scf->cf (which destroys structured control flow
// that wasm needs), this entry point first lifts cf back to scf and
// then lowers arith/memref/func/vector/etc. to LLVM, leaving scf.* ops
// in place for stage 1 of the wasm pipeline to consume directly.
bool MLIR_LowerToLLVMDialectForWasm(MLIR_Context *ctx, MLIR_OpHandle module);

// Translate a `builtin.module` op already lowered to the LLVM dialect
// into LLVM IR text (`.ll`). Returns the IR text on success or an
// empty string on failure.
string MLIR_TranslateModuleToLLVMIR(MLIR_Context *ctx, MLIR_OpHandle module);

// Translate a `builtin.module` op already lowered to the LLVM dialect
// into a WebAssembly relocatable object file (wasm32-wasi). Returns
// the raw object bytes on success or an empty string on failure. The
// returned bytes are NOT a runnable wasm module — they still need to
// be linked (typically with `wasm-ld`) against any runtime/imports the
// program uses, producing a final `.wasm` module that can be run by
// wasmtime. This implementation chains the in-tree native pipeline
// (mlir_llvm_to_wasmssa -> mlir_wasmssa_to_wasmstack ->
// mlir_wasmstack_to_bin).
string MLIR_TranslateModuleToWasm(MLIR_Context *ctx, MLIR_OpHandle module);

// Upstream-only siblings. See the family description above.
bool MLIR_LowerToLLVMDialectUpstream(MLIR_Context *ctx, MLIR_OpHandle module);
bool MLIR_LowerToLLVMDialectForWasmUpstream(MLIR_Context *ctx, MLIR_OpHandle module);
string MLIR_TranslateModuleToLLVMIRUpstream(MLIR_Context *ctx, MLIR_OpHandle module);
string MLIR_TranslateModuleToWasmUpstream(MLIR_Context *ctx, MLIR_OpHandle module);

// Accessors
MLIR_OpType MLIR_GetOpType(MLIR_OpHandle op);
string MLIR_GetOpName(MLIR_OpHandle op);
string MLIR_GetOpName_string(MLIR_OpHandle op);
MLIR_LocationHandle MLIR_GetOpLocation(MLIR_OpHandle op);
string MLIR_GetOpTrailingComment(MLIR_OpHandle op);
int64_t MLIR_GetOpSourceLineStart(MLIR_OpHandle op);
MLIR_LocationHandle MLIR_GetOpUnnumberedLocationDef(MLIR_OpHandle op);
size_t MLIR_GetOpNumOperands(MLIR_OpHandle op);
MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumResults(MLIR_OpHandle op);
MLIR_ValueHandle MLIR_GetOpResult(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumResultTypes(MLIR_OpHandle op);
MLIR_TypeHandle MLIR_GetOpResult_type(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumAttributes(MLIR_OpHandle op);
MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle op, size_t idx);
// Like MLIR_GetOpAttribute, but looks up by name including inherent
// attributes that are stored in the op's Properties (e.g. LLVM::CallOp
// `callee` and `var_callee_type`). Returns MLIR_INVALID_HANDLE on miss.
MLIR_AttributeHandle MLIR_GetOpAttributeByName(MLIR_OpHandle op, const char *name);
size_t MLIR_GetOpNumRegions(MLIR_OpHandle op);
MLIR_RegionHandle MLIR_GetOpRegion(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumSuccessors(MLIR_OpHandle op);
MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumSuccessorOperands(MLIR_OpHandle op, size_t succ_idx);
MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_OpHandle op, size_t succ_idx, size_t op_idx);

// -----------------------------------------------------------------------------
// Region API
// -----------------------------------------------------------------------------

MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *ctx);
void MLIR_AppendRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle region, MLIR_BlockHandle block);
size_t MLIR_GetRegionNumBlocks(MLIR_RegionHandle region);
MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_RegionHandle region, size_t idx);

// -----------------------------------------------------------------------------
// Block API
// -----------------------------------------------------------------------------

MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *ctx);
void MLIR_AppendBlockOp(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op);
void MLIR_InsertBlockOpBeforeTerminator(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op);
// Insert `op` at position `idx` in `block`'s op list, shifting later ops
// down by one. `idx` is clamped to [0, n_ops].
void MLIR_InsertBlockOpAtIndex(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op, size_t idx);
void MLIR_AppendBlockArg(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_ValueHandle arg);
size_t MLIR_GetBlockNumOps(MLIR_BlockHandle block);
MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle block, size_t idx);
size_t MLIR_GetBlockNumArgs(MLIR_BlockHandle block);
MLIR_ValueHandle MLIR_GetBlockArg(MLIR_BlockHandle block, size_t idx);
// Returns the 0-indexed position of `block` within its parent region.
// Returns SIZE_MAX if the block has no parent or the backend doesn't track it.
size_t MLIR_GetBlockIndex(MLIR_BlockHandle block);

// -----------------------------------------------------------------------------
// Value API
// -----------------------------------------------------------------------------

// Value kind used for SSA values
typedef enum MLIR_ValueKind {
    BLOCK_ARG,
    OP_RESULT
} MLIR_ValueKind;

// Creation
MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *ctx, string register_name, uint32_t result_index, MLIR_TypeHandle type, MLIR_LocationHandle location);
MLIR_ValueHandle MLIR_CreateValueOpResult(MLIR_Context *ctx, MLIR_OpHandle def, uint32_t result_index, MLIR_TypeHandle type, string register_name, MLIR_LocationHandle location);

// Accessors
MLIR_ValueKind MLIR_GetValueKind(MLIR_ValueHandle value);
MLIR_TypeHandle MLIR_GetValueType(MLIR_ValueHandle value);
string MLIR_GetValueRegisterName(MLIR_ValueHandle value);
uint32_t MLIR_GetValueResultIndex(MLIR_ValueHandle value);
MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_ValueHandle value);
MLIR_LocationHandle MLIR_GetValueLocation(MLIR_ValueHandle value);

// -----------------------------------------------------------------------------
// Type API
// -----------------------------------------------------------------------------

// Creation
MLIR_TypeHandle MLIR_CreateTypeInteger(MLIR_Context *ctx, uint32_t width, bool is_signed);
MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *ctx, uint32_t width, bool is_bfloat);
MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space);

// LLVM dialect types — used by frontends that emit `!llvm.ptr`,
// `!llvm.struct`, and `!llvm.array`. Identified structs are mutable: create
// with MLIR_CreateTypeLLVMStructIdentified, then call
// MLIR_SetTypeLLVMStructBody once. Recursion through `!llvm.ptr` (which is
// opaque) is fully supported because the body never mentions the recursive
// struct's body type.
MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *ctx, string name);
void            MLIR_SetTypeLLVMStructBody(MLIR_Context *ctx, MLIR_TypeHandle struct_ty,
                                           const MLIR_TypeHandle *fields, size_t n_fields);
MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *ctx, MLIR_TypeHandle elem, uint64_t count);

// LLVM-dialect global helpers. Each returns a freshly-created (unattached)
// op; the caller appends it to the module body. Implemented by the
// upstream backend; the native backend returns MLIR_INVALID_HANDLE.
//
// MLIR_CreateLLVMGlobalString: emits
//   llvm.mlir.global private constant @<sym>("<bytes>") : !llvm.array<N x i8>
//
// MLIR_CreateLLVMGlobal: emits
//   llvm.mlir.global internal @<sym>(<init>) : <elem_ty>
// Init shapes:
//   init_kind == 0  -> simple integer initializer (init_int)
//   init_kind == 1  -> simple float initializer (init_float)
//   init_kind == 2  -> region initializer: caller emits ops into
//                      *out_init_block and terminates with llvm.return.
//   init_kind == 3  -> no initializer (zero-initialized).
//   init_kind == 4  -> external linkage, no initializer (for `extern T x;`
//                      declarations resolved by the linker, e.g. `stdout`).
MLIR_OpHandle MLIR_CreateLLVMGlobalString(MLIR_Context *ctx,
                                          string sym_name,
                                          string bytes,
                                          MLIR_LocationHandle loc);
MLIR_OpHandle MLIR_CreateLLVMGlobal(MLIR_Context *ctx,
                                    string sym_name,
                                    MLIR_TypeHandle elem_ty,
                                    bool is_constant,
                                    int init_kind,
                                    int64_t init_int,
                                    double init_float,
                                    MLIR_BlockHandle *out_init_block,
                                    MLIR_LocationHandle loc);
// Aggregate-array global with a raw-bytes initializer:
//   llvm.mlir.global internal @<sym>("<bytes>") : !llvm.array<N x iX>
// `bytes` is the packed little-endian initializer (size must equal
// N * sizeof(elem)). Backends interpret it as raw bytes.
MLIR_OpHandle MLIR_CreateLLVMGlobalArrayInit(MLIR_Context *ctx,
                                              string sym_name,
                                              MLIR_TypeHandle array_ty,
                                              bool is_constant,
                                              string bytes,
                                              MLIR_LocationHandle loc);

MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name);
// Function type: (input_types) -> (result_types). Both arrays are
// copied; passing 0 inputs/results is allowed.
MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *ctx,
                                         const MLIR_TypeHandle *inputs, size_t n_inputs,
                                         const MLIR_TypeHandle *results, size_t n_results);

// LLVM-dialect function type (LLVMFunctionType). Differs from the standard
// FunctionType in two ways: it has at most one result (use the LLVM `void`
// type for "no result"), and it has an `is_var_arg` flag used to model C
// variadic functions like `int printf(const char *, ...)`. The result type
// must be either an LLVM-compatible scalar/pointer type or the LLVM void
// type (see MLIR_CreateTypeLLVMVoid).
MLIR_TypeHandle MLIR_CreateTypeLLVMFunction(MLIR_Context *ctx,
                                             MLIR_TypeHandle result,
                                             const MLIR_TypeHandle *inputs,
                                             size_t n_inputs,
                                             bool is_var_arg);
// LLVM `void` type — only valid as the result of an LLVMFunctionType.
MLIR_TypeHandle MLIR_CreateTypeLLVMVoid(MLIR_Context *ctx);

// Mutation
void MLIR_SetTypeIntegerProperties(MLIR_TypeHandle type, uint32_t width, bool is_signed);
void MLIR_SetTypeFloatProperties(MLIR_TypeHandle type, uint32_t width, bool is_bfloat);
void MLIR_SetTypeTensorProperties(MLIR_TypeHandle type, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
void MLIR_SetTypeMemrefProperties(MLIR_TypeHandle type, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
void MLIR_SetTypePointerProperties(MLIR_TypeHandle type, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space);

// Introspection & formatting
bool MLIR_IsTypeInteger(MLIR_TypeHandle type);
bool MLIR_IsTypeFloat(MLIR_TypeHandle type);
bool MLIR_IsTypeTensor(MLIR_TypeHandle type);
bool MLIR_IsTypeMemref(MLIR_TypeHandle type);
bool MLIR_IsTypePointer(MLIR_TypeHandle type);
bool MLIR_IsTypeIndex(MLIR_TypeHandle type);
bool MLIR_IsTypeUnknown(MLIR_TypeHandle type);
bool MLIR_IsTypeOpaque(MLIR_TypeHandle type);
bool MLIR_IsTypeFunction(MLIR_TypeHandle type);
size_t MLIR_GetTypeFunctionNumInputs(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_TypeHandle type, size_t idx);
size_t MLIR_GetTypeFunctionNumResults(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_TypeHandle type, size_t idx);
// True if the type is an `LLVMFunctionType` whose `isVarArg` flag is set.
// Returns false for the standard `FunctionType` (which is never variadic).
bool MLIR_GetTypeFunctionIsVarArg(MLIR_TypeHandle type);
// Element type of a tensor/memref/vector type, or invalid if the type is
// not a shaped type.
MLIR_TypeHandle MLIR_GetTypeShapedElement(MLIR_TypeHandle type);
// LLVM struct/array type introspection. Used by lowering passes that
// need to compute byte offsets and sizes (e.g. native LLVM->WASM).
bool            MLIR_IsTypeLLVMStruct(MLIR_TypeHandle type);
size_t          MLIR_GetTypeLLVMStructNumFields(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeLLVMStructField(MLIR_TypeHandle type, size_t idx);
bool            MLIR_IsTypeLLVMArray(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeLLVMArrayElement(MLIR_TypeHandle type);
uint64_t        MLIR_GetTypeLLVMArrayNumElements(MLIR_TypeHandle type);
string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle type);

// -----------------------------------------------------------------------------
// Attribute API
// -----------------------------------------------------------------------------

// Creation & mutation
//
// Integer and Float attributes carry an MLIR Type indicating their numeric
// type (e.g. i32, i64, index, f32, f64). The caller must construct that
// type first via the Type API. There is no untyped form: typing every
// numeric attribute eagerly is what allows ops like arith.constant to
// build proper IR (where `value`'s attribute type must equal the result
// type) without later fixups.
MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *ctx, string name, int64_t value, MLIR_TypeHandle type);
MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *ctx, string name, double value, MLIR_TypeHandle type);
MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *ctx, string name, bool value);
MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *ctx, string name, string value);
MLIR_AttributeHandle MLIR_CreateAttributeLLVMLinkageInternal(MLIR_Context *ctx, string name);
MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count);
// Dense i32 array attribute (DenseI32ArrayAttr) — used e.g. for
// `llvm.getelementptr`'s `rawConstantIndices`.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *ctx, string name,
                                                        const int32_t *values, size_t count);
// Dense i64 array attribute (DenseI64ArrayAttr) — used e.g. for
// `scf.index_switch`'s `cases` attribute.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI64Array(MLIR_Context *ctx, string name,
                                                        const int64_t *values, size_t count);
MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count);
// TypeAttr: an attribute that wraps a Type (e.g. func.func's
// `function_type` attribute, which wraps a FunctionType).
MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *ctx, string name, MLIR_TypeHandle type);

// SymbolRef attribute (e.g. `callee = @foo` on func.call). The `value`
// must be the bare symbol name (no leading `@`).
MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *ctx, string name, string value);

// -----------------------------------------------------------------------------
// IR mutation primitives (used by lowering passes).
// -----------------------------------------------------------------------------

// Replace every use of `old_value` with `new_value` everywhere in the
// module that contains it. Operand lists, successor operand lists, and
// any other places the value appears are updated. Both values must have
// compatible types (the API does not check). After this call the old
// value is unused and the op that produced it can typically be erased
// with MLIR_EraseOp.
//
// Implementation note: the upstream backend uses
// mlir::Value::replaceAllUsesWith. The native backend walks the module
// linearly. In both cases the op definitions and value handles
// themselves remain valid; only operand references move.
void MLIR_ReplaceAllUsesOfValue(MLIR_Context *ctx,
                                MLIR_ValueHandle old_value,
                                MLIR_ValueHandle new_value);

// Remove `op` from its parent block (if any) and detach it. The op is
// not freed (the API leaves storage to the arena/allocator). Operands
// to and results from `op` are not touched; callers should typically
// have already called MLIR_ReplaceAllUsesOfValue on each result.
void MLIR_EraseOp(MLIR_Context *ctx, MLIR_OpHandle op);

// Replace one of `op`'s regions with `region`. The previously attached
// region (if any) is detached and abandoned. Used when a lowering
// rebuilds a function body wholesale rather than mutating it in place.
void MLIR_SetOpRegion(MLIR_Context *ctx, MLIR_OpHandle op, size_t idx,
                      MLIR_RegionHandle region);

// Detach `op`'s region at `idx` and return it as a fresh handle. After
// this call, `op` has an empty region at `idx`. The returned handle can
// be passed as part of a new MLIR_CreateOp call (or to MLIR_SetOpRegion)
// to graft the body onto a replacement op without copying.
MLIR_RegionHandle MLIR_TakeOpRegion(MLIR_Context *ctx, MLIR_OpHandle op,
                                    size_t idx);

// Returns the block that contains `op`, or MLIR_INVALID_HANDLE if the
// op is not currently attached to a block.
MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_OpHandle op);

// Returns the 0-indexed position of `op` within `block`'s operations
// list, or SIZE_MAX if `op` is not in `block`.
size_t MLIR_GetBlockOpIndex(MLIR_BlockHandle block, MLIR_OpHandle op);

// Returns the region that contains `block`, or MLIR_INVALID_HANDLE if
// the block is not currently attached to a region.
MLIR_RegionHandle MLIR_GetBlockParentRegion(MLIR_BlockHandle block);

// Detaches `op` from its current parent block (if any) and appends it
// to the end of `dest`. Operand/result handles, attributes, and regions
// of `op` are unchanged.
void MLIR_MoveOpToBlockEnd(MLIR_Context *ctx, MLIR_OpHandle op,
                           MLIR_BlockHandle dest);

// Detaches `block` from its current parent region (if any) and appends
// it to the end of `dest`. The block's ops and arguments are preserved.
void MLIR_MoveBlockToRegionEnd(MLIR_Context *ctx, MLIR_BlockHandle block,
                               MLIR_RegionHandle dest);

// Replace operand `idx` of `op` with `value`. The previous operand
// reference is dropped (use lists are updated transparently by the
// upstream backend; the native backend simply overwrites the slot).
void MLIR_SetOpOperand(MLIR_Context *ctx, MLIR_OpHandle op,
                       size_t idx, MLIR_ValueHandle value);

// Replace successor `succ_idx` of `op` with `block`. `op` must be a
// terminator with at least `succ_idx + 1` successors. This does not
// touch the successor operands; use MLIR_SetOpSuccessorOperands to
// rewrite them.
void MLIR_SetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle op,
                         size_t succ_idx, MLIR_BlockHandle block);

// Replace the entire successor-operand list at `succ_idx` of `op` with
// the given `values[0..n)`. On the upstream backend, successor operands
// share storage with regular operands via BranchOpInterface; this
// function resizes that segment so the count after the call is exactly
// `n`. On the native backend, the per-successor operand vector is
// reallocated.
void MLIR_SetOpSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle op,
                                 size_t succ_idx,
                                 const MLIR_ValueHandle *values, size_t n);

// Detach `block` from its parent region (if any) and forget about it.
// Ops contained in `block` are NOT erased; callers should detach or
// erase them first to avoid dangling references.
void MLIR_EraseBlock(MLIR_Context *ctx, MLIR_BlockHandle block);

// Insert `block` into `region` immediately after `after`. If `after`
// is MLIR_INVALID_HANDLE the block is inserted at the start of the
// region. `block` must not already belong to a region; otherwise it is
// detached from its current parent first.
void MLIR_InsertRegionBlockAfter(MLIR_Context *ctx, MLIR_RegionHandle region,
                                 MLIR_BlockHandle block, MLIR_BlockHandle after);

// Insert `block` into `region` immediately before `before`. If
// `before` is MLIR_INVALID_HANDLE the block is appended to the region.
// `block` must not already belong to a region; otherwise it is
// detached from its current parent first.
void MLIR_InsertRegionBlockBefore(MLIR_Context *ctx, MLIR_RegionHandle region,
                                  MLIR_BlockHandle block, MLIR_BlockHandle before);

// Returns the last operation in `block` (its terminator), or
// MLIR_INVALID_HANDLE if the block is empty.
MLIR_OpHandle MLIR_GetBlockTerminator(MLIR_BlockHandle block);

// Returns the op that contains the parent region of `block`. If the
// region is a top-level (orphan) region, returns MLIR_INVALID_HANDLE.
MLIR_OpHandle MLIR_GetBlockParentOp(MLIR_BlockHandle block);

// Returns true if `block` is the first block of its parent region.
bool MLIR_BlockIsEntry(MLIR_BlockHandle block);

// Returns the number of incoming edges to `block` from terminators in
// other blocks of the same parent region. Each (predecessor block,
// successor slot) pair is counted once even if a predecessor branches
// to `block` from multiple slots.
size_t MLIR_GetBlockNumPredecessors(MLIR_BlockHandle block);

// Returns the predecessor at index `idx` of `block`. The 0-indexed
// successor slot in the predecessor's terminator that points at
// `block` is written to *out_succ_idx (if non-null). Returns
// MLIR_INVALID_HANDLE if `idx` is out of range.
MLIR_BlockHandle MLIR_GetBlockPredecessor(MLIR_BlockHandle block, size_t idx,
                                          size_t *out_succ_idx);

// Adds a new block argument of type `type` to the end of `block`'s
// argument list and returns its Value handle.
MLIR_ValueHandle MLIR_AddBlockArgument(MLIR_Context *ctx, MLIR_BlockHandle block,
                                       MLIR_TypeHandle type,
                                       MLIR_LocationHandle loc);

// Erases the `count` consecutive block arguments starting at `start`.
// Each argument must have no remaining uses; the caller is responsible
// for replacing them first via MLIR_ReplaceAllUsesOfValue.
void MLIR_EraseBlockArguments(MLIR_Context *ctx, MLIR_BlockHandle block,
                              size_t start, size_t count);

// Returns the block containing the definition of `value`. For block
// arguments this is the block carrying the argument; for op results it
// is the parent block of the defining op. Returns MLIR_INVALID_HANDLE
// if the value is detached.
MLIR_BlockHandle MLIR_GetValueParentBlock(MLIR_ValueHandle value);

// Returns the number of uses of `value`. A use is a single operand
// reference (operand, successor-operand of a terminator, etc.).
size_t MLIR_GetValueNumUses(MLIR_Context *ctx, MLIR_ValueHandle value);

// Returns one use of `value`. The owning operation is returned; the
// operand index within that op is written to *out_operand_idx (if
// non-null). Uses are returned in implementation-defined order. The
// order may change if any operand of any op is mutated; callers must
// capture uses into a vector before mutating.
MLIR_OpHandle MLIR_GetValueUseOwner(MLIR_Context *ctx, MLIR_ValueHandle value,
                                    size_t idx, size_t *out_operand_idx);

// Replaces all of `op`'s operands with the `n` values in `values`.
void MLIR_SetOpOperands(MLIR_Context *ctx, MLIR_OpHandle op,
                        const MLIR_ValueHandle *values, size_t n);

// Appends `value` to the successor-operand list at `succ_idx` of `op`.
// On the upstream backend this resizes the segment via
// `BranchOpInterface::SuccessorOperands::append`; on the native backend
// the per-successor operand vector is reallocated.
void MLIR_AppendOpSuccessorOperand(MLIR_Context *ctx, MLIR_OpHandle op,
                                   size_t succ_idx, MLIR_ValueHandle value);

// Moves all ops of `src` to the end of `dst`. `src` becomes empty but
// remains attached to its parent region.
void MLIR_SpliceBlockOps(MLIR_Context *ctx, MLIR_BlockHandle dst,
                         MLIR_BlockHandle src);

// Lift control-flow-graph operations (cf.br / cf.cond_br / cf.switch)
// in `module` into structured control flow (scf.if / scf.while /
// scf.index_switch). This is a prerequisite for the wasm pipeline,
// which rejects unstructured CFGs. Walks every `func.func` and
// `llvm.func` body in the module.
//
// Returns true on success, false on failure (e.g. a CFG that the lift
// could not transform into SCF). On the upstream backend this calls
// upstream MLIR's `transformCFGToSCF` (Bahmann/Reissmann 2015) via the
// in-tree `liftLLVMFuncCFGToSCF` shim so both `func.func` and
// `llvm.func` bodies are covered. On the native backend this calls
// the in-tree agnostic C port of the same algorithm in
// `mlir_lift_cf_to_scf.c` (see that file's header for the documented
// limitations).
//
// In the upstream-backed build, setting the environment variable
// `TINYC_LIFT_USE_NATIVE=1` opts into running the native port first
// (best effort; any leftover cf.* ops are then finished off by the
// upstream pass). This is a debugging / cross-validation knob; it has
// no effect in the native-only build (which always uses the native
// port).
bool MLIR_LiftCfToScf(MLIR_Context *ctx, MLIR_OpHandle module);

// Introspection
typedef enum {
    MLIR_ATTR_KIND_INTEGER,
    MLIR_ATTR_KIND_FLOAT,
    MLIR_ATTR_KIND_STRING,
    MLIR_ATTR_KIND_BOOL,
    MLIR_ATTR_KIND_ARRAY,
    MLIR_ATTR_KIND_DICT,
    MLIR_ATTR_KIND_TYPE,
    MLIR_ATTR_KIND_OTHER
} MLIR_AttrKind;

MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle attr);
string MLIR_GetAttributeName(MLIR_AttributeHandle attr);
// Print any attribute as a string (e.g. "0 : i64", "@callee", "[1, 2]").
// Used as a fallback when the kind is MLIR_ATTR_KIND_OTHER.
string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle attr);
int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle attr);
double MLIR_GetAttributeFloat(MLIR_AttributeHandle attr);
// For Integer and Float attributes, returns the numeric type. For other
// attribute kinds, returns MLIR_INVALID_HANDLE.
MLIR_TypeHandle MLIR_GetAttributeType(MLIR_AttributeHandle attr);
bool MLIR_GetAttributeBool(MLIR_AttributeHandle attr);
string MLIR_GetAttributeString(MLIR_AttributeHandle attr);
size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle attr);
MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle attr, size_t idx);
size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle attr);
MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle attr, size_t idx);
// For TypeAttr, returns the wrapped Type. For other kinds returns
// MLIR_INVALID_HANDLE.
MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_AttributeHandle attr);

// -----------------------------------------------------------------------------
// Location API
// -----------------------------------------------------------------------------

typedef enum {
    MLIR_LOC_UNKNOWN,
    MLIR_LOC_FILE,
    MLIR_LOC_NAME,
    MLIR_LOC_CALLSITE,
    MLIR_LOC_FUSED,
    MLIR_LOC_REF
} MLIR_LocationKind;

// Creation
MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *ctx, string original_text);
MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *ctx, string filename, int line, int column);
MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *ctx, string name);
MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *ctx, int ref_id);

// Accessors
MLIR_LocationKind MLIR_GetLocationKind(MLIR_LocationHandle loc);
string MLIR_GetLocationOriginalText(MLIR_LocationHandle loc);
string MLIR_GetLocationFileFilename(MLIR_LocationHandle loc);
int MLIR_GetLocationFileLine(MLIR_LocationHandle loc);
int MLIR_GetLocationFileColumn(MLIR_LocationHandle loc);
string MLIR_GetLocationName(MLIR_LocationHandle loc);
int MLIR_GetLocationRefId(MLIR_LocationHandle loc);

#ifdef __cplusplus
}
#endif
