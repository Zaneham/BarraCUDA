// Op-type → string mapping. Split out of mlir_parser.c so consumers like
// mlir_api_impl.c that need only the table can link against it without
// pulling in the parser/tokenizer/op_parsers tree.
//
// We split the table into two helper functions to keep each `switch`
// well under the wasm->macho backend's 256-deep block-stack limit
// (clang lowers a giant sparse switch into one nested block per case,
// and the combined enum now has 260+ entries). The two halves are
// chained: if the first returns a sentinel, the second runs.

#include <base/string.h>
#include "mlir_api.h"
#include "mlir_op_names.h"

// Sentinel used only between the two helpers below; never escapes.
#define OP_NAME_NOT_IN_THIS_HALF str_lit("")

static string op_type_to_string_half1(MLIR_OpType type) {
    switch (type) {
        case OP_TYPE_UNREGISTERED: return str_lit("unregistered");
        case OP_TYPE_MODULE: return str_lit("module");
        case OP_TYPE_ARITH_ADDI: return str_lit("arith.addi");
        case OP_TYPE_ARITH_SUBI: return str_lit("arith.subi");
        case OP_TYPE_ARITH_MULI: return str_lit("arith.muli");
        case OP_TYPE_ARITH_DIVI: return str_lit("arith.divi");
        case OP_TYPE_ARITH_ADDF: return str_lit("arith.addf");
        case OP_TYPE_ARITH_SUBF: return str_lit("arith.subf");
        case OP_TYPE_ARITH_MULF: return str_lit("arith.mulf");
        case OP_TYPE_ARITH_DIVF: return str_lit("arith.divf");
        case OP_TYPE_ARITH_CONSTANT: return str_lit("arith.constant");
        case OP_TYPE_ARITH_CMPI: return str_lit("arith.cmpi");
        case OP_TYPE_ARITH_CMPF: return str_lit("arith.cmpf");
        case OP_TYPE_MEMREF_LOAD: return str_lit("memref.load");
        case OP_TYPE_MEMREF_STORE: return str_lit("memref.store");
        case OP_TYPE_MEMREF_ALLOC: return str_lit("memref.alloc");
        case OP_TYPE_MEMREF_DEALLOC: return str_lit("memref.dealloc");
        case OP_TYPE_CF_BR: return str_lit("cf.br");
        case OP_TYPE_CF_COND_BR: return str_lit("cf.cond_br");
        case OP_TYPE_CF_SWITCH: return str_lit("cf.switch");
        case OP_TYPE_FUNC_FUNC: return str_lit("func.func");
        case OP_TYPE_FUNC_RETURN: return str_lit("func.return");
        case OP_TYPE_FUNC_CALL: return str_lit("func.call");
        case OP_TYPE_FUNC_CALL_INDIRECT: return str_lit("func.call_indirect");
        case OP_TYPE_FUNC_CONSTANT: return str_lit("func.constant");
        case OP_TYPE_UNREALIZED_CONVERSION_CAST: return str_lit("builtin.unrealized_conversion_cast");
        case OP_TYPE_SCF_FOR: return str_lit("scf.for");
        case OP_TYPE_SCF_WHILE: return str_lit("scf.while");
        case OP_TYPE_SCF_IF: return str_lit("scf.if");
        case OP_TYPE_SCF_YIELD: return str_lit("scf.yield");
        case OP_TYPE_SCF_CONDITION: return str_lit("scf.condition");
        case OP_TYPE_SCF_INDEX_SWITCH: return str_lit("scf.index_switch");
        case OP_TYPE_TT_GET_PROGRAM_ID: return str_lit("tt.get_program_id");
        case OP_TYPE_TT_LOAD: return str_lit("tt.load");
        case OP_TYPE_TT_STORE: return str_lit("tt.store");
        case OP_TYPE_TT_MAKE_RANGE: return str_lit("tt.make_range");
        case OP_TYPE_TT_SPLAT: return str_lit("tt.splat");
        case OP_TYPE_TT_ADDPTR: return str_lit("tt.addptr");
        case OP_TYPE_TT_RETURN: return str_lit("tt.return");
        case OP_TYPE_ARITH_SELECT: return str_lit("arith.select");
        case OP_TYPE_ARITH_BITCAST: return str_lit("arith.bitcast");
        case OP_TYPE_ARITH_SITOFP: return str_lit("arith.sitofp");
        case OP_TYPE_ARITH_FPTOSI: return str_lit("arith.fptosi");
        case OP_TYPE_ARITH_INDEX_CAST: return str_lit("arith.index_cast");
        case OP_TYPE_ARITH_EXTSI: return str_lit("arith.extsi");
        case OP_TYPE_ARITH_TRUNCI: return str_lit("arith.trunci");
        case OP_TYPE_ARITH_EXTF: return str_lit("arith.extf");
        case OP_TYPE_ARITH_TRUNCF: return str_lit("arith.truncf");
        case OP_TYPE_ARITH_EXTUI: return str_lit("arith.extui");
        case OP_TYPE_ARITH_MAXF: return str_lit("arith.maxf");
        case OP_TYPE_ARITH_DIVSI: return str_lit("arith.divsi");
        case OP_TYPE_ARITH_REMSI: return str_lit("arith.remsi");
        case OP_TYPE_ARITH_DIVUI: return str_lit("arith.divui");
        case OP_TYPE_ARITH_REMUI: return str_lit("arith.remui");
        case OP_TYPE_ARITH_SHRUI: return str_lit("arith.shrui");
        case OP_TYPE_ARITH_ORI: return str_lit("arith.ori");
        case OP_TYPE_ARITH_MINSI: return str_lit("arith.minsi");
        case OP_TYPE_ARITH_ANDI: return str_lit("arith.andi");
        case OP_TYPE_MATH_EXP: return str_lit("math.exp");
        case OP_TYPE_MATH_LOG: return str_lit("math.log");
        case OP_TYPE_TT_FUNC: return str_lit("tt.func");
        case OP_TYPE_TT_CALL: return str_lit("tt.call");
        case OP_TYPE_TT_REDUCE: return str_lit("tt.reduce");
        case OP_TYPE_TT_BROADCAST: return str_lit("tt.broadcast");
        case OP_TYPE_TT_EXPAND_DIMS: return str_lit("tt.expand_dims");
        case OP_TYPE_TT_DOT: return str_lit("tt.dot");
        case OP_TYPE_TT_PURE_EXTERN_ELEMENTWISE: return str_lit("tt.pure_extern_elementwise");
        case OP_TYPE_GPU_LAUNCH: return str_lit("gpu.launch");
        case OP_TYPE_AFFINE_FOR: return str_lit("affine.for");
        case OP_TYPE_AFFINE_LOAD: return str_lit("affine.load");
        case OP_TYPE_VECTOR_PRINT: return str_lit("vector.print");
        case OP_TYPE_STD_CONSTANT: return str_lit("std.constant");
        case OP_TYPE_STD_RETURN: return str_lit("std.return");
        case OP_TYPE_TENSOR_EXTRACT: return str_lit("tensor.extract");
        case OP_TYPE_TENSOR_SPLAT: return str_lit("tensor.splat");
        case OP_TYPE_TENSOR_COLLAPSE_SHAPE: return str_lit("tensor.collapse_shape");
        case OP_TYPE_LINALG_FILL: return str_lit("linalg.fill");
        case OP_TYPE_LINALG_COPY: return str_lit("linalg.copy");
        case OP_TYPE_INDEX_CONSTANT: return str_lit("index.constant");
        case OP_TYPE_LLVM_MLIR_UNDEF: return str_lit("llvm.mlir.undef");
        case OP_TYPE_LLVM_ALLOCA: return str_lit("llvm.alloca");
        case OP_TYPE_LLVM_LOAD: return str_lit("llvm.load");
        case OP_TYPE_LLVM_STORE: return str_lit("llvm.store");
        case OP_TYPE_LLVM_GEP: return str_lit("llvm.getelementptr");
        case OP_TYPE_LLVM_MLIR_ZERO: return str_lit("llvm.mlir.zero");
        case OP_TYPE_LLVM_MLIR_CONSTANT: return str_lit("llvm.mlir.constant");
        case OP_TYPE_LLVM_ICMP: return str_lit("llvm.icmp");
        case OP_TYPE_LLVM_MLIR_ADDRESSOF: return str_lit("llvm.mlir.addressof");
        case OP_TYPE_LLVM_MLIR_GLOBAL: return str_lit("llvm.mlir.global");
        case OP_TYPE_LLVM_RETURN: return str_lit("llvm.return");
        case OP_TYPE_LLVM_PTRTOINT: return str_lit("llvm.ptrtoint");
        case OP_TYPE_LLVM_FUNC: return str_lit("llvm.func");
        case OP_TYPE_LLVM_CALL: return str_lit("llvm.call");
        case OP_TYPE_LLVM_SEXT: return str_lit("llvm.sext");
        case OP_TYPE_LLVM_ADD: return str_lit("llvm.add");
        case OP_TYPE_LLVM_SUB: return str_lit("llvm.sub");
        case OP_TYPE_LLVM_MUL: return str_lit("llvm.mul");
        case OP_TYPE_LLVM_SDIV: return str_lit("llvm.sdiv");
        case OP_TYPE_LLVM_UDIV: return str_lit("llvm.udiv");
        case OP_TYPE_LLVM_SREM: return str_lit("llvm.srem");
        case OP_TYPE_LLVM_UREM: return str_lit("llvm.urem");
        case OP_TYPE_LLVM_AND: return str_lit("llvm.and");
        case OP_TYPE_LLVM_OR: return str_lit("llvm.or");
        case OP_TYPE_LLVM_XOR: return str_lit("llvm.xor");
        case OP_TYPE_LLVM_SHL: return str_lit("llvm.shl");
        case OP_TYPE_LLVM_LSHR: return str_lit("llvm.lshr");
        case OP_TYPE_LLVM_ASHR: return str_lit("llvm.ashr");
        case OP_TYPE_LLVM_TRUNC: return str_lit("llvm.trunc");
        case OP_TYPE_LLVM_ZEXT: return str_lit("llvm.zext");
        case OP_TYPE_ARITH_XORI: return str_lit("arith.xori");
        case OP_TYPE_ARITH_SHLI: return str_lit("arith.shli");
        case OP_TYPE_ARITH_SHRSI: return str_lit("arith.shrsi");
        case OP_TYPE_RETURN: return str_lit("return");
        case OP_TYPE_TT_REDUCE_RETURN: return str_lit("tt.reduce.return");
        // wasmssa dialect
        case OP_TYPE_WASMSSA_FUNC: return str_lit("wasmssa.func");
        case OP_TYPE_WASMSSA_IMPORT_FUNC: return str_lit("wasmssa.import_func");
        case OP_TYPE_WASMSSA_IMPORT_GLOBAL: return str_lit("wasmssa.import_global");
        case OP_TYPE_WASMSSA_CONST: return str_lit("wasmssa.const");
        case OP_TYPE_WASMSSA_ADD: return str_lit("wasmssa.add");
        case OP_TYPE_WASMSSA_SUB: return str_lit("wasmssa.sub");
        case OP_TYPE_WASMSSA_BINOP: return str_lit("wasmssa.binop");
        case OP_TYPE_WASMSSA_UNOP: return str_lit("wasmssa.unop");
        case OP_TYPE_WASMSSA_LOAD: return str_lit("wasmssa.load");
        case OP_TYPE_WASMSSA_STORE: return str_lit("wasmssa.store");
        case OP_TYPE_WASMSSA_GLOBAL_GET: return str_lit("wasmssa.global_get");
        case OP_TYPE_WASMSSA_GLOBAL_SET: return str_lit("wasmssa.global_set");
        case OP_TYPE_WASMSSA_EXTEND_I32_S: return str_lit("wasmssa.extend_i32_s");
        case OP_TYPE_WASMSSA_RETURN: return str_lit("wasmssa.return");
        case OP_TYPE_WASMSSA_CALL: return str_lit("wasmssa.call");
        default: return OP_NAME_NOT_IN_THIS_HALF;
    }
}

static string op_type_to_string_half2(MLIR_OpType type) {
    switch (type) {
        case OP_TYPE_WASMSSA_BLOCK:        return str_lit("wasmssa.block");
        case OP_TYPE_WASMSSA_LOOP:         return str_lit("wasmssa.loop");
        case OP_TYPE_WASMSSA_IF:           return str_lit("wasmssa.if");
        case OP_TYPE_WASMSSA_BLOCK_RETURN: return str_lit("wasmssa.block_return");
        case OP_TYPE_WASMSSA_UNREACHABLE:  return str_lit("wasmssa.unreachable");
        case OP_TYPE_WASMSSA_BR:          return str_lit("wasmssa.br");
        case OP_TYPE_WASMSSA_BR_IF:       return str_lit("wasmssa.br_if");
        case OP_TYPE_WASMSSA_SELECT:      return str_lit("wasmssa.select");
        case OP_TYPE_WASMSSA_EQZ:         return str_lit("wasmssa.eqz");
        case OP_TYPE_WASMSSA_ADDRESSOF:   return str_lit("wasmssa.addressof");
        case OP_TYPE_WASMSSA_FUNC_ADDR:   return str_lit("wasmssa.func_addr");
        case OP_TYPE_WASMSSA_CALL_INDIRECT: return str_lit("wasmssa.call_indirect");
        case OP_TYPE_WASMSSA_MEMORY_SIZE: return str_lit("wasmssa.memory_size");
        case OP_TYPE_WASMSSA_MEMORY_GROW: return str_lit("wasmssa.memory_grow");
        case OP_TYPE_WASMSSA_LOCAL_GET:   return str_lit("wasmssa.local_get");
        case OP_TYPE_WASMSSA_LOCAL_SET:   return str_lit("wasmssa.local_set");
        // wasmstack dialect
        case OP_TYPE_WASMSTACK_FUNC: return str_lit("wasmstack.func");
        case OP_TYPE_WASMSTACK_IMPORT_FUNC: return str_lit("wasmstack.import_func");
        case OP_TYPE_WASMSTACK_IMPORT_GLOBAL: return str_lit("wasmstack.import_global");
        case OP_TYPE_WASMSTACK_LOCAL_GET: return str_lit("wasmstack.local.get");
        case OP_TYPE_WASMSTACK_LOCAL_SET: return str_lit("wasmstack.local.set");
        case OP_TYPE_WASMSTACK_LOCAL_TEE: return str_lit("wasmstack.local.tee");
        case OP_TYPE_WASMSTACK_CONST: return str_lit("wasmstack.const");
        case OP_TYPE_WASMSTACK_ADD: return str_lit("wasmstack.add");
        case OP_TYPE_WASMSTACK_SUB: return str_lit("wasmstack.sub");
        case OP_TYPE_WASMSTACK_BINOP: return str_lit("wasmstack.binop");
        case OP_TYPE_WASMSTACK_UNOP: return str_lit("wasmstack.unop");
        case OP_TYPE_WASMSTACK_LOAD: return str_lit("wasmstack.load");
        case OP_TYPE_WASMSTACK_STORE: return str_lit("wasmstack.store");
        case OP_TYPE_WASMSTACK_GLOBAL_GET: return str_lit("wasmstack.global.get");
        case OP_TYPE_WASMSTACK_GLOBAL_SET: return str_lit("wasmstack.global.set");
        case OP_TYPE_WASMSTACK_EXTEND_I32_S: return str_lit("wasmstack.extend_i32_s");
        case OP_TYPE_WASMSTACK_RETURN: return str_lit("wasmstack.return");
        case OP_TYPE_WASMSTACK_CALL: return str_lit("wasmstack.call");
        case OP_TYPE_WASMSTACK_BLOCK:  return str_lit("wasmstack.block");
        case OP_TYPE_WASMSTACK_LOOP:   return str_lit("wasmstack.loop");
        case OP_TYPE_WASMSTACK_IF:     return str_lit("wasmstack.if");
        case OP_TYPE_WASMSTACK_ELSE:   return str_lit("wasmstack.else");
        case OP_TYPE_WASMSTACK_END:    return str_lit("wasmstack.end");
        case OP_TYPE_WASMSTACK_BR:     return str_lit("wasmstack.br");
        case OP_TYPE_WASMSTACK_BR_IF:  return str_lit("wasmstack.br_if");
        case OP_TYPE_WASMSTACK_UNREACHABLE: return str_lit("wasmstack.unreachable");
        case OP_TYPE_WASMSTACK_SELECT: return str_lit("wasmstack.select");
        case OP_TYPE_WASMSTACK_EQZ:    return str_lit("wasmstack.eqz");
        case OP_TYPE_WASMSTACK_ADDRESSOF: return str_lit("wasmstack.addressof");
        case OP_TYPE_WASMSTACK_FUNC_ADDR: return str_lit("wasmstack.func_addr");
        case OP_TYPE_WASMSTACK_CALL_INDIRECT: return str_lit("wasmstack.call_indirect");
        case OP_TYPE_WASMSTACK_MEMORY_SIZE: return str_lit("wasmstack.memory_size");
        case OP_TYPE_WASMSTACK_MEMORY_GROW: return str_lit("wasmstack.memory_grow");
        case OP_TYPE_WASMSTACK_DROP:        return str_lit("wasmstack.drop");
        case OP_TYPE_WASMSTACK_BR_TABLE:    return str_lit("wasmstack.br_table");
        case OP_TYPE_WASMSTACK_DATA_SEGMENT: return str_lit("wasmstack.data_segment");
        case OP_TYPE_WASMSTACK_GLOBAL_DECL: return str_lit("wasmstack.global_decl");
        case OP_TYPE_WASMSTACK_FUNC_ADDR_DECL: return str_lit("wasmstack.func_addr_decl");
        // aarch64 dialect
        case OP_TYPE_AARCH64_FUNC:         return str_lit("aarch64.func");
        case OP_TYPE_AARCH64_MOVZ:         return str_lit("aarch64.movz");
        case OP_TYPE_AARCH64_MOVK:         return str_lit("aarch64.movk");
        case OP_TYPE_AARCH64_MOV_X:        return str_lit("aarch64.mov_x");
        case OP_TYPE_AARCH64_BL:           return str_lit("aarch64.bl");
        case OP_TYPE_AARCH64_BLR:          return str_lit("aarch64.blr");
        case OP_TYPE_AARCH64_SVC:          return str_lit("aarch64.svc");
        case OP_TYPE_AARCH64_RET:          return str_lit("aarch64.ret");
        case OP_TYPE_AARCH64_ADD_IMM:      return str_lit("aarch64.add_imm");
        case OP_TYPE_AARCH64_SUB_IMM:      return str_lit("aarch64.sub_imm");
        case OP_TYPE_AARCH64_ADD_REG:      return str_lit("aarch64.add_reg");
        case OP_TYPE_AARCH64_SUB_REG:      return str_lit("aarch64.sub_reg");
        case OP_TYPE_AARCH64_MUL:          return str_lit("aarch64.mul");
        case OP_TYPE_AARCH64_SDIV:         return str_lit("aarch64.sdiv");
        case OP_TYPE_AARCH64_UDIV:         return str_lit("aarch64.udiv");
        case OP_TYPE_AARCH64_MSUB:         return str_lit("aarch64.msub");
        case OP_TYPE_AARCH64_AND_REG:      return str_lit("aarch64.and_reg");
        case OP_TYPE_AARCH64_AND_IMM:      return str_lit("aarch64.and_imm");
        case OP_TYPE_AARCH64_ORR_REG:      return str_lit("aarch64.orr_reg");
        case OP_TYPE_AARCH64_EOR_REG:      return str_lit("aarch64.eor_reg");
        case OP_TYPE_AARCH64_LSL_REG:      return str_lit("aarch64.lsl_reg");
        case OP_TYPE_AARCH64_LSR_REG:      return str_lit("aarch64.lsr_reg");
        case OP_TYPE_AARCH64_ASR_REG:      return str_lit("aarch64.asr_reg");
        case OP_TYPE_AARCH64_LSL_IMM:      return str_lit("aarch64.lsl_imm");
        case OP_TYPE_AARCH64_LSR_IMM:      return str_lit("aarch64.lsr_imm");
        case OP_TYPE_AARCH64_ASR_IMM:      return str_lit("aarch64.asr_imm");
        case OP_TYPE_AARCH64_SXTW:         return str_lit("aarch64.sxtw");
        case OP_TYPE_AARCH64_SXTB:         return str_lit("aarch64.sxtb");
        case OP_TYPE_AARCH64_SXTH:         return str_lit("aarch64.sxth");
        case OP_TYPE_AARCH64_UXTW:         return str_lit("aarch64.uxtw");
        case OP_TYPE_AARCH64_LDR_W:        return str_lit("aarch64.ldr_w");
        case OP_TYPE_AARCH64_STR_W:        return str_lit("aarch64.str_w");
        case OP_TYPE_AARCH64_LDR_X:        return str_lit("aarch64.ldr_x");
        case OP_TYPE_AARCH64_STR_X:        return str_lit("aarch64.str_x");
        case OP_TYPE_AARCH64_STRB_IMM:     return str_lit("aarch64.strb_imm");
        case OP_TYPE_AARCH64_LDRB_IMM:     return str_lit("aarch64.ldrb_imm");
        case OP_TYPE_AARCH64_LDR_W_REG:    return str_lit("aarch64.ldr_w_reg");
        case OP_TYPE_AARCH64_STR_W_REG:    return str_lit("aarch64.str_w_reg");
        case OP_TYPE_AARCH64_LDR_X_REG:    return str_lit("aarch64.ldr_x_reg");
        case OP_TYPE_AARCH64_STR_X_REG:    return str_lit("aarch64.str_x_reg");
        case OP_TYPE_AARCH64_LDRB_REG:     return str_lit("aarch64.ldrb_reg");
        case OP_TYPE_AARCH64_STRB_REG:     return str_lit("aarch64.strb_reg");
        case OP_TYPE_AARCH64_ADRP_DATA:    return str_lit("aarch64.adrp_data");
        case OP_TYPE_AARCH64_ADD_DATA_LO:  return str_lit("aarch64.add_data_lo");
        case OP_TYPE_AARCH64_PROLOGUE:     return str_lit("aarch64.prologue");
        case OP_TYPE_AARCH64_EPILOGUE:     return str_lit("aarch64.epilogue");
        case OP_TYPE_AARCH64_CMP_REG:      return str_lit("aarch64.cmp_reg");
        case OP_TYPE_AARCH64_CMP_IMM:      return str_lit("aarch64.cmp_imm");
        case OP_TYPE_AARCH64_CSET:         return str_lit("aarch64.cset");
        case OP_TYPE_AARCH64_CSEL:         return str_lit("aarch64.csel");
        case OP_TYPE_AARCH64_B:            return str_lit("aarch64.b");
        case OP_TYPE_AARCH64_B_COND:       return str_lit("aarch64.b_cond");
        case OP_TYPE_AARCH64_CBZ:          return str_lit("aarch64.cbz");
        case OP_TYPE_AARCH64_CBNZ:         return str_lit("aarch64.cbnz");
        case OP_TYPE_AARCH64_LABEL:        return str_lit("aarch64.label");
        case OP_TYPE_AARCH64_BRK:          return str_lit("aarch64.brk");
        case OP_TYPE_AARCH64_FMOV_GP_V:    return str_lit("aarch64.fmov_gp_v");
        case OP_TYPE_AARCH64_FP_BINOP:     return str_lit("aarch64.fp_binop");
        case OP_TYPE_AARCH64_FP_UNOP:      return str_lit("aarch64.fp_unop");
        case OP_TYPE_AARCH64_FCMP:         return str_lit("aarch64.fcmp");
        case OP_TYPE_AARCH64_FP_CVT:       return str_lit("aarch64.fp_cvt");
        case OP_TYPE_AARCH64_DATA_INIT:    return str_lit("aarch64.data_init");
        default: return OP_NAME_NOT_IN_THIS_HALF;
    }
}

string op_type_to_string(MLIR_OpType type) {
    string s = op_type_to_string_half1(type);
    if (s.size != 0) return s;
    s = op_type_to_string_half2(type);
    if (s.size != 0) return s;
    return str_lit("unknown");
}
