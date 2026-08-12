/*
This file contains specific operation parsers.

Each parser is using Parser to parse a specific operation and return it in `op`.
*/
#pragma once

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"
#include "mlir_parser.h"

#ifdef __cplusplus
extern "C" {
#endif


// Specific operation parsers

OperationParserResult parse_affine_load_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_arith_binary_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_arith_cmpi_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_arith_constant_op(Parser *parser, OperationParserParams *params);
OperationParserResult parse_arith_select_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_cf_br_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_cf_cond_br_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_func_call_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_gpu_launch_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_index_constant_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_linalg_fill_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_memref_load_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_memref_store_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_scf_if_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_scf_for_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_scf_while_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_scf_yield_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_addptr_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_load_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_store_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_func_func_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_affine_for_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_return_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_std_constant_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_tensor_collapse_shape_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tensor_extract_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tensor_splat_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_tt_call_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_func_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_get_program_id_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_make_range_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_reduce_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_splat_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_vector_print_op(Parser *parser, const OperationParserParams *params);


OperationParserResult parse_generic_op(Parser *parser, const OperationParserParams *params);

#ifdef __cplusplus
}
#endif
