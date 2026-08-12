#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/strbuf.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"
#include "tokenizer.h"
#include "op_parsers.h"


OperationParserResult parse_arith_constant_op(Parser *parser, OperationParserParams *params) {
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    MLIR_ValueHandle *results = params->lhs_results;
    size_t n_results = params->n_lhs_results;
    MLIR_ValueHandle result_value = (n_results > 0) ? results[0] : MLIR_INVALID_HANDLE;

    MLIR_LocationHandle op_location = MLIR_INVALID_HANDLE;
    bool has_explicit_type = false;

    // Defer building the typed integer/float attribute until the result type
    // has been parsed: arith.constant requires `value`'s attribute type to
    // equal the result type (so `0 : index` becomes IntegerAttr typed index,
    // not the i64 default).
    bool pending_int = false;
    int64_t pending_int_val = 0;
    bool pending_float = false;
    double pending_float_val = 0.0;

    if (parser_peek(parser, TK_INTEGER)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_INTEGER);

        int64_t parsed_value = 0;
        for (size_t i = 0; i < value_str.size; i++) {
            if (value_str.str[i] >= '0' && value_str.str[i] <= '9') {
                parsed_value = parsed_value * 10 + (value_str.str[i] - '0');
            }
        }
        pending_int = true;
        pending_int_val = parsed_value;
    } else if (parser_peek(parser, TK_REAL)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_REAL);

        char *buffer = arena_new_array(parser->arena, char, value_str.size + 1);
        memcpy(buffer, value_str.str, value_str.size);
        buffer[value_str.size] = '\0';
        double parsed_value = atof(buffer);
        pending_float = true;
        pending_float_val = parsed_value;
    } else if (parser_peek(parser, TK_NAME)) {
        string name_str = parser_token_str(parser);
        if (str_eq(name_str, str_lit("true")) || str_eq(name_str, str_lit("false"))) {
            parser_expect(parser, TK_NAME);
            MLIR_TypeHandle bool_type = mlir_type_create_from_string(parser->ctx, str_lit("i1"));
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        MLIR_CreateAttributeBool(parser->ctx, str_lit("value"),
                                                 str_eq(name_str, str_lit("true"))));

            if (!result_value) {
                result_value = MLIR_CreateValueOpResult(parser->ctx, MLIR_INVALID_HANDLE, 0, bool_type, (string){MLIR_INVALID_HANDLE, 0}, MLIR_INVALID_HANDLE);
            }
            result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
            result_types[0] = bool_type;
            n_result_types = 1;
        } else {
            string payload = str_lit("");
            int angle = 0;
            do {
                string tok = parser_token_str(parser);
                payload = payload.size ? str_concat(parser->arena, payload, tok) : tok;
                parser_next_token(parser);
                if (parser_peek(parser, TK_LANGLE)) angle++;
                else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                if (angle == 0 && parser_peek(parser, TK_COLON)) break;
            } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_string_attr(parser, str_lit("value_text"), payload));
        }
    } else {
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_COLON)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            parser_next_token(parser);
        }
    }

    if (!has_explicit_type && parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        string type_str = str_lit("");
        if (parse_type_string(parser, &type_str)) {
            MLIR_TypeHandle type = mlir_type_create_from_string(parser->ctx, type_str);
            if (!result_types) {
                result_types = arena_new_array(parser->arena, MLIR_TypeHandle, 1);
            }
            result_types[0] = type;
            n_result_types = 1;
            if (!result_value) {
                result_value = MLIR_CreateValueOpResult(parser->ctx, MLIR_INVALID_HANDLE, 0, type, (string){MLIR_INVALID_HANDLE, 0}, MLIR_INVALID_HANDLE);
            }
            has_explicit_type = true;
        }
    }

    // Now that the result type is known (or defaulted), build the deferred
    // typed integer/float `value` attribute so it carries the correct type.
    if (pending_int || pending_float) {
        MLIR_TypeHandle attr_type =
            (n_result_types > 0) ? result_types[0]
                                 : mlir_type_create_from_string(parser->ctx,
                                       pending_int ? str_lit("i64") : str_lit("f64"));
        if (pending_int) {
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_integer_attr(parser, str_lit("value"), pending_int_val, attr_type));
        } else {
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_float_attr(parser, str_lit("value"), pending_float_val, attr_type));
        }
    }

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op_location = parse_loc(parser);
    }

    if (!result_value) {
        result_value = MLIR_CreateValueOpResult(parser->ctx, MLIR_INVALID_HANDLE, 0, MLIR_INVALID_HANDLE, (string){MLIR_INVALID_HANDLE, 0}, MLIR_INVALID_HANDLE);
    }

    if (!results) {
        results = arena_new_array(parser->arena, MLIR_ValueHandle, 1);
        results[0] = result_value;
        n_results = 1;
    }

    // Parse additional attributes and result types
    parse_generic_attrs_and_result_type(parser, &attributes, &n_attributes, &attributes_capacity,
                                         &result_types, &n_result_types, params->op_type);

    // Fallback location if none was parsed
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // NOW create operation with all collected data
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        params->op_type,
        params->opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        MLIR_INVALID_HANDLE, 0,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_arith_binary_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    if (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, false)) {
            OperationParserResult empty = {0};
            return empty;
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str2 = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                if (!operand2) {
                    parser_warning(parser, format(parser->arena, str_lit("Undefined value: {}"), reg_str2), parser->first, parser->last);
                    MLIR_TypeHandle unknown_type = mlir_type_create_from_string(parser->ctx, str_lit("unknown"));
                    operand2 = MLIR_CreateValueBlockArg(parser->ctx, reg_str2, 0, unknown_type, MLIR_INVALID_HANDLE);
                }
                VecValue_push_back(parser->arena, &operands, operand2);
            }
        }
    }

    // Parse any attributes and result types using helper functions
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0; // tracks attribute list growth for arith binary ops

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    // Optional default result type for specific ops when not provided
    if (n_result_types == 0 && params->op_type == OP_TYPE_ARITH_ADDF) {
        result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
        result_types[0] = mlir_type_create_from_string(params->ctx, str_lit("tensor<16xf32>"));
        n_result_types = 1;
    }

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create results from the operation's result types
    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, MLIR_INVALID_HANDLE, result_types, n_result_types, &n_results);

    // Create the operation at the end
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      results, n_results,
                                      operands.data, operands.size,
                                      MLIR_INVALID_HANDLE, 0,
                                      op_location, params->unnumbered_loc_def,
                                      params->trailing_comment, params->source_line_start);

    // Set result definitions
    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_func_call_op(Parser *parser, const OperationParserParams *params) {
    // Parse call @function(%args) : (arg_types) -> result_type
    // Function name (@name is tokenized as TK_FUNCTION_NAME)
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture callee
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        // Store as attribute 'callee'
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity, create_string_attr(parser, str_lit("callee"), fname));
    }

    // Parse operands in parentheses
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, operand);
            } else {
                parser_next_token(parser); // skip unknown tokens
            }

            if (parser_peek(parser, TK_COMMA)) {
                parser_expect(parser, TK_COMMA);
            } else if (!parser_peek(parser, TK_RPAREN)) {
                break;
            }
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Parse any additional attributes and result types
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    // Parse function signature: : (arg_types) -> result_type
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip arg types in parentheses
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int paren_depth = 1;
            while (paren_depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) paren_depth++;
                else if (parser_peek(parser, TK_RPAREN)) paren_depth--;
                parser_next_token(parser);
            }
        }

        // Parse arrow and result type
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, type_str);
                n_result_types = 1;
            }
        }
    }

    // Parse any remaining result types if not already parsed
    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);
    }

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      MLIR_INVALID_HANDLE, 0,
                                      op_location, params->unnumbered_loc_def,
                                      params->trailing_comment, params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tt_get_program_id_op(Parser *parser, const OperationParserParams *params) {
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_NAME)) {
        string axis = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_string_attr(parser, str_lit("axis"), axis));
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                      params->op_type, MLIR_INVALID_HANDLE);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        MLIR_INVALID_HANDLE,
        0,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tt_splat_op(Parser *parser, const OperationParserParams *params) {
    MLIR_ValueHandle *operands = MLIR_INVALID_HANDLE;
    size_t n_operands = 0;
    MLIR_ValueHandle operand = MLIR_INVALID_HANDLE;

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        operands = arena_new_array(params->arena, MLIR_ValueHandle, 1);
        operands[0] = operand;
        n_operands = 1;
    }

    // Parse any attributes and result types using helper functions
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    // Infer result type if not already parsed from trailing type
    if (n_result_types == 0) {
        MLIR_TypeHandle operand_type = operand ? MLIR_GetValueType(operand) : MLIR_INVALID_HANDLE;
        MLIR_TypeHandle result_type = MLIR_INVALID_HANDLE;
        if (operand_type) {
            string operand_type_str = MLIR_GetTypeString(parser->ctx, operand_type);
            if (str_eq(operand_type_str, str_lit("!tt.ptr<f32>"))) {
                result_type = mlir_type_create_from_string(params->ctx, str_lit("tensor<16x!tt.ptr<f32>>"));
            } else if (str_eq(operand_type_str, str_lit("i32"))) {
                result_type = mlir_type_create_from_string(params->ctx, str_lit("tensor<16xi32>"));
            } else {
                result_type = mlir_type_create_from_string(params->ctx, str_lit("tensor<16xi32>"));
            }
        } else {
            result_type = mlir_type_create_from_string(params->ctx, str_lit("tensor<16xi32>"));
        }
        result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
        result_types[0] = result_type;
        n_result_types = 1;
    }

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands, n_operands,
                                      MLIR_INVALID_HANDLE, 0,
                                      op_location, params->unnumbered_loc_def,
                                      params->trailing_comment, params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tt_make_range_op(Parser *parser, const OperationParserParams *params) {
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    bool have_end = false;
    int64_t end_val = 0;
    bool have_start = false;
    int64_t start_val = 0;

    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);

        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // "end"
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                char *buffer = arena_new_array(parser->arena, char, ival.size + 1);
                memcpy(buffer, ival.str, ival.size);
                buffer[ival.size] = '\0';
                end_val = atoll(buffer);
                parser_expect(parser, TK_INTEGER);
            }
            if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
            }
            have_end = true;
        }

        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);

        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // "start"
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                char *buffer = arena_new_array(parser->arena, char, ival.size + 1);
                memcpy(buffer, ival.str, ival.size);
                buffer[ival.size] = '\0';
                start_val = atoll(buffer);
                parser_expect(parser, TK_INTEGER);
            }
            if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
            }
            have_start = true;
        }

        parser_expect(parser, TK_RBRACE);
    }

    MLIR_TypeHandle i32_ty = mlir_type_create_from_string(parser->ctx, str_lit("i32"));
    if (have_end) {
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_integer_attr(parser, str_lit("end"), end_val, i32_ty));
    }
    if (have_start) {
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_integer_attr(parser, str_lit("start"), start_val, i32_ty));
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        MLIR_INVALID_HANDLE,
        0,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tt_addptr_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    while (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, true)) {
            OperationParserResult empty = {0};
            return empty;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

        string pointer_sig = str_lit("");
        if (parse_type_string(parser, &pointer_sig)) {
            result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
            result_types[0] = mlir_type_create_from_string(params->ctx, pointer_sig);
            n_result_types = 1;
        }

        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
        if (parser_peek(parser, TK_COMMA)) {
            do {
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE));
        }
    } else if (operands.size > 0) {
        MLIR_ValueHandle first = operands.data[0];
        MLIR_TypeHandle first_type = first ? MLIR_GetValueType(first) : MLIR_INVALID_HANDLE;
        if (first_type) {
            result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
            result_types[0] = first_type;
            n_result_types = 1;
        }
    }

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    MLIR_ValueHandle *results = MLIR_INVALID_HANDLE;
    size_t n_results = 0;
    results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tensor_extract_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    // Parse the tensor operand to extract from
    if (parser_peek(parser, TK_REGISTER)) {
        string tensor_reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle tensor_val = symbol_table_lookup(&parser->symbol_table, tensor_reg);
        if (!tensor_val) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, tensor_val);
    } else {
        parser_error(parser, str_lit("Expected tensor operand"), parser->first, parser->last);
        OperationParserResult empty = {0};
        return empty;
    }

    // Parse indices in brackets if present
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string idx_reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                while (parser_peek(parser, TK_HASH_NAME)) {
                    consume_optional_hash_selector(parser);
                }
                MLIR_ValueHandle idx_val = symbol_table_lookup(&parser->symbol_table, idx_reg);
                if (!idx_val) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, idx_val);
                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    // The classic syntax `tensor.extract %t[%i...] : tensor<...xT>` types the
    // SOURCE tensor after `:`; the result is the element type. Peel it out.
    if (n_result_types == 1 && result_types) {
        MLIR_TypeHandle elem = MLIR_GetTypeShapedElement(result_types[0]);
        if (elem != MLIR_INVALID_HANDLE) {
            result_types[0] = elem;
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_memref_load_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle val = lookup_or_create_value(parser, reg, str_lit("unknown"));
        VecValue_push_back(parser->arena, &operands, val);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        bool first = true;
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string idx = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle val = lookup_or_create_value(parser, idx, str_lit("index"));
                VecValue_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    MLIR_ValueHandle *results = params->lhs_results;
    size_t n_results = params->n_lhs_results;

    // Parse attributes from both <{...}> and {...} blocks
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    // Parse result types (memref.load operations can have result types)
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    // The trailing `: memref<NxMxELEM[, layout][, memspace]>` describes the
    // source memref, not the result type. memref.load returns the element
    // type. Stash the original memref type as a hidden `_source_type` string
    // attribute so the classic printer can roundtrip it (the first operand's
    // type is not reliable: SSA shadowing in nested regions can cause it to
    // resolve to a different `%name` of type `index`).
    //
    // Parse the memref type structurally to find ELEM:
    //   - the dim/element chunk runs from after `memref<` to the first `,` at
    //     bracket depth 0 (or to the closing `>`);
    //   - within that chunk, ELEM follows the last `x` at depth 0 (so nested
    //     types like `vector<4xf32>` are not split).
    if (n_result_types == 1) {
        string ts = MLIR_GetTypeString(parser->ctx, result_types[0]);
        if (ts.size > 7 && strncmp(ts.str, "memref<", 7) == 0 && ts.str[ts.size - 1] == '>') {
            size_t chunk_end = ts.size - 1;  // exclusive; defaults to closing '>'
            int depth = 0;
            for (size_t i = 7; i < ts.size - 1; i++) {
                char c = ts.str[i];
                if (c == '<') depth++;
                else if (c == '>') depth--;
                else if (c == ',' && depth == 0) { chunk_end = i; break; }
            }
            size_t last_x = 0;
            depth = 0;
            for (size_t i = 7; i < chunk_end; i++) {
                char c = ts.str[i];
                if (c == '<') depth++;
                else if (c == '>') depth--;
                else if (c == 'x' && depth == 0) last_x = i;
            }
            if (last_x > 0) {
                string elem = (string){ .str = ts.str + last_x + 1, .size = chunk_end - last_x - 1 };
                append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                            create_string_attr(parser, str_lit("_source_type"), ts));
                result_types[0] = mlir_type_create_from_string(parser->ctx, elem);
            }
        }
    }

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        params->opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        operands.data, operands.size,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_memref_store_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    // Parse operands: store value, memref, [indices...]
    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle val = lookup_or_create_value(parser, reg, str_lit("unknown"));
        VecValue_push_back(parser->arena, &operands, val);
    }
    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle val = lookup_or_create_value(parser, reg, str_lit("unknown"));
        VecValue_push_back(parser->arena, &operands, val);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        bool first = true;
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string idx = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle val = lookup_or_create_value(parser, idx, str_lit("index"));
                VecValue_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    // memref.store operations do not have results
    MLIR_ValueHandle *results = MLIR_INVALID_HANDLE;
    size_t n_results = 0;

    // Parse attributes from both <{...}> and {...} blocks
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    // Parse type information and location after colon (memref.store doesn't have result types).
    // Capture the first type as `_source_type` so the classic printer can roundtrip
    // `memref.store %v, %m[i,j] : memref<...>`. The remaining tokens are skipped.
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        string source_type = (string){0};
        if (parse_type_string(parser, &source_type) && source_type.size > 0) {
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_string_attr(parser, str_lit("_source_type"), source_type));
        }
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
    }

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        params->opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        operands.data, operands.size,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_vector_print_op(Parser *parser, const OperationParserParams *params) {
    // Parse operands inside parentheses or as bare register(s)
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle val = lookup_or_create_value(parser, reg, str_lit("unknown"));
                VecValue_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    } else {
        while (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MLIR_ValueHandle val = lookup_or_create_value(parser, reg, str_lit("unknown"));
            VecValue_push_back(parser->arena, &operands, val);
            if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            else break;
        }
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        MLIR_INVALID_HANDLE,
        0,
        MLIR_INVALID_HANDLE,
        0,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}

OperationParserResult parse_std_constant_op(Parser *parser, const OperationParserParams *params) {
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            parser_next_token(parser);
        }
        parser_expect(parser, TK_RPAREN);
    }

    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        int depth = 1;
        while (depth > 0 && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LBRACE)) depth++;
            else if (parser_peek(parser, TK_RBRACE)) depth--;
            parser_next_token(parser);
        }
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                      params->op_type, MLIR_INVALID_HANDLE);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        MLIR_INVALID_HANDLE,
        0,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tt_reduce_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_LANGLE)) {
        parser_expect(parser, TK_LANGLE);
        if (parser_peek(parser, TK_LBRACE)) {
            parser_expect(parser, TK_LBRACE);
            while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME)) {
                    string attr_name = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                    if (parser_peek(parser, TK_EQUAL)) {
                        parser_expect(parser, TK_EQUAL);
                        if (parser_peek(parser, TK_INTEGER)) {
                            string int_val = parser_token_str(parser);
                            parser_expect(parser, TK_INTEGER);
                            int64_t val = 0;
                            for (size_t i = 0; i < int_val.size; i++) {
                                char c = int_val.str[i];
                                if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
                            }

                            string type_str = str_lit("");
                            if (parser_peek(parser, TK_COLON)) {
                                parser_expect(parser, TK_COLON);
                                if (parser_peek(parser, TK_NAME)) {
                                    type_str = parser_token_str(parser);
                                    parser_expect(parser, TK_NAME);
                                }
                            }
                            MLIR_TypeHandle attr_ty = mlir_type_create_from_string(
                                parser->ctx, type_str.size > 0 ? type_str : str_lit("i64"));
                            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                                        create_integer_attr(parser, attr_name, val, attr_ty));
                        }
                    }
                } else {
                    parser_next_token(parser);
                }
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RBRACE);
        }
        if (parser_peek(parser, TK_RANGLE)) parser_expect(parser, TK_RANGLE);
    }

    MLIR_RegionHandle *regions = MLIR_INVALID_HANDLE;
    size_t n_regions = 0;
    if (parser_peek(parser, TK_LPAREN_BRACE) || parser_peek(parser, TK_LPAREN)) {
        if (parser_peek(parser, TK_LPAREN_BRACE)) {
            parser_expect(parser, TK_LPAREN_BRACE);
        } else {
            parser_expect(parser, TK_LPAREN);
        }
        MLIR_RegionHandle region = MLIR_INVALID_HANDLE;
        if (parser_peek(parser, TK_LBRACE_END)) {
            region = parse_region(parser);
        }
        parser_expect(parser, TK_RPAREN);
        if (region) {
            regions = arena_new_array(params->arena, MLIR_RegionHandle, 1);
            regions[0] = region;
            n_regions = 1;
        }
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int depth = 0;
            while (!parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) {
                    if (depth == 0) break;
                    depth--;
                }
                parser_next_token(parser);
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        string type_str = str_lit("");
        if (parse_type_string(parser, &type_str)) {
            result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
            result_types[0] = mlir_type_create_from_string(params->ctx, type_str);
            n_result_types = 1;
        }
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                          params->op_type, MLIR_INVALID_HANDLE);
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        n_regions,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_cf_br_op(Parser *parser, const OperationParserParams *params) {
    // Parse cf.br ^bbX(%args : types)
    string target = str_lit("");
    if (parser_peek(parser, TK_CARET_NAME)) {
        target = parser_token_str(parser);
        parser_expect(parser, TK_CARET_NAME);
    }

    VecValue branch_args;
    VecValue_reserve(parser->arena, &branch_args, 4);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string vr = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle v = symbol_table_lookup(&parser->symbol_table, vr);
                if (v) VecValue_push_back(parser->arena, &branch_args, v);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                int angle = 0;
                while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                    if (parser_peek(parser, TK_LANGLE)) angle++;
                    else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                    parser_next_token(parser);
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Successor: resolve label to a (possibly forward-referenced) block.
    MLIR_BlockHandle target_block = parser_get_or_create_block(parser, target);
    MLIR_BlockHandle *successors = arena_new_array(parser->arena, MLIR_BlockHandle, 1);
    successors[0] = target_block;
    MLIR_ValueHandle **succ_operands_arrs = arena_new_array(parser->arena, MLIR_ValueHandle *, 1);
    succ_operands_arrs[0] = branch_args.data;
    size_t *succ_operand_counts = arena_new_array(parser->arena, size_t, 1);
    succ_operand_counts[0] = branch_args.size;

    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(parser->ctx,
        params->op_type,
        str_lit(""),
        MLIR_INVALID_HANDLE, 0,
        MLIR_INVALID_HANDLE, 0,
        params->lhs_results, params->n_lhs_results,
        MLIR_INVALID_HANDLE, 0,
        MLIR_INVALID_HANDLE, 0,
        successors, 1,
        succ_operands_arrs, succ_operand_counts,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}

OperationParserResult parse_cf_cond_br_op(Parser *parser, const OperationParserParams *params) {
    // Parse: cf.cond_br %cond, ^bbX[(args : types)], ^bbY[(args : types)] [loc]
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle cond = symbol_table_lookup(&parser->symbol_table, reg);
        if (cond) VecValue_push_back(parser->arena, &operands, cond);
    }

    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);

    string ttrue = str_lit("");
    string tfalse = str_lit("");
    VecValue true_args, false_args;
    VecValue_reserve(parser->arena, &true_args, 4);
    VecValue_reserve(parser->arena, &false_args, 4);

    if (parser_peek(parser, TK_CARET_NAME)) {
        ttrue = parser_token_str(parser);
        parser_expect(parser, TK_CARET_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string vr = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    MLIR_ValueHandle v = symbol_table_lookup(&parser->symbol_table, vr);
                    if (v) VecValue_push_back(parser->arena, &true_args, v);
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    int angle = 0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);

    if (parser_peek(parser, TK_CARET_NAME)) {
        tfalse = parser_token_str(parser);
        parser_expect(parser, TK_CARET_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string vr = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    MLIR_ValueHandle v = symbol_table_lookup(&parser->symbol_table, vr);
                    if (v) VecValue_push_back(parser->arena, &false_args, v);
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    int angle = 0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_BlockHandle true_blk = parser_get_or_create_block(parser, ttrue);
    MLIR_BlockHandle false_blk = parser_get_or_create_block(parser, tfalse);
    MLIR_BlockHandle *successors = arena_new_array(parser->arena, MLIR_BlockHandle, 2);
    successors[0] = true_blk;
    successors[1] = false_blk;
    MLIR_ValueHandle **succ_operands_arrs = arena_new_array(parser->arena, MLIR_ValueHandle *, 2);
    succ_operands_arrs[0] = true_args.data;
    succ_operands_arrs[1] = false_args.data;
    size_t *succ_operand_counts = arena_new_array(parser->arena, size_t, 2);
    succ_operand_counts[0] = true_args.size;
    succ_operand_counts[1] = false_args.size;

    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(parser->ctx,
        params->op_type,
        str_lit(""),
        MLIR_INVALID_HANDLE, 0,
        MLIR_INVALID_HANDLE, 0,
        params->lhs_results, params->n_lhs_results,
        operands.data, operands.size,
        MLIR_INVALID_HANDLE, 0,
        successors, 2,
        succ_operands_arrs, succ_operand_counts,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}

OperationParserResult parse_linalg_fill_op(Parser *parser, const OperationParserParams *params) {
    // Parse linalg.fill ins(%val : type) outs(%tensor : type) [-> result_type]
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    // Parse a parenthesized "ins" or "outs" list, capturing each `%reg : type`
    // into operands (regs become operands; types are kept implicitly via
    // the operand's MLIR_ValueHandle).
    for (int section = 0; section < 2; section++) {
        if (!(parser_peek(parser, TK_NAME) &&
              (str_eq(parser_token_str(parser), str_lit("ins")) ||
               str_eq(parser_token_str(parser), str_lit("outs"))))) break;
        parser_expect(parser, TK_NAME);
        if (!parser_peek(parser, TK_LPAREN)) break;
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle v = symbol_table_lookup(&parser->symbol_table, reg);
                if (v) VecValue_push_back(parser->arena, &operands, v);
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    string ty;
                    parse_type_string(parser, &ty);
                }
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    // Optional result type after `->` (linalg.fill on tensor returns).
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types,
                       &attributes, &n_attributes, &attributes_capacity,
                       params->op_type, MLIR_INVALID_HANDLE);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx,

        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
    };
    return out;
}

OperationParserResult parse_affine_load_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle memref_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!memref_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, memref_operand);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle idx = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!idx) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, idx);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("memref"))) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);
                string element_type = str_lit("unknown");

                if (parser_peek(parser, TK_TYPE_DIM)) {
                    string type_dim = parser_token_str(parser);
                    parser_expect(parser, TK_TYPE_DIM);
                    size_t last_x = SIZE_MAX;
                    for (size_t i = 0; i < type_dim.size; i++) {
                        if (type_dim.str[i] == 'x') last_x = i;
                    }
                    if (last_x != SIZE_MAX && last_x + 1 < type_dim.size) {
                        element_type = str_substr(type_dim, last_x + 1, type_dim.size - last_x - 1);
                    }
                }

                if (parser_peek(parser, TK_NAME)) {
                    element_type = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                }

                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, element_type);
                n_result_types = 1;

                if (parser_peek(parser, TK_RANGLE)) parser_expect(parser, TK_RANGLE);
            }
        }
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                          params->op_type, MLIR_INVALID_HANDLE);
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_index_constant_op(Parser *parser, const OperationParserParams *params) {
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    bool pending_int = false;
    int64_t pending_int_val = 0;

    // Parse the constant value if present
    if (parser_peek(parser, TK_INTEGER)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_INTEGER);
        int64_t parsed_value = 0;
        for (size_t i = 0; i < value_str.size; i++) {
            if (value_str.str[i] >= '0' && value_str.str[i] <= '9') {
                parsed_value = parsed_value * 10 + (value_str.str[i] - '0');
            }
        }
        pending_int = true;
        pending_int_val = parsed_value;
    } else {
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               !parser_peek(parser, TK_NAME) && !parser_peek(parser, TK_COLON)) {
            parser_next_token(parser);
        }
    }

    // Capture any additional attributes and result type information
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    if (n_result_types == 0) {
        result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
        result_types[0] = mlir_type_create_from_string(params->ctx, str_lit("index"));
        n_result_types = 1;
    }

    if (pending_int) {
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_integer_attr(parser, str_lit("value"), pending_int_val, result_types[0]));
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        MLIR_INVALID_HANDLE,
        0,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tensor_splat_op(Parser *parser, const OperationParserParams *params) {
    // Parse tensor.splat %value[%dim1, %dim2] : tensor<type>
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle value_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!value_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, value_operand);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle dim = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!dim) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, dim);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("tensor"))) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);
                strbuf tensor_type = strbuf_make();
                strbuf_append(parser->arena, &tensor_type, str_lit("tensor<"));
                int depth = 1;
                while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                    string token_str = parser_token_str(parser);
                    if (parser_peek(parser, TK_LANGLE)) depth++;
                    else if (parser_peek(parser, TK_RANGLE)) depth--;
                    strbuf_append(parser->arena, &tensor_type, token_str);
                    parser_next_token(parser);
                }

                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, strbuf_to_string(tensor_type));
                n_result_types = 1;
            }
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_arith_select_op(Parser *parser, const OperationParserParams *params) {
    // Parse arith.select %cond, %true_val, %false_val : cond_type, val_type
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    // Parse three operands
    for (int i = 0; i < 3; i++) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, operand);
            if (i < 2 && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        } else {
            parser_error(parser, str_lit("Expected register operand"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
    }

    // Parse any additional attributes
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    // Parse ": type1, type2" - result type is type2 (the value type)
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip first type (condition type)
        while (!parser_peek(parser, TK_COMMA) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE)) {
            parser_next_token(parser);
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);

            // Parse the second type (result type)
            strbuf result_type = strbuf_make();
            while (!parser_peek(parser, TK_EOF) &&
                   !parser_peek(parser, TK_NEWLINE) &&
                   !(parser_peek(parser, TK_NAME) &&
                     str_eq(parser_token_str(parser), str_lit("loc")))) {
                strbuf_append(parser->arena, &result_type, parser_token_str(parser));
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            if (result_type.size > 0) {
                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, strbuf_to_string(result_type));
                n_result_types = 1;
            }
        }
    }

    // Parse any remaining result types if not already parsed
    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);
    }

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      MLIR_INVALID_HANDLE, 0,
                                      op_location, params->unnumbered_loc_def,
                                      params->trailing_comment, params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tt_call_op(Parser *parser, const OperationParserParams *params) {
    // Parse tt.call @function(%args) : (arg_types) -> result_type

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_string_attr(parser, str_lit("callee"), fname));
    }

    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        if (parser_peek(parser, TK_LPAREN)) {
            int depth = 0;
            do {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            } while (depth > 0 && !parser_peek(parser, TK_EOF));
        }

        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);

            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   !(parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc")))) {
                string tok = parser_token_str(parser);
                result_type = result_type.size ? str_concat(parser->arena, result_type, tok) : tok;
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            if (result_type.size > 0) {
                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, result_type);
                n_result_types = 1;
            }
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tensor_collapse_shape_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle tensor_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!tensor_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, tensor_operand);
    }

    string reassoc = str_lit("");
    if (parser_peek(parser, TK_LBRACKET)) {
        int depth = 0;
        do {
            string tok = parser_token_str(parser);
            if (parser_peek(parser, TK_LBRACKET)) depth++;
            else if (parser_peek(parser, TK_RBRACKET)) depth--;
            reassoc = reassoc.size ? str_concat(parser->arena, reassoc, tok) : tok;
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF));
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 4;
    if (reassoc.size > 0) {
        attributes = arena_new_array(parser->arena, MLIR_AttributeHandle, attributes_capacity);
        attributes[n_attributes++] = create_string_attr(parser, str_lit("_reassoc"), reassoc);
    }

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("into")))) {
            parser_next_token(parser);
        }

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("into"))) {
            parser_expect(parser, TK_NAME);

            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   !(parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc")))) {
                string tok = parser_token_str(parser);
                result_type = result_type.size ? str_concat(parser->arena, result_type, tok) : tok;
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            if (result_type.size > 0) {
                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, result_type);
                n_result_types = 1;
            }
        }
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                          params->op_type, MLIR_INVALID_HANDLE);
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE,
        0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_generic_op(Parser *parser, const OperationParserParams *params) {
    string opname = (params->op_type == OP_TYPE_UNREGISTERED && params->opname.size > 0)
        ? params->opname
        : str_lit("");

    // Collect all data before creating operation (inlined from parse_generic_operation_impl)

    // Remember the start of this operation's source line
    int64_t source_line_start;
    {
        int64_t pos = (int64_t)parser->first;
        while (pos > 0) {
            unsigned char c = parser->input[pos - 1];
            if (c == '\n' || c == '\r') break;
            pos--;
        }
        source_line_start = pos;
    }

    // Skip non-structural tokens until operands/attrs/types
    while (!parser_peek(parser, TK_EOF) &&
           !(parser_peek(parser, TK_REGISTER) || parser_peek(parser, TK_LBRACKET) ||
             parser_peek(parser, TK_COLON) || parser_peek(parser, TK_LBRACE) ||
             parser_peek(parser, TK_LBRACE_END) || parser_peek(parser, TK_NEWLINE) ||
             parser_peek(parser, TK_LPAREN_BRACE))) {
        parser_next_token(parser);
    }

    // Parse operands
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    while ((parser_peek(parser, TK_REGISTER) || parser_peek(parser, TK_LBRACKET)) &&
           !parser_peek(parser, TK_COLON) && !parser_peek(parser, TK_LBRACE)) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, 0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, operand);
        } else if (parser_peek(parser, TK_LBRACKET)) {
            parser_expect(parser, TK_LBRACKET);
            while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str2 = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    MLIR_ValueHandle operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                    if (!operand2) {
                        parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                        OperationParserResult empty = {MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, 0};
                        return empty;
                    }
                    VecValue_push_back(parser->arena, &operands, operand2);
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RBRACKET);
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
        } else {
            break;
        }
    }

    // Collect attributes, result types, and location
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    MLIR_LocationHandle op_location = MLIR_INVALID_HANDLE;

    // Parse attributes and result types (inline parse_generic_attrs_and_result_type #1)
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);
    MLIR_LocationHandle loc1 = parse_optional_location(parser);
    if (loc1) op_location = loc1;

    // Capture trailing loc() if present
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op_location = parse_loc(parser);
    }

    // Parse regions
    MLIR_RegionHandle *regions = MLIR_INVALID_HANDLE;
    size_t n_regions = 0;

    while (!(parser_peek(parser, TK_LBRACE_END)
             || parser_peek(parser, TK_NEWLINE)
             || parser_peek(parser, TK_LPAREN_BRACE))) {
        parser_next_token(parser);
    }
    bool lparen_brace = false;
    if (parser_peek(parser, TK_LPAREN_BRACE)) {
        lparen_brace = true;
        parser_next_token(parser);
    }
    if (parser_peek(parser, TK_LBRACE_END)) {
        // Parse attributes/result types before region (inline parse_generic_attrs_and_result_type #2)
        parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
        parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);
        MLIR_LocationHandle loc2 = parse_optional_location(parser);
        if (loc2) op_location = loc2;

        MLIR_RegionHandle region = parse_region(parser);
        regions = arena_new_array(parser->arena, MLIR_RegionHandle, 1);
        regions[0] = region;
        n_regions = 1;

        // Parse attributes/result types after region (inline parse_generic_attrs_and_result_type #3)
        parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
        parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);
        MLIR_LocationHandle loc3 = parse_optional_location(parser);
        if (loc3) op_location = loc3;
    }
    if (lparen_brace) {
        parser_expect(parser, TK_RPAREN);
        while (!parser_peek(parser, TK_NEWLINE)) {
            parser_next_token(parser);
        }
    }

    // Final location check
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op_location = parse_loc(parser);
    }

    // Use fallback location if none was parsed
    if (!op_location) op_location = params->unnumbered_loc_def;

    // NOW create the operation with all collected data
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        opname,
        attributes, n_attributes,
        result_types, n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data, operands.size,
        regions, n_regions,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        source_line_start);

    // Finalize results
    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, MLIR_INVALID_HANDLE, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}



OperationParserResult parse_tt_func_op(Parser *parser, const OperationParserParams *params) {
    // Record source line start - will be set on the operation created at the end
    // Parse tt.func public @function_name(%arg0: type, %arg1: type, ...)

    // Initialize variables that will be used at the end
    MLIR_LocationHandle op_location = MLIR_INVALID_HANDLE;
    MLIR_RegionHandle func_region = MLIR_INVALID_HANDLE;
    MLIR_ValueHandle *func_operands = MLIR_INVALID_HANDLE;
    size_t n_func_operands = 0;

    // Vector to collect function arguments as we parse them
    VecValue func_args;
    VecValue_reserve(parser->arena, &func_args, 8);

    // Vector to collect per-argument attributes (parallel to func_args)
    // Each element is a DictionaryAttr containing the attributes for that argument
    VecAttribute arg_attr_dicts;
    VecAttribute_reserve(parser->arena, &arg_attr_dicts, 8);

    // Capture visibility keyword if present
    string visibility = str_lit("private");  // default
    if (parser_peek(parser, TK_NAME) && (str_eq(parser_token_str(parser), str_lit("public")) || str_eq(parser_token_str(parser), str_lit("private")))) {
        visibility = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }

    MLIR_AttributeHandle *attrs = MLIR_INVALID_HANDLE; size_t n_attrs = 0, cap_attrs = 0;
    append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("visibility"), visibility));

    // Parse @function_name
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture function symbol name into an attribute for printing
        string fname_with_at = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);

        // Remove the '@' prefix to get just the function name
        string fname = str_substr(fname_with_at, 1, fname_with_at.size - 1);

        // Store as string attribute 'sym_name'
        append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("sym_name"), fname));
    } else {
        // Try to skip tokens until we find a parenthesis
        while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
            parser_next_token(parser);
        }
    }

    // Parse function arguments
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        // Arguments will be collected in the func_args vector

        // Accumulate a textual params signature when args are type-only (declarations)
        strbuf params_sig = strbuf_make();

        // Parse arguments directly in one pass
        int paren_depth = 0;
        while (!parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LPAREN)) {
                paren_depth++;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_RPAREN)) {
                if (paren_depth == 0) {
                    // Found the main closing paren
                    break;
                }
                paren_depth--;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);

                // Parse colon and type
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);

                    // Collect all information before creating the value
                    MLIR_TypeHandle arg_type = MLIR_INVALID_HANDLE;
                    bool has_div = false;
                    int64_t div_value = 0;
                    bool has_max_div = false;
                    int64_t max_div_value = 0;
                    MLIR_LocationHandle arg_location = MLIR_INVALID_HANDLE;

                    // Parse the argument type
                    if (parser_peek(parser, TK_EXCLAMATION)) {
                        parser_expect(parser, TK_EXCLAMATION);

                        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                            string type_name = parser_token_str(parser);
                            if (parser_peek(parser, TK_NAME)) {
                                parser_expect(parser, TK_NAME);
                            } else {
                                parser_expect(parser, TK_NAME_DOT_NAME);
                            }

                            // Handle complex types like !tt.ptr<f32, 1>
                            if (str_eq(type_name, str_lit("tt.ptr"))) {
                                if (parser_peek(parser, TK_LANGLE)) {
                                    parser_expect(parser, TK_LANGLE);

                                    // Parse everything inside angle brackets
                                    strbuf type_content = strbuf_make();
                                    while (!parser_peek(parser, TK_RANGLE) && !parser_peek(parser, TK_EOF)) {
                                        string token_str = parser_token_str(parser);
                                        strbuf_append(parser->arena, &type_content, token_str);
                                        parser_next_token(parser);
                                    }

                                    if (parser_peek(parser, TK_RANGLE)) {
                                        parser_expect(parser, TK_RANGLE);
                                        string full_type_str = str_concat(parser->arena, str_lit("!tt.ptr<"), str_concat(parser->arena, strbuf_to_string(type_content), str_lit(">")));
                                        arg_type = mlir_type_create_from_string(parser->ctx, full_type_str);
                                    }
                                } else {
                                    arg_type = mlir_type_create_from_string(parser->ctx, str_lit("!tt.ptr"));
                                }
                            } else {
                                arg_type = mlir_type_create_from_string(parser->ctx, str_concat(parser->arena, str_lit("!"), type_name));
                            }
                        } else {
                            arg_type = mlir_type_create_from_string(parser->ctx, str_lit("!unknown"));
                        }
                    } else if (parser_peek(parser, TK_NAME)) {
                        // Simple type like i32 or complex type like tensor<2x256xf32>
                        string type_name = parser_token_str(parser);
                        parser_expect(parser, TK_NAME);

                        // Handle complex types with angle brackets (e.g., tensor<2x256xf32>)
                        if (parser_peek(parser, TK_LANGLE)) {
                            parser_expect(parser, TK_LANGLE);
                            strbuf angle_content = strbuf_make();
                            int bracket_depth = 1;
                            while (bracket_depth > 0 && !parser_peek(parser, TK_EOF)) {
                                if (parser_peek(parser, TK_LANGLE)) {
                                    bracket_depth++;
                                } else if (parser_peek(parser, TK_RANGLE)) {
                                    bracket_depth--;
                                    if (bracket_depth == 0) {
                                        parser_expect(parser, TK_RANGLE);
                                        break;
                                    }
                                }
                                string token_str = parser_token_str(parser);
                                strbuf_append(parser->arena, &angle_content, token_str);
                                parser_next_token(parser);
                            }
                            // Reconstruct the full type string
                            string full_type = str_concat(parser->arena, type_name, str_concat(parser->arena, str_lit("<"), str_concat(parser->arena, strbuf_to_string(angle_content), str_lit(">"))));
                            arg_type = mlir_type_create_from_string(parser->ctx, full_type);
                        } else {
                            arg_type = mlir_type_create_from_string(parser->ctx, type_name);
                        }
                    } else {
                        arg_type = mlir_type_create_from_string(parser->ctx, str_lit("unknown"));
                    }

                    // Optional per-arg attribute dict: { ... }
                    if (parser_peek(parser, TK_LBRACE)) {
                        int depth = 0;
                        // Try to detect simple 'tt.divisibility = <int> : <type>' pattern
                        // without constructing a full attribute AST.
                        // Consume '{'
                        parser_expect(parser, TK_LBRACE);
                        depth++;
                        // Read tokens until matching '}'
                        while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                            if (parser_peek(parser, TK_LBRACE)) { depth++; parser_next_token(parser); continue; }
                            if (parser_peek(parser, TK_RBRACE)) { depth--; parser_next_token(parser); if (depth==0) break; else continue; }
                            // Match name possibly with dots
                            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                                string name = parser_token_str(parser);
                                parser_next_token(parser);
                                if (str_eq(name, str_lit("tt.divisibility"))) {
                                    // Expect '=' integer ':' type
                                    if (parser_peek(parser, TK_EQUAL)) parser_expect(parser, TK_EQUAL);
                                    if (parser_peek(parser, TK_INTEGER)) {
                                        string ival = parser_token_str(parser);
                                        // parse integer
                                        int64_t v = 0; for (size_t k=0;k<ival.size;k++){ char c=ival.str[k]; if (c>='0' && c<='9') v = v*10 + (c-'0'); }
                                        parser_expect(parser, TK_INTEGER);
                                        // optional ':' type
                                        if (parser_peek(parser, TK_COLON)) {
                                            parser_expect(parser, TK_COLON);
                                            string tstr = str_lit("");
                                            // Reuse type parsing helper from mlir_parser.c via forward decl
                                            // Here, mimic parse_type_string: collect until '}' or ',' or 'loc'
                                            int angle = 0;
                                            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_COMMA)) {
                                                if (parser_peek(parser, TK_LANGLE)) angle++;
                                                else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                                                if (angle==0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                                                string tok = parser_token_str(parser);
                                                tstr = tstr.size ? str_concat(parser->arena, tstr, tok) : tok;
                                                parser_next_token(parser);
                                                if (angle==0 && (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_COMMA))) break;
                                            }
                                        }
                                        // Store in temporaries instead of setting on value
                                        has_div = true;
                                        div_value = v;
                                    }
                                } else if (str_eq(name, str_lit("tt.max_divisibility"))) {
                                    // Expect '=' integer ':' type
                                    if (parser_peek(parser, TK_EQUAL)) parser_expect(parser, TK_EQUAL);
                                    if (parser_peek(parser, TK_INTEGER)) {
                                        string ival = parser_token_str(parser);
                                        // parse integer
                                        int64_t v = 0; for (size_t k=0;k<ival.size;k++){ char c=ival.str[k]; if (c>='0' && c<='9') v = v*10 + (c-'0'); }
                                        parser_expect(parser, TK_INTEGER);
                                        // optional ':' type
                                        if (parser_peek(parser, TK_COLON)) {
                                            parser_expect(parser, TK_COLON);
                                            string tstr = str_lit("");
                                            int angle = 0;
                                            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_COMMA)) {
                                                if (parser_peek(parser, TK_LANGLE)) angle++;
                                                else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                                                if (angle==0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                                                string tok = parser_token_str(parser);
                                                tstr = tstr.size ? str_concat(parser->arena, tstr, tok) : tok;
                                                parser_next_token(parser);
                                                if (angle==0 && (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_COMMA))) break;
                                            }
                                        }
                                        // Store in temporaries instead of setting on value
                                        has_max_div = true;
                                        max_div_value = v;
                                    }
                                }
                            } else {
                                parser_next_token(parser);
                            }
                        }
                    }

                    // Optional trailing per-arg loc()
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                        arg_location = parse_loc(parser);
                    }

                    // Now create the value with all information
                    MLIR_ValueHandle arg = MLIR_CreateValueBlockArg(parser->ctx, reg_str, (uint32_t)func_args.size, arg_type, arg_location);

                    // Build structured dictionary attribute for this argument
                    VecAttribute dict_elems;
                    VecAttribute_reserve(parser->arena, &dict_elems, 2);

                    MLIR_TypeHandle i32_ty = mlir_type_create_from_string(parser->ctx, str_lit("i32"));
                    if (has_div) {
                        MLIR_AttributeHandle div_attr = MLIR_CreateAttributeInteger(parser->ctx, str_lit("tt.divisibility"), div_value, i32_ty);
                        VecAttribute_push_back(parser->arena, &dict_elems, div_attr);
                    }
                    if (has_max_div) {
                        MLIR_AttributeHandle max_div_attr = MLIR_CreateAttributeInteger(parser->ctx, str_lit("tt.max_divisibility"), max_div_value, i32_ty);
                        VecAttribute_push_back(parser->arena, &dict_elems, max_div_attr);
                    }

                    // Create dictionary attribute (empty if no attributes)
                    MLIR_AttributeHandle arg_dict = MLIR_CreateAttributeDict(parser->ctx, str_lit(""), dict_elems.data, dict_elems.size);

                    VecValue_push_back(parser->arena, &func_args, arg);
                    VecAttribute_push_back(parser->arena, &arg_attr_dicts, arg_dict);
                }
            } else if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
                // Type-only param (no SSA name); accumulate its textual form
                string ty = str_lit("");
                if (parse_type_string(parser, &ty)) {
                    if (params_sig.size > 0) strbuf_append(parser->arena, &params_sig, str_lit(", "));
                    strbuf_append(parser->arena, &params_sig, ty);
                } else {
                    parser_next_token(parser);
                }
            } else {
                parser_next_token(parser);
            }
        }

        // Store operands for later assignment to operation
        if (func_args.size > 0) {
            func_operands = func_args.data;
            n_func_operands = func_args.size;
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        } else {
        }

        // Save params signature if any
        if (params_sig.size > 0) {
            append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("params_sig"), strbuf_to_string(params_sig)));
        }
    }

    // Optional return signature after '->' (capture conservatively until 'attributes' or body)
    strbuf ret_sig = strbuf_make();
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
            if (parser_peek(parser, TK_NAME)) {
                string nm = parser_token_str(parser);
                if (str_eq(nm, str_lit("attributes")) || str_eq(nm, str_lit("loc"))) break;
            }
            strbuf_append(parser->arena, &ret_sig, parser_token_str(parser));
            parser_next_token(parser);
        }
    }
    (void)ret_sig;

    // Optionally capture function attributes: attributes { ... }
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("attributes"))) {
        parser_expect(parser, TK_NAME); // attributes
        if (parser_peek(parser, TK_LBRACE)) {
            int depth = 0; parser_expect(parser, TK_LBRACE); depth++;
            // Conservatively skip content; generic op attribute parser can be used later if needed
            while (depth>0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LBRACE)) { depth++; }
                else if (parser_peek(parser, TK_RBRACE)) { depth--; }
                parser_next_token(parser);
            }
        }
    }

    // If there is a function body, parse it; otherwise leave as a declaration
    if (parser_peek(parser, TK_LBRACE_END)) {
        // Parse the function body region with special handling for function arguments
        parser_expect(parser, TK_LBRACE_END);
        parser_expect(parser, TK_NEWLINE);

        // Push new scope for this region
        symbol_table_push_scope(parser->arena, &parser->symbol_table);

        // Register function arguments in the new scope
        for (size_t i = 0; i < func_args.size; i++) {
            MLIR_ValueHandle operand = func_args.data[i];
            if (!operand) continue;
            string name = MLIR_GetValueRegisterName(operand);
            if (name.size > 0) {
                symbol_table_add_value(parser->arena, &parser->symbol_table, name, operand);
            }
        }

        // Parse blocks
        VecBlock blocks;
        VecBlock_reserve(parser->arena, &blocks, 8);
        while (!parser_peek(parser, TK_RBRACE)) {
            MLIR_BlockHandle block = parse_block(parser);
            VecBlock_push_back(parser->arena, &blocks, block);
        }
        parser_expect(parser, TK_RBRACE);

        // Pop scope when leaving region
        symbol_table_pop_scope(&parser->symbol_table);

        // Create region
        MLIR_RegionHandle region = MLIR_CreateRegion(parser->ctx);
        for (size_t i = 0; i < blocks.size; i++) {
            MLIR_AppendRegionBlock(parser->ctx, region, blocks.data[i]);
        }

        // Store region for later assignment to operation
        func_region = region;

        if (parser_peek(parser, TK_NAME)) {
            if (str_eq(parser_token_str(parser), str_lit("loc"))) {
                op_location = parse_loc(parser);
            }
        }
    } else {
        // Declaration: no body. Optionally consume trailing loc
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            op_location = parse_loc(parser);
        }
    }

    // Parse any additional attributes and result types using helper functions
    parse_angle_brace_attributes(parser, &attrs, &n_attrs, &cap_attrs);
    parse_brace_attributes(parser, &attrs, &n_attrs, &cap_attrs);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attrs, &n_attrs, &cap_attrs, params->op_type, MLIR_INVALID_HANDLE);

    // Parse optional location if not already set
    if (!op_location) {
        op_location = parse_optional_location(parser);
        if (!op_location) {
            op_location = params->unnumbered_loc_def;
        }
    }

    // Set regions
    MLIR_RegionHandle *regions = MLIR_INVALID_HANDLE;
    size_t n_regions = 0;
    if (func_region) {
        regions = arena_new_array(params->arena, MLIR_RegionHandle, 1);
        regions[0] = func_region;
        n_regions = 1;
    }

    // Create the operation at the end
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, params->op_type, str_lit(""),
                                      attrs, n_attrs,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      func_operands, n_func_operands,
                                      regions, n_regions,
                                      op_location, params->unnumbered_loc_def,
                                      params->trailing_comment, params->source_line_start);

    // Add arg_attrs array attribute to the operation if there are any argument attributes
    if (arg_attr_dicts.size > 0) {
        // Create ArrayAttr containing all DictionaryAttrs
        MLIR_AttributeHandle arg_attrs = MLIR_CreateAttributeArray(parser->ctx, 
            str_lit("arg_attrs"), arg_attr_dicts.data, arg_attr_dicts.size);
        MLIR_AppendOpAttribute(parser->ctx, op, arg_attrs);
    }

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_scf_if_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_REGISTER)) {
        string cond_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle condition = symbol_table_lookup(&parser->symbol_table, cond_str);
        if (!condition) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, condition);
    } else {
        parser_error(parser, str_lit("Expected condition register for scf.if"), parser->first, parser->last);
        OperationParserResult empty = {0};
        return empty;
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("attributes"))) {
        parser_expect(parser, TK_NAME);
        parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    }

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            size_t cap = 2;
            result_types = arena_new_array(parser->arena, MLIR_TypeHandle, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string type_str = str_lit("");
                if (!parse_type_string(parser, &type_str)) break;
                if (n_result_types >= cap) {
                    cap *= 2;
                    MLIR_TypeHandle *expanded = arena_new_array(parser->arena, MLIR_TypeHandle, cap);
                    for (size_t i = 0; i < n_result_types; i++) expanded[i] = result_types[i];
                    result_types = expanded;
                }
                result_types[n_result_types++] = mlir_type_create_from_string(params->ctx, type_str);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            if (n_result_types == 0) result_types = MLIR_INVALID_HANDLE;
        } else {
            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, type_str);
                n_result_types = 1;
            }
        }
    }

    MLIR_RegionHandle then_region = parse_region(parser);
    MLIR_RegionHandle else_region = MLIR_INVALID_HANDLE;
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("else"))) {
        parser_expect(parser, TK_NAME);
        else_region = parse_region(parser);
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    size_t n_regions = else_region ? 2 : 1;
    MLIR_RegionHandle *regions = arena_new_array(parser->arena, MLIR_RegionHandle, n_regions);
    regions[0] = then_region;
    if (else_region) regions[1] = else_region;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        n_regions,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_scf_for_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    MLIR_ValueHandle loop_var = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle iv_type = MLIR_INVALID_HANDLE;

    if (parser_peek(parser, TK_REGISTER)) {
        string loop_var_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        loop_var = MLIR_CreateValueBlockArg(parser->ctx, loop_var_name, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE);

        parser_expect(parser, TK_EQUAL);

        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MLIR_ValueHandle start_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!start_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, start_operand);
        }

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
        }

        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MLIR_ValueHandle end_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!end_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, end_operand);
        }

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
        }

        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MLIR_ValueHandle step_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!step_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, step_operand);
        }
    }

    VecValue iter_vars;
    VecValue_reserve(parser->arena, &iter_vars, 4);

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("iter_args"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (!parser_peek(parser, TK_REGISTER)) break;
                string iter_var_name = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);

                MLIR_ValueHandle iter_var = MLIR_CreateValueBlockArg(parser->ctx, iter_var_name, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE);
                VecValue_push_back(parser->arena, &iter_vars, iter_var);

                parser_expect(parser, TK_EQUAL);
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    consume_optional_hash_selector(parser);
                    consume_optional_hash_selector(parser);
                    MLIR_ValueHandle init_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                    if (!init_operand) {
                        parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                        OperationParserResult empty = {0};
                        return empty;
                    }
                    VecValue_push_back(parser->arena, &operands, init_operand);
                }

                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                } else {
                    break;
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    MLIR_TypeHandle *iter_result_types = MLIR_INVALID_HANDLE;
    size_t n_iter_results = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            size_t cap = 4;
            iter_result_types = arena_new_array(params->arena, MLIR_TypeHandle, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string t = str_lit("");
                if (!parse_type_string(parser, &t)) break;
                if (n_iter_results >= cap) {
                    cap *= 2;
                    MLIR_TypeHandle *tmp = arena_new_array(params->arena, MLIR_TypeHandle, cap);
                    for (size_t i = 0; i < n_iter_results; i++) tmp[i] = iter_result_types[i];
                    iter_result_types = tmp;
                }
                iter_result_types[n_iter_results++] = mlir_type_create_from_string(params->ctx, t);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            if (n_iter_results == 0) iter_result_types = MLIR_INVALID_HANDLE;
        }
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        string t = str_lit("");
        if (parse_type_string(parser, &t)) {
            iv_type = mlir_type_create_from_string(parser->ctx, t);
        }
    }

    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    MLIR_BlockHandle block = MLIR_CreateBlock(parser->ctx);
    if (loop_var) {
        string loop_name = MLIR_GetValueRegisterName(loop_var);
        MLIR_TypeHandle arg_type;
        if (iv_type) {
            arg_type = iv_type;
        } else if (operands.size > 0 && operands.data[0] && MLIR_GetValueType(operands.data[0])) {
            arg_type = MLIR_GetValueType(operands.data[0]);
        } else {
            arg_type = mlir_type_create_from_string(parser->ctx, str_lit("index"));
        }
        MLIR_ValueHandle iv_block_arg = MLIR_CreateValueBlockArg(parser->ctx, loop_name, 0, arg_type, MLIR_INVALID_HANDLE);
        MLIR_AppendBlockArg(parser->ctx, block, iv_block_arg);
        if (loop_name.size > 0) symbol_table_add_value(parser->arena, &parser->symbol_table, loop_name, iv_block_arg);
    }

    for (size_t i = 0; i < iter_vars.size; i++) {
        MLIR_ValueHandle iter_var = iter_vars.data[i];
        string iter_name = MLIR_GetValueRegisterName(iter_var);
        size_t init_index = 3 + i;
        MLIR_TypeHandle arg_type;
        if (operands.size > init_index) {
            MLIR_ValueHandle init_operand = operands.data[init_index];
            if (init_operand && MLIR_GetValueType(init_operand)) {
                arg_type = MLIR_GetValueType(init_operand);
            } else {
                arg_type = mlir_type_create_from_string(parser->ctx, str_lit("unknown"));
            }
        } else {
            arg_type = mlir_type_create_from_string(parser->ctx, str_lit("unknown"));
        }
        MLIR_ValueHandle iter_block_arg = MLIR_CreateValueBlockArg(parser->ctx, iter_name, (uint32_t)(i + 1), arg_type, MLIR_INVALID_HANDLE);
        MLIR_AppendBlockArg(parser->ctx, block, iter_block_arg);
        if (iter_name.size > 0) symbol_table_add_value(parser->arena, &parser->symbol_table, iter_name, iter_block_arg);
    }

    bool prev_flag = parser->capture_trailing_comments;
    parser->capture_trailing_comments = true;
    while (!parser_peek(parser, TK_RBRACE)) {
        MLIR_OpHandle inner = parse_operation(parser);
        MLIR_AppendBlockOp(parser->ctx, block, inner);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    parser->capture_trailing_comments = prev_flag;

    symbol_table_pop_scope(&parser->symbol_table);

    MLIR_RegionHandle region = MLIR_CreateRegion(parser->ctx);
    MLIR_AppendRegionBlock(parser->ctx, region, block);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MLIR_RegionHandle *regions = arena_new_array(params->arena, MLIR_RegionHandle, 1);
    regions[0] = region;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        MLIR_INVALID_HANDLE,
        0,
        iter_result_types,
        n_iter_results,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        1,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    MLIR_ValueHandle *results = MLIR_INVALID_HANDLE;
    size_t n_results = 0;
    if (n_iter_results > 0 && params->n_lhs_results > 0) {
        // lhs_results now contains the correct count matching n_iter_results
        // (expanded when parsing %reg:N syntax)
        results = params->lhs_results;
        n_results = params->n_lhs_results;

        // Use the lhs_results array as the op's results. Result Values that
        // were created without explicit names keep their handles so the
        // printer can auto-assign sequential names to them (matching the
        // upstream backend's behavior); result types are auto-synced from
        // n_iter_results by MLIR_CreateOp.
    }

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}
OperationParserResult parse_scf_while_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (!parser_peek(parser, TK_REGISTER)) break;
            parser_expect(parser, TK_REGISTER);
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MLIR_ValueHandle init = symbol_table_lookup(&parser->symbol_table, reg);
                if (!init) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, init);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int depth = 1;
            while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            }
            if (parser_peek(parser, TK_RPAREN)) parser_expect(parser, TK_RPAREN);
        } else {
            string tmp = str_lit("");
            (void)parse_type_string(parser, &tmp);
        }
    }

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            size_t cap = 2;
            result_types = arena_new_array(params->arena, MLIR_TypeHandle, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string t = str_lit("");
                if (!parse_type_string(parser, &t)) break;
                if (n_result_types >= cap) {
                    cap *= 2;
                    MLIR_TypeHandle *tmp = arena_new_array(params->arena, MLIR_TypeHandle, cap);
                    for (size_t i = 0; i < n_result_types; i++) tmp[i] = result_types[i];
                    result_types = tmp;
                }
                result_types[n_result_types++] = mlir_type_create_from_string(params->ctx, t);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            if (n_result_types == 0) result_types = MLIR_INVALID_HANDLE;
        } else {
            string t = str_lit("");
            if (parse_type_string(parser, &t)) {
                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, t);
                n_result_types = 1;
            }
        }
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    MLIR_RegionHandle cond_region = parse_region(parser);

    MLIR_RegionHandle body_region = MLIR_INVALID_HANDLE;
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("do"))) {
        parser_expect(parser, TK_NAME);
        body_region = parse_region(parser);
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    size_t n_regions = body_region ? 2 : 1;
    MLIR_RegionHandle *regions = arena_new_array(params->arena, MLIR_RegionHandle, n_regions);
    regions[0] = cond_region;
    if (body_region) regions[1] = body_region;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        n_regions,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_gpu_launch_op(Parser *parser, const OperationParserParams *params) {
    // Parse gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %c1, %arg10 = %c1)
    //                  threads(%arg5, %arg6, %arg7) in (%arg11 = %1, %arg12 = %c1, %arg13 = %c1) {

    VecValue launch_args;
    VecValue_reserve(parser->arena, &launch_args, 16);
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 8);

    MLIR_TypeHandle index_ty = mlir_type_create_from_string(parser->ctx, str_lit("index"));

    // Helper-ish inline: parse "kw(%a, %b, %c)" producing 3 block args of `index` type.
    string kw_blocks = str_lit("blocks");
    string kw_threads = str_lit("threads");
    string kw_in = str_lit("in");
    for (int section = 0; section < 2; section++) {
        string kw = (section == 0) ? kw_blocks : kw_threads;
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), kw)) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_LPAREN)) {
                parser_expect(parser, TK_LPAREN);
                while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                    if (parser_peek(parser, TK_REGISTER)) {
                        string reg_str = parser_token_str(parser);
                        parser_expect(parser, TK_REGISTER);
                        MLIR_ValueHandle arg = MLIR_CreateValueBlockArg(parser->ctx, reg_str, (uint32_t)launch_args.size, index_ty, MLIR_INVALID_HANDLE);
                        VecValue_push_back(parser->arena, &launch_args, arg);
                        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                    } else {
                        parser_next_token(parser);
                    }
                }
                parser_expect(parser, TK_RPAREN);
            }
        }

        // Optional `in (%argN = %op, ...)` clause: 3 extra block args + 3 operands.
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), kw_in)) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_LPAREN)) {
                parser_expect(parser, TK_LPAREN);
                while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                    if (parser_peek(parser, TK_REGISTER)) {
                        string arg_reg = parser_token_str(parser);
                        parser_expect(parser, TK_REGISTER);
                        MLIR_ValueHandle arg = MLIR_CreateValueBlockArg(parser->ctx, arg_reg, (uint32_t)launch_args.size, index_ty, MLIR_INVALID_HANDLE);
                        VecValue_push_back(parser->arena, &launch_args, arg);

                        if (parser_peek(parser, TK_EQUAL)) parser_expect(parser, TK_EQUAL);

                        if (parser_peek(parser, TK_REGISTER)) {
                            string op_reg = parser_token_str(parser);
                            parser_expect(parser, TK_REGISTER);
                            MLIR_ValueHandle val = symbol_table_lookup(&parser->symbol_table, op_reg);
                            if (!val) {
                                val = lookup_or_create_value(parser, op_reg, str_lit("index"));
                            }
                            VecValue_push_back(parser->arena, &operands, val);
                        }

                        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                    } else {
                        parser_next_token(parser);
                    }
                }
                parser_expect(parser, TK_RPAREN);
            }
        }
    }

    // Parse any attributes and result types using helper functions
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Skip until opening brace
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }

    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    MLIR_BlockHandle gpu_block = MLIR_CreateBlock(parser->ctx);
    for (size_t i = 0; i < launch_args.size; i++) {
        MLIR_ValueHandle arg = launch_args.data[i];
        MLIR_AppendBlockArg(parser->ctx, gpu_block, arg);
    }

    symbol_table_push_scope(parser->arena, &parser->symbol_table);
    for (size_t i = 0; i < launch_args.size; i++) {
        string name = MLIR_GetValueRegisterName(launch_args.data[i]);
        if (name.size > 0) symbol_table_add_value(parser->arena, &parser->symbol_table, name, launch_args.data[i]);
    }

    while (!parser_peek(parser, TK_RBRACE)) {
        MLIR_OpHandle inner_op = parse_operation(parser);
        MLIR_AppendBlockOp(parser->ctx, gpu_block, inner_op);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    symbol_table_pop_scope(&parser->symbol_table);

    MLIR_RegionHandle gpu_region = MLIR_CreateRegion(parser->ctx);
    MLIR_AppendRegionBlock(parser->ctx, gpu_region, gpu_block);

    // Set regions
    MLIR_RegionHandle *regions = arena_new_array(params->arena, MLIR_RegionHandle, 1);
    regions[0] = gpu_region;
    size_t n_regions = 1;

    // Create the operation at the end
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      regions, n_regions,
                                      op_location, params->unnumbered_loc_def,
                                      params->trailing_comment, params->source_line_start);

    // Create results from the operation's result types
    MLIR_ValueHandle *results = MLIR_INVALID_HANDLE;
    size_t n_results = 0;

    results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_arith_cmpi_op(Parser *parser, const OperationParserParams *params) {
    // Parse arith.cmpi <predicate>, %lhs, %rhs : type
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    // Parse predicate (slt, sge, etc.)
    string predicate = str_lit("slt"); // default
    if (parser_peek(parser, TK_NAME)) {
        predicate = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0, attributes_capacity = 0;
    append_attr(parser, &attributes, &n_attributes, &attributes_capacity, create_string_attr(parser, str_lit("predicate"), predicate));

    // Expect comma
    parser_expect(parser, TK_COMMA);

    // Parse operands
    if (parser_peek(parser, TK_REGISTER)) {
        string lhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle lhs = symbol_table_lookup(&parser->symbol_table, lhs_str);
        if (!lhs) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, lhs);
    }

    parser_expect(parser, TK_COMMA);

    if (parser_peek(parser, TK_REGISTER)) {
        string rhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle rhs = symbol_table_lookup(&parser->symbol_table, rhs_str);
        if (!rhs) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, rhs);
    }

    // Parse any additional attributes and result types using helper functions
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);

    // Default result type if not specified
    if (n_result_types == 0) {
        result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
        result_types[0] = mlir_type_create_from_string(params->ctx, str_lit("i1"));
        n_result_types = 1;
    }

    // Parse optional location
    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      MLIR_INVALID_HANDLE, 0,
                                      op_location, params->unnumbered_loc_def,
                                      params->trailing_comment, params->source_line_start);

    // Create results from the operation's result types
    MLIR_ValueHandle *results = MLIR_INVALID_HANDLE;
    size_t n_results = 0;

    results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}


OperationParserResult parse_scf_yield_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    // Handle optional type information after all operands
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // Parse type list until we hit loc() or end
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                break;
            }
            parser_next_token(parser);
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        MLIR_INVALID_HANDLE, 0,
        MLIR_INVALID_HANDLE, 0,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}

OperationParserResult parse_return_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    // Handle optional parentheses
    bool has_parens = false;
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        has_parens = true;
    }

    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MLIR_ValueHandle operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    if (has_parens) {
        parser_expect(parser, TK_RPAREN);
    }

    // Handle optional type information
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // Skip type information
        while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE)) {
            parser_next_token(parser);
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    bool keep_operands = params->op_type != OP_TYPE_STD_RETURN;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        MLIR_INVALID_HANDLE, 0,
        MLIR_INVALID_HANDLE, 0,
        params->lhs_results,
        params->n_lhs_results,
        (keep_operands && operands.size > 0) ? operands.data : MLIR_INVALID_HANDLE,
        keep_operands ? operands.size : 0,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}


OperationParserResult parse_tt_load_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    while (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, true)) {
            OperationParserResult empty = {0};
            return empty;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

        string pointer_sig = str_lit("");
        if (parse_type_string(parser, &pointer_sig)) {
            string result_sig = pointer_sig;

            while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
            if (parser_peek(parser, TK_ARROW)) {
                parser_expect(parser, TK_ARROW);
                while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
                string explicit_result = str_lit("");
                if (parse_type_string(parser, &explicit_result)) {
                    result_sig = explicit_result;
                }
            } else {
                size_t pos = (size_t)-1;
                const size_t needle_len = 8; // strlen("!tt.ptr<")
                for (size_t i = 0; i + needle_len <= pointer_sig.size; i++) {
                    if (pointer_sig.str[i] == '!' &&
                        pointer_sig.str[i + 1] == 't' &&
                        pointer_sig.str[i + 2] == 't' &&
                        pointer_sig.str[i + 3] == '.' &&
                        pointer_sig.str[i + 4] == 'p' &&
                        pointer_sig.str[i + 5] == 't' &&
                        pointer_sig.str[i + 6] == 'r' &&
                        pointer_sig.str[i + 7] == '<') {
                        pos = i;
                        break;
                    }
                }

                if (pos != (size_t)-1) {
                    size_t start = pos + needle_len;
                    size_t end = start;
                    while (end < pointer_sig.size && pointer_sig.str[end] != '>' && pointer_sig.str[end] != ',') end++;
                    string elem = str_substr(pointer_sig, start, end - start);

                    size_t close = end;
                    while (close < pointer_sig.size && pointer_sig.str[close] != '>') close++;
                    if (close < pointer_sig.size) close++;

                    string before = str_substr(pointer_sig, 0, pos);
                    string after = str_substr(pointer_sig, close, pointer_sig.size - close);

                    strbuf val_sig = strbuf_make();
                    strbuf_append(parser->arena, &val_sig, before);
                    strbuf_append(parser->arena, &val_sig, elem.size > 0 ? elem : str_lit("f32"));
                    strbuf_append(parser->arena, &val_sig, after);
                    result_sig = strbuf_to_string(val_sig);
                }
            }

            if (result_sig.size > 0) {
                result_types = arena_new_array(params->arena, MLIR_TypeHandle, 1);
                result_types[0] = mlir_type_create_from_string(params->ctx, result_sig);
                n_result_types = 1;
            }
        }

        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
        if (parser_peek(parser, TK_COMMA)) {
            do {
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE));
        }
    }

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    size_t n_results = 0;
    MLIR_ValueHandle *results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results
    };
    return out;
}

OperationParserResult parse_tt_store_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    // Parse operands: pointer, value, mask
    while (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, true)) {
            OperationParserResult empty = {0};
            return empty;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    // Custom attribute parsing for tt.store to handle ellipsis and create proper integer attributes
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        while (!parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_NAME)) {
                string attr_name = parser_token_str(parser);
                parser_expect(parser, TK_NAME);
                parser_expect(parser, TK_EQUAL);

                if (parser_peek(parser, TK_DOT)) {
                    // Handle "..." (ellipsis) tokens - convert to integer 1
                    parser_expect(parser, TK_DOT);
                    parser_expect(parser, TK_DOT);
                    parser_expect(parser, TK_DOT);

                    MLIR_TypeHandle i32_ty = mlir_type_create_from_string(parser->ctx, str_lit("i32"));
                    append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                                create_integer_attr(parser, attr_name, 1, i32_ty));
                } else if (parser_peek(parser, TK_INTEGER)) {
                    // Handle regular integer values
                    string ival = parser_token_str(parser);
                    char *buffer = arena_new_array(parser->arena, char, ival.size + 1);
                    memcpy(buffer, ival.str, ival.size);
                    buffer[ival.size] = '\0';
                    int64_t int_val = atoll(buffer);
                    parser_expect(parser, TK_INTEGER);

                    string type_str = str_lit("");
                    if (parser_peek(parser, TK_COLON)) {
                        parser_expect(parser, TK_COLON);
                        if (parser_peek(parser, TK_NAME)) {
                            type_str = parser_token_str(parser);
                            parser_expect(parser, TK_NAME);
                        }
                    }
                    MLIR_TypeHandle attr_ty = mlir_type_create_from_string(
                        parser->ctx, type_str.size > 0 ? type_str : str_lit("i32"));
                    append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                                create_integer_attr(parser, attr_name, int_val, attr_ty));
                } else {
                    // Skip unknown tokens
                    parser_next_token(parser);
                }
            }
            if (parser_peek(parser, TK_COMMA)) {
                parser_expect(parser, TK_COMMA);
            }
        }
        parser_expect(parser, TK_RBRACE);
    }
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    // Handle optional type information
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // Skip type information for store operations
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                break;
            }
            parser_next_token(parser);
        }
    }

    MLIR_LocationHandle op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        MLIR_INVALID_HANDLE, 0,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        MLIR_INVALID_HANDLE, 0,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}

OperationParserResult parse_func_func_op(Parser *parser, const OperationParserParams *params) {
    // Collect all data before creating operation (inlined from parse_func_func)

    // Capture optional visibility and function name
    string visibility = str_lit("");
    string fname = str_lit("");
    MLIR_AttributeHandle *attrs = MLIR_INVALID_HANDLE;
    size_t n_attrs = 0;
    size_t cap_attrs = 0;

    while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF)) {
        if (parser_peek(parser, TK_NAME)) {
            string nm = parser_token_str(parser);
            if (str_eq(nm, str_lit("private")) || str_eq(nm, str_lit("public"))) {
                visibility = nm;
                parser_expect(parser, TK_NAME);
                continue;
            }
        }
        if (parser_peek(parser, TK_FUNCTION_NAME)) {
            fname = parser_token_str(parser);
            // Strip leading '@' (token text is "@name")
            if (fname.size > 0 && fname.str[0] == '@') {
                fname.str += 1;
                fname.size -= 1;
            }
            parser_expect(parser, TK_FUNCTION_NAME);
            continue;
        }
        parser_next_token(parser);
    }

    // Parse argument list (and collect arg types for the function_type)
    VecValue args;
    VecValue_reserve(parser->arena, &args, 8);
    MLIR_TypeHandle *arg_types = arena_new_array(parser->arena, MLIR_TypeHandle, 16);
    size_t n_arg_types = 0;
    size_t cap_arg_types = 16;
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            MLIR_TypeHandle this_arg_ty = MLIR_INVALID_HANDLE;
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                // Optional type annotation
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    string t = str_lit("");
                    if (parse_type_string(parser, &t)) {
                        this_arg_ty = mlir_type_create_from_string(parser->ctx, t);
                    }
                }
                MLIR_ValueHandle arg = MLIR_CreateValueBlockArg(parser->ctx, reg, 0, this_arg_ty, MLIR_INVALID_HANDLE);
                VecValue_push_back(parser->arena, &args, arg);
            } else if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
                // Type-only argument in declaration form
                string t = str_lit("");
                if (parse_type_string(parser, &t)) {
                    this_arg_ty = mlir_type_create_from_string(parser->ctx, t);
                } else {
                    parser_next_token(parser);
                }
            } else {
                parser_next_token(parser);
            }
            if (this_arg_ty != MLIR_INVALID_HANDLE) {
                if (n_arg_types == cap_arg_types) {
                    size_t new_cap = cap_arg_types * 2;
                    MLIR_TypeHandle *new_arr = arena_new_array(parser->arena, MLIR_TypeHandle, new_cap);
                    for (size_t i = 0; i < n_arg_types; i++) new_arr[i] = arg_types[i];
                    arg_types = new_arr;
                    cap_arg_types = new_cap;
                }
                arg_types[n_arg_types++] = this_arg_ty;
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    // Optional return type sequence after '->': either a single type or a
    // parenthesized comma-separated list of types.
    MLIR_TypeHandle *ret_types = arena_new_array(parser->arena, MLIR_TypeHandle, 4);
    size_t n_ret_types = 0;
    size_t cap_ret_types = 4;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        bool parens = parser_peek(parser, TK_LPAREN);
        if (parens) parser_expect(parser, TK_LPAREN);
        bool first_ret = true;
        while (!(parser_peek(parser, TK_LBRACE_END) || parser_peek(parser, TK_EOF))
                && !(parens && parser_peek(parser, TK_RPAREN))) {
            if (parser_peek(parser, TK_NAME) &&
                (str_eq(parser_token_str(parser), str_lit("loc")) ||
                 str_eq(parser_token_str(parser), str_lit("attributes")))) break;
            if (!first_ret && parser_peek(parser, TK_COMMA)) {
                parser_expect(parser, TK_COMMA);
            }
            first_ret = false;
            string t = str_lit("");
            if (parse_type_string(parser, &t)) {
                MLIR_TypeHandle th = mlir_type_create_from_string(parser->ctx, t);
                if (n_ret_types == cap_ret_types) {
                    size_t new_cap = cap_ret_types * 2;
                    MLIR_TypeHandle *new_arr = arena_new_array(parser->arena, MLIR_TypeHandle, new_cap);
                    for (size_t i = 0; i < n_ret_types; i++) new_arr[i] = ret_types[i];
                    ret_types = new_arr;
                    cap_ret_types = new_cap;
                }
                ret_types[n_ret_types++] = th;
            } else {
                // Unrecognized token; bail out to avoid an infinite loop.
                break;
            }
        }
        if (parens && parser_peek(parser, TK_RPAREN)) parser_expect(parser, TK_RPAREN);
    }

    // Optional attributes clause, which in real MLIR sits between the
    // signature and the body. func.func only looked for one after the body, so
    // any function carrying attributes failed to parse at all. Skipped rather
    // than captured, same as parse_tt_func_op above.
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("attributes"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LBRACE)) {
            int depth = 0; parser_expect(parser, TK_LBRACE); depth++;
            while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LBRACE)) { depth++; }
                else if (parser_peek(parser, TK_RBRACE)) { depth--; }
                parser_next_token(parser);
            }
        }
    }

    // Parse optional body region (definition) or leave as declaration
    MLIR_RegionHandle *regions = MLIR_INVALID_HANDLE;
    size_t n_regions = 0;
    if (parser_peek(parser, TK_LBRACE_END)) {
        // The body names the header's arguments, so they have to be in scope
        // before it is read rather than attached to the entry block after.
        // parse_tt_func_op above already does this; func.func did not, which
        // made every function with a named argument and a body fail to parse.
        symbol_table_push_scope(parser->arena, &parser->symbol_table);
        for (size_t i = 0; i < args.size; i++) {
            if (!args.data[i]) continue;
            string arg_name = MLIR_GetValueRegisterName(args.data[i]);
            if (arg_name.size > 0) {
                symbol_table_add_value(parser->arena, &parser->symbol_table,
                                       arg_name, args.data[i]);
            }
        }
        MLIR_RegionHandle region = parse_region(parser);
        symbol_table_pop_scope(&parser->symbol_table);
        regions = arena_new_array(parser->arena, MLIR_RegionHandle, 1);
        regions[0] = region;
        n_regions = 1;
        // Attach the function header's arguments to the entry block, unless
        // the body already declared its own (^bb0(...) form).
        if (region && MLIR_GetRegionNumBlocks(region) > 0 && args.size > 0) {
            MLIR_BlockHandle entry = MLIR_GetRegionBlock(region, 0);
            if (entry && MLIR_GetBlockNumArgs(entry) == 0) {
                for (size_t i = 0; i < args.size; i++) {
                    MLIR_AppendBlockArg(parser->ctx, entry, args.data[i]);
                }
            }
        }
    }

    // Store attributes for classic printing
    if (visibility.size > 0) append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("visibility"), visibility));
    if (fname.size > 0) append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("sym_name"), fname));
    {
        MLIR_TypeHandle func_type = MLIR_CreateTypeFunction(parser->ctx,
                                                              arg_types, n_arg_types,
                                                              ret_types, n_ret_types);
        append_attr(parser, &attrs, &n_attrs, &cap_attrs,
                    MLIR_CreateAttributeType(parser->ctx, str_lit("function_type"), func_type));
    }

    // Parse additional attributes, result types, and location
    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    MLIR_LocationHandle op_location = MLIR_INVALID_HANDLE;
    parse_generic_attrs_and_result_type(parser, &attrs, &n_attrs, &cap_attrs,
                                         &result_types, &n_result_types, params->op_type);

    // Fallback location if none was parsed
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // NOW create operation with all collected data
    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx, 
        
        params->op_type,
        str_lit(""),
        attrs, n_attrs,
        result_types, n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        MLIR_INVALID_HANDLE, 0,
        regions, n_regions,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}

OperationParserResult parse_affine_for_op(Parser *parser, const OperationParserParams *params) {
    MLIR_AttributeHandle *attributes = MLIR_INVALID_HANDLE;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    string iv_name = str_lit("");
    bool have_lb = false; int64_t lb_val = 0;
    bool have_ub = false; int64_t ub_val = 0;
    bool have_step = false; int64_t step_val = 1;

    if (parser_peek(parser, TK_REGISTER)) {
        iv_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        // '=' lower bound (literal int only; otherwise consume tokens)
        if (parser_peek(parser, TK_EQUAL)) {
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_INTEGER)) {
                string s = parser_token_str(parser);
                int64_t v = 0;
                for (size_t i = 0; i < s.size; i++) if (s.str[i]>='0' && s.str[i]<='9') v = v*10 + (s.str[i]-'0');
                lb_val = v; have_lb = true;
                parser_expect(parser, TK_INTEGER);
            }
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) break;
                parser_next_token(parser);
            }
        }

        // 'to' upper bound
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_INTEGER)) {
                string s = parser_token_str(parser);
                int64_t v = 0;
                for (size_t i = 0; i < s.size; i++) if (s.str[i]>='0' && s.str[i]<='9') v = v*10 + (s.str[i]-'0');
                ub_val = v; have_ub = true;
                parser_expect(parser, TK_INTEGER);
            }
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) break;
                parser_next_token(parser);
            }
        }

        // optional 'step'
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_INTEGER)) {
                string s = parser_token_str(parser);
                int64_t v = 0;
                for (size_t i = 0; i < s.size; i++) if (s.str[i]>='0' && s.str[i]<='9') v = v*10 + (s.str[i]-'0');
                step_val = v; have_step = true;
                parser_expect(parser, TK_INTEGER);
            }
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                parser_next_token(parser);
            }
        }
    }

    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    MLIR_BlockHandle block = MLIR_CreateBlock(parser->ctx);
    if (iv_name.size > 0) {
        MLIR_ValueHandle iv_block_arg = MLIR_CreateValueBlockArg(parser->ctx, iv_name, (uint32_t)MLIR_GetBlockNumArgs(block), mlir_type_create_from_string(parser->ctx, str_lit("index")), MLIR_INVALID_HANDLE);
        MLIR_AppendBlockArg(parser->ctx, block, iv_block_arg);
        symbol_table_add_value(parser->arena, &parser->symbol_table, iv_name, iv_block_arg);
    }

    bool prev_flag = parser->capture_trailing_comments;
    parser->capture_trailing_comments = true;
    while (!parser_peek(parser, TK_RBRACE)) {
        MLIR_OpHandle inner = parse_operation(parser);
        MLIR_AppendBlockOp(parser->ctx, block, inner);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    parser->capture_trailing_comments = prev_flag;

    MLIR_LocationHandle op_location = MLIR_INVALID_HANDLE;
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op_location = parse_loc(parser);
    }

    symbol_table_pop_scope(&parser->symbol_table);

    MLIR_RegionHandle region = MLIR_CreateRegion(parser->ctx);
    MLIR_AppendRegionBlock(parser->ctx, region, block);
    MLIR_RegionHandle *regions = arena_new_array(parser->arena, MLIR_RegionHandle, 1);
    regions[0] = region;

    MLIR_TypeHandle *result_types = MLIR_INVALID_HANDLE;
    size_t n_result_types = 0;
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, MLIR_INVALID_HANDLE);
    MLIR_LocationHandle loc_after_attrs = parse_optional_location(parser);
    if (loc_after_attrs) op_location = loc_after_attrs;
    if (!op_location) op_location = params->unnumbered_loc_def;

    // Stash bounds on the op as private attrs so the classic printer can emit them.
    MLIR_TypeHandle idx_ty = mlir_type_create_from_string(parser->ctx, str_lit("index"));
    if (have_lb) append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
        create_integer_attr(parser, str_lit("_lb"), lb_val, idx_ty));
    if (have_ub) append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
        create_integer_attr(parser, str_lit("_ub"), ub_val, idx_ty));
    if (have_step) append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
        create_integer_attr(parser, str_lit("_step"), step_val, idx_ty));

    MLIR_OpHandle op = MLIR_CreateOp(parser->ctx,
        params->op_type,
        str_lit(""),
        attributes, n_attributes,
        result_types, n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        MLIR_INVALID_HANDLE, 0,
        regions, 1,
        op_location,
        params->unnumbered_loc_def,
        params->trailing_comment,
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = MLIR_INVALID_HANDLE,
        .n_results = 0,
    };
    return out;
}
