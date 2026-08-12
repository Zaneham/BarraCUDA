#pragma once

#include <base/arena.h>
#include <base/string.h>
#include <base/hashtable.h>
#include <base/vector.h>

#include "tokenizer.h"
#include "mlir_api.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct OperationParserParams {
    MLIR_Context *ctx;
    Arena  *arena;
    MLIR_OpType  op_type;
    string  opname; /* only non-empty for unregistered ops */

    MLIR_ValueHandle *lhs_results;
    size_t     n_lhs_results;

    MLIR_LocationHandle unnumbered_loc_def;
    int64_t       source_line_start;
    string        trailing_comment;
} OperationParserParams;

typedef struct OperationParserResult {
    MLIR_OpHandle operation;
    MLIR_ValueHandle  *results;
    size_t         n_results;
} OperationParserResult;

// Hash function for strings
static inline size_t string_hash(string str) {
    size_t hash = 5381;
    for (size_t i = 0; i < str.size; i++) {
        hash = ((hash << 5) + hash) + (size_t)(unsigned char)str.str[i];
    }
    return hash;
}

// Equality function for strings
static inline bool string_equal(string a, string b) {
    return str_eq(a, b);
}

// Define hashtable for string -> MLIR_ValueHandle mapping
#define SymbolTable_HASH string_hash
#define SymbolTable_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, MLIR_ValueHandle, SymbolTable)

// Scoped symbol table for SSA values
typedef struct ScopedSymbolTable {
    SymbolTable *scopes;
    size_t num_scopes;
    size_t scope_capacity;
} ScopedSymbolTable;

// Location map for named location references
#define LocationMap_HASH string_hash
#define LocationMap_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, MLIR_LocationHandle, LocationMap)

// Per-region map from block label (e.g. "^bb1") to MLIR_BlockHandle
#define BlockLabelMap_HASH string_hash
#define BlockLabelMap_EQUAL string_equal
DEFINE_HASHTABLE_FOR_TYPES(string, MLIR_BlockHandle, BlockLabelMap)


typedef struct {
    MLIR_Context *ctx;
    Arena *arena;
    unsigned char *input;
    TokenType sym;
    uint64_t cur;
    uint64_t first, last;
    ScopedSymbolTable symbol_table;
    LocationMap location_map;  // For #locN -> Location mapping
    int next_loc_id;          // Counter for generating #locN IDs
    MLIR_LocationHandle unnumbered_loc_def; // Optional: definition of unnumbered '#loc' at file start
    // Parsing mode flag to enable robust trailing comment capture in special contexts
    bool capture_trailing_comments;
    // Stack of per-region block-by-label maps for forward-reference handling.
    BlockLabelMap *block_label_stack;
    size_t block_label_depth;
    size_t block_label_capacity;
} Parser;



// Symbol table functions
void symbol_table_init(Arena *arena, ScopedSymbolTable *st);
void symbol_table_push_scope(Arena *arena, ScopedSymbolTable *st);
void symbol_table_pop_scope(ScopedSymbolTable *st);
void symbol_table_add_value(Arena *arena, ScopedSymbolTable *st, string name, MLIR_ValueHandle value);
MLIR_ValueHandle symbol_table_lookup(ScopedSymbolTable *st, string name);

// Block-by-label helpers (used by branch parsers + parse_block).
void parser_push_region_blocks(Parser *parser);
void parser_pop_region_blocks(Parser *parser);
MLIR_BlockHandle parser_get_or_create_block(Parser *parser, string label);

// Forward declarations for core IR nodes (opaque here)
// Specialized parsing functions
void parse_gpu_launch(Parser *parser, MLIR_OpHandle op);

string tokentype_to_string(TokenType tt);
void mlir_parser_init(MLIR_Context *ctx, Parser *parser, string text);
void parser_next_token(Parser *parser);
bool parser_peek(Parser *parser, TokenType s);
void parser_expect(Parser *parser, TokenType s);
string parser_token_str(Parser *parser);
/* Called after parser_error has printed. Upstream this was a bare exit(1),
 * which is right for a standalone parser and wrong for anything embedding
 * one, so the choice belongs to whoever links the reader. src/mlir/mlir_fe.c
 * is Booth's, and unwinds back to ml_parse instead of taking the process
 * down. It must not return. */
_Noreturn void mlir_parse_fail(void);

void parser_error(Parser *parser, string msg, uint64_t first, uint64_t last);
void parser_warning(Parser *parser, string msg, uint64_t first, uint64_t last);
MLIR_OpHandle parse_module(Parser *parser);
MLIR_LocationHandle parse_loc(Parser *parser);
MLIR_OpHandle parse_operation(Parser *parser);
MLIR_RegionHandle parse_region(Parser *parser);
MLIR_BlockHandle parse_block(Parser *parser);

bool parse_type_string(Parser *parser, string *out);
void consume_optional_hash_selector(Parser *parser);


const char *string_data_or_null(string s);

bool parse_register_operand(Parser *parser, VecValue *operands, bool allow_hash_selector);
MLIR_ValueHandle *finalize_results(const OperationParserParams *params,
                                    MLIR_OpHandle op,
                                    MLIR_TypeHandle *result_types,
                                    size_t n_result_types,
                                    size_t *out_n_results);
MLIR_AttributeHandle create_string_attr(Parser *parser, string name, string value);
MLIR_AttributeHandle create_integer_attr(Parser *parser, string name, int64_t value, MLIR_TypeHandle type);
MLIR_AttributeHandle create_float_attr(Parser *parser, string name, double value, MLIR_TypeHandle type);
MLIR_AttributeHandle create_bool_attr(Parser *parser, string name, bool value);
void operation_append_attribute(Parser *parser, MLIR_OpHandle op, MLIR_AttributeHandle attr);
MLIR_ValueHandle lookup_or_create_value(Parser *parser, string reg, string default_type);
void append_attr(Parser *parser, MLIR_AttributeHandle **attrs, size_t *n, size_t *cap, MLIR_AttributeHandle attr);
void attr_list_init_from_op(Parser *parser, MLIR_OpHandle op, MLIR_AttributeHandle **attrs, size_t *n, size_t *cap);
void parse_angle_brace_attributes(Parser *parser, MLIR_AttributeHandle **attributes, size_t *n_attributes, size_t *attributes_capacity);
void parse_brace_attributes(Parser *parser, MLIR_AttributeHandle **attributes, size_t *n_attributes, size_t *attributes_capacity);
void parse_result_types(Parser *parser, MLIR_TypeHandle **result_types, size_t *n_result_types,
                              MLIR_AttributeHandle **attributes, size_t *n_attributes, size_t *attributes_capacity,
                              MLIR_OpType op_type, MLIR_OpHandle op_for_attributes);
MLIR_LocationHandle parse_optional_location(Parser *parser);
void parse_generic_attrs_and_result_type(Parser *parser,
                                          MLIR_AttributeHandle **attributes,
                                          size_t *n_attributes,
                                          size_t *attributes_capacity,
                                          MLIR_TypeHandle **result_types,
                                          size_t *n_result_types,
                                          MLIR_OpType op_type);

void get_newlines(Arena *arena, const string s, vector_i64 *newlines);
void linear_to_line_column(const vector_i64 newlines, uint64_t first,
        int64_t *start_of_line, int64_t *end_of_line,
        int64_t *line_first, int64_t *column_first);

MLIR_OpType op_string_to_type(string name);

// TODO: use Type by value
MLIR_TypeHandle parse_type_from_string(MLIR_Context *ctx, string type_str);
string type_to_string(MLIR_Context *ctx, MLIR_TypeHandle type);

// Public parser facade
MLIR_OpHandle mlir_parse_module(MLIR_Context *ctx, const char *input, size_t input_len, MLIR_LocationMap **out_location_map);
const char *mlir_tokentype_to_string(int token_type);

MLIR_TypeHandle mlir_type_create_from_string(MLIR_Context *ctx, string type_str);

#ifdef __cplusplus
}
#endif
