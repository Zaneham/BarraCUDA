#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LIST_OF_TOKENS \
    X(TK_EOF) \
    X(TK_NEWLINE) \
    X(TK_WHITESPACE) \
    X(TK_ERROR) \
    X(TK_NAME) \
    X(TK_NAME_DOT_NAME) \
    X(TK_HASH_NAME) \
    X(TK_CARET_NAME) \
    X(TK_COMMENT) \
    X(TK_LPAREN) \
    X(TK_RPAREN) \
    X(TK_LBRACKET) \
    X(TK_RBRACKET) \
    X(TK_LBRACE) \
    X(TK_LBRACE_END) \
    X(TK_RBRACE) \
    X(TK_LPAREN_BRACE) \
    X(TK_LANGLE) \
    X(TK_RANGLE) \
    X(TK_EXCLAMATION) \
    X(TK_DOLLAR) \
    X(TK_PLUS) \
    X(TK_MINUS) \
    X(TK_STAR) \
    X(TK_SLASH) \
    X(TK_EQUAL) \
    X(TK_COLON) \
    X(TK_SEMICOLON) \
    X(TK_PERCENT) \
    X(TK_COMMA) \
    X(TK_VBAR) \
    X(TK_AT) \
    X(TK_DOT) \
    X(TK_ARROW) \
    X(TK_INTEGER) \
    X(TK_REAL) \
    X(TK_STRING) \
    X(TK_REGISTER) \
    X(TK_TYPE_DIM) \
    X(TK_FUNCTION_NAME) \
    X(KW_ABSTRACT) \
    X(KW_ALL) \
    X(KW_WRITE)

typedef enum {
#define X(token) token,
    LIST_OF_TOKENS
#undef X
} TokenType;

// Get the token in the `string` at the `position`. The function
// returns the `token_type` and the `position` one character after the token.
void tokenizer_get_next_token(const unsigned char *string,
        uint64_t *position, TokenType *token_type);

#ifdef __cplusplus
}
#endif
