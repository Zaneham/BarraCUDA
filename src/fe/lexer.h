#ifndef BARRACUDA_LEXER_H
#define BARRACUDA_LEXER_H

#include "token.h"

typedef struct {
    const char *src;
    uint32_t    src_len;
    uint32_t    pos;
    uint32_t    line;
    uint32_t    line_start;

    token_t    *tokens;
    uint32_t    num_tokens;
    uint32_t    max_tokens;

    bc_error_t  errors[BC_MAX_ERRORS];
    int         num_errors;
} lexer_t;

void lexer_init(lexer_t *L, const char *src, uint32_t len,
                token_t *tokens, uint32_t max_tokens);

int  lexer_tokenize(lexer_t *L);

int  lexer_token_text(const lexer_t *L, const token_t *tok,
                      char *buf, int bufsize);

/* lookup_keyword binary searches the keyword table, so keeping that table in
 * order is VERY IMPORTANT. Put an entry in the wrong place and the keyword
 * quietly lexes as an identifier, then blows up as a parse error somewhere
 * else entirely. Handed out so the suite can check the order instead of
 * trusting the joke above the table. */
int         lexer_kw_count(void);
const char *lexer_kw_at(int i);

#endif /* BARRACUDA_LEXER_H */
