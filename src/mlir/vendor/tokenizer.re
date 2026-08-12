// Compile with:
// re2c --no-generation-date -b tokenizer.re -o tokenizer.c

#include <stdbool.h>

#include "tokenizer.h"


#define RET(x) \
    *token_type=x; \
    *position = (uint64_t)(cur-string); \
    return;

void tokenizer_get_next_token(const unsigned char *string,
        uint64_t *position, TokenType *token_type)
{
    const unsigned char *cur = string + *position;
    while (true) {
        /*
        Re2c has excellent documentation at:

        https://re2c.org/manual/manual_c.html

        The first paragraph there explains the basics:

        * If multiple rules match, the longest match takes precedence
        * If multiple rules match the same string, the earlier rule takes
          precedence
        * Default rule `*` should always be defined, it has the lowest priority
          regardless of its place and matches any code unit
        * We use the "Sentinel character" method for end of input:
            * The end of the input text is denoted with a null character \x00
            * Thus the null character cannot be part of the input otherwise
            * There is one rule to match \x00 to end the parser
            * No other rule is allowed to match \x00, otherwise the re2c block
              would parse past the end of the string and segfaults
            * A special case of the previous point are negated character
              ranges, such as [^"\x00], where one must include \x00 in it to
              ensure this rule does not match \x00 (all other rules simply do
              not mention \x00)
            * See the "Handling the end of input" section in the re2c
              documentation for more info

        The re2c block interacts with the rest of the code via just one pointer
        variable `cur`. On entering the re2c block, the `cur` variable must
        point to the first character of the token to be tokenized by the block.
        The re2c block below then executes on its own until a rule is matched:
        the action in {} is then executed. In that action `cur` points to the
        first character of the next token.

        Before the re2c block we save the current `cur` into `tok`, so that we
        can use `tok` and `cur` in the action in {} to extract the token that
        corresponds to the rule that got matched:

        * `tok` points to the first character of the token
        * `cur-1` points to the last character of the token
        * `cur` points to the first character of the next token
        * `cur-tok` is the length of the token

        In the action, we do one of:

        * call `continue` which executes another cycle in the for loop (which
          will parse the next token); we use this to skip a token
        * call `return` which returns from this function; we return a token

        In both cases, `cur` points to first character of the next
        token, which becomes `tok` at the next iteration of the loop (either
        right away after `continue` or after the `lex` function is called again
        after `return`).

        See the manual for more details.
        */


        // These two variables are needed by the re2c block below internally,
        // initialization is not needed. One can think of them as local
        // variables of the re2c block.
        const unsigned char *mar, *ctxmar;
        /*!re2c
            re2c:define:YYCURSOR = cur;
            re2c:define:YYMARKER = mar;
            re2c:define:YYCTXMARKER = ctxmar;
            re2c:yyfill:enable = 0;
            re2c:define:YYCTYPE = "unsigned char";

            end = "\x00";
            whitespace = [ \t\v]+;
            newline = "\n" | "\r\n";
            digit = [0-9];
            char =  [a-zA-Z_-];
            name = char (char | digit)*;
            significand = (digit+"."digit*) | ("."digit+);
            exp = [edED][-+]? digit+;
            integer = digit+;
            real = (significand exp?) | (digit+ exp);
            string = '"' [^"\x00]* '"';
            comment = "//" [^\n\x00]*;
            register = "%" (name | integer);
            fn_name = "@" name;
            type = integer ("x" (integer | "?"))* "x";

            * { RET(TK_ERROR) }
            end { RET(TK_EOF) }
            whitespace { RET(TK_WHITESPACE) }
            newline { RET(TK_NEWLINE) }

            // Keywords
            'abstract' { RET(KW_ABSTRACT) }
            'all' { RET(KW_ALL) }
            'write' { RET(KW_WRITE) }

            // Single character symbols
            "(" { RET(TK_LPAREN) }
            "(" / "{" { RET(TK_LPAREN_BRACE) }
            ")" { RET(TK_RPAREN) }
            "[" { RET(TK_LBRACKET) }
            "]" { RET(TK_RBRACKET) }
            "{" { RET(TK_LBRACE) }
            "{" / newline { RET(TK_LBRACE_END) }
            "}" { RET(TK_RBRACE) }
            "<" { RET(TK_LANGLE) }
            ">" { RET(TK_RANGLE) }
            "!" { RET(TK_EXCLAMATION) }
            "$" { RET(TK_DOLLAR) }
            "+" { RET(TK_PLUS) }
            "-" { RET(TK_MINUS) }
            "=" { RET(TK_EQUAL) }
            ":" { RET(TK_COLON) }
            ";" { RET(TK_SEMICOLON) }
            "/" { RET(TK_SLASH) }
            "%" { RET(TK_PERCENT) }
            "," { RET(TK_COMMA) }
            "*" { RET(TK_STAR) }
            "|" { RET(TK_VBAR) }
            "@" { RET(TK_AT) }
            "." { RET(TK_DOT) }

            // Multiple character symbols
            "//" [^\n\x00]* { RET(TK_COMMENT) }
            "->" { RET(TK_ARROW) }

            "#" name { RET(TK_HASH_NAME); }
            // This is used for vector values, e.g.: %49#0
            "#" integer { RET(TK_HASH_NAME); }
            "^" name { RET(TK_CARET_NAME); }

            name { RET(TK_NAME) }
            name "." name { RET(TK_NAME_DOT_NAME) }
            register { RET(TK_REGISTER) }
            fn_name { RET(TK_FUNCTION_NAME) }
            integer { RET(TK_INTEGER) }
            type { RET(TK_TYPE_DIM) }
            real { RET(TK_REAL) }
            string { RET(TK_STRING) }
        */
    }
}
