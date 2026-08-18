/* bir_parse.h -- read BIR text back in. The other half of bir_print.c. */
#ifndef BIR_PARSE_H
#define BIR_PARSE_H

#include "booth_build.h"

/* NULL on error, after a message to stderr. Caller frees with bb_free. */
bb_t *bir_parse(const char *text, const char *name);


bb_t *bir_parse_file(const char *path);

#endif /* BIR_PARSE_H */
