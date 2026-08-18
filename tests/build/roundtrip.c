/* roundtrip.c -- parse BIR text, print it again, expect the same bytes.
 *
 *   roundtrip kernel.bir            print the reparsed module
 *   roundtrip kernel.bir out.o      and compile it for the CPU backend
 */

#include <stdio.h>
#include "bir_parse.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: roundtrip kernel.bir [out.o]\n");
        return 2;
    }

    bb_t *B = bir_parse_file(argv[1]);
    if (B == NULL) return 1;

    if (bb_full(B)) {
        fprintf(stderr, "roundtrip: a pool overflowed\n");
        bb_free(B);
        return 1;
    }

    int rc = 0;
    if (argc > 2) rc = bb_emit(B, "cpu", argv[2]);
    else          rc = bb_print(B, NULL);

    bb_free(B);
    return rc == 0 ? 0 : 1;
}
