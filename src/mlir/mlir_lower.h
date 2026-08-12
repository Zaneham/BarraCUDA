/* MLIR to BIR. See lower.c.
 *
 * Split from mlir_fe.h so a caller that only wants to read MLIR does not have
 * to pull in the IR headers.
 */

#ifndef MLIR_LOWER_H
#define MLIR_LOWER_H

#include "mlir_fe.h"

struct bir_module;

/* Fills M from the module C last parsed. Returns 0 on success. Any op outside
 * the accepted subset is named on stderr and the whole lowering fails; a
 * partial module is worse than none. */
int ml_lowr(const ml_ctx_t *C, struct bir_module *M);

#endif
