#pragma once

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Classic MLIR printer that produces upstream MLIR textual format.
 */

string print_operation_classic(MLIR_Context *ctx, int indent_level, MLIR_OpHandle op);
string print_region_classic(MLIR_Context *ctx, int indent_level, MLIR_RegionHandle region);
string print_block_classic(MLIR_Context *ctx, int bb_index, int indent_level, MLIR_BlockHandle block);
string print_module_classic(MLIR_Context *ctx, MLIR_OpHandle module, MLIR_LocationMap *location_map);

#ifdef __cplusplus
}
#endif
