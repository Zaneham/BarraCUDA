#pragma once

#include <base/string.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

string op_type_to_string(MLIR_OpType type);

#ifdef __cplusplus
}
#endif
