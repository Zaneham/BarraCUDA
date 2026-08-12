// Agnostic in-tree port of upstream's
// mlir/lib/Transforms/Utils/CFGToSCF.cpp (Bahmann/Reissmann 2015,
// "Perfect Reconstructability of Control Flow from Demand Dependence
// Graphs"). Drives the lift entirely through mlir_api.h so the same
// implementation runs against both backends.
//
// Status: complete. Mirrors the wasm-flavored variant
// (CFGToSCFForWasm in mlir_api_impl_upstream.cpp): i32 switch
// discriminators, ub.poison/llvm.mlir.undef for undef values,
// llvm.return when an unreachable terminator is needed inside an
// llvm.func. Exercised end-to-end by `test_tinyc_native_wasm` on
// Linux/macOS/Windows in CI; that suite must stay green.
//
// Known limitations:
//   - Only `func.func` and `llvm.func` are recognised as carriers of
//     a CFG body. Other dialect function ops (gpu.func, spirv.func,
//     etc.) are silently skipped.
//   - Only walks the module's top-level region. Nested module-like
//     ops (e.g. builtin.module nested inside another op's region) are
//     not visited.
//   - When every arm of a multi-successor terminator returns with a
//     *different* return-like op kind (e.g. one arm `func.return`,
//     another `llvm.return`) the algorithm provably cannot lift it
//     and leaves the cf.* terminator in place. Downstream
//     (wasmssa-lower) rejects this and reports a hard error.
//   - The M9 driver caps the per-region worklist at 4096 iterations
//     as a runaway guard; bodies that would need more are rare in
//     practice. The driver emits a diagnostic to stderr when the cap
//     fires and leaves any unlifted cf.* ops behind for downstream
//     stages to flag.
//   - No predecessor cache: MLIR_GetBlockNumPredecessors /
//     MLIR_GetBlockPredecessor are O(R) per call. Cycle-edge and
//     reduce-loop analyses query them repeatedly, so the algorithm
//     is O(R^2) in the number of blocks per region. Fine in practice
//     (functions stay small) but a single shared region-scoped cache
//     would make this O(R).

#ifndef MLIR_LIFT_CF_TO_SCF_H
#define MLIR_LIFT_CF_TO_SCF_H

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lift cf.* terminators in every func.func and llvm.func body of the
// `module` op into scf.* structured control flow. Mirrors the upstream
// MLIR_LiftCfToScf contract: returns true on success, false on failure
// or on input the algorithm cannot handle. On success there are no
// remaining cf.br / cf.cond_br / cf.switch ops other than those left
// behind for multi-kind return-like dispatches (which the algorithm
// proves cannot be lifted further).
bool MLIR_LiftCfToScfNative(MLIR_Context *ctx, MLIR_OpHandle module);

#ifdef __cplusplus
}
#endif

#endif // MLIR_LIFT_CF_TO_SCF_H
