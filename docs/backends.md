# Backend Authoring

Hello human or LLM asked to clone this. If you are here because you want to add a new backend to Booth, or promote a stub to first-class, you are in the right place. The whole shape is in `src/backend/backend.h` plus the reference skeleton in `src/backend/skeleton/`. This document walks the pipeline, the contract, and the recipe.

## The pipeline

BIR comes in, machine bytes go out. Between those two ends, Booth's backend contract splits the work into named phases:

```
  BIR module -> isel -> [sched] -> [regalloc] -> [verify] -> emit -> file(s)
                             (optional phases in brackets)
```

Each phase is a function pointer on the backend's `be_desc_t` descriptor. Optional phases with a NULL pointer are skipped by the driver. Only `isel` and `emit` are strictly required.

- **isel**: consume the BIR module, produce a backend-specific machine module (`void *`). This is where all your target-specific instruction selection lives.
- **sched**: reorder instructions for latency, throughput, whatever your target cares about. Skip if you have nothing to do here.
- **regalloc**: assign physical registers. Skip if your emitter goes straight from BIR values to instructions (the CPU backend does this).
- **verify**: sanity check the machine module at each defined phase (`BE_VFY_ISEL`, `BE_VFY_RA`). Skip if you have no invariants to check. AMD's `bc_vfy` is a good reference.
- **emit**: write the final output(s) to disk. Required. Multi-file backends (like Tensix Metalium) do everything inside their `emit` and set the `BE_F_MULTIOUT` feature flag.
- **mfree**: free the backend module that `isel` allocated.

## The contract

`src/backend/backend.h` defines everything. The two important types:

```c
typedef struct be_desc {
    const char *name;              /* stable id */
    const char *triple;            /* target triple, or NULL */
    uint32_t    feats;             /* BE_F_* bitmask */

    int  (*is_on)  (const struct be_cfg *cfg);
    int  (*isel)   (const struct bir_module *M, const struct be_cfg *cfg,
                    void **out_mmod);
    int  (*sched)  (void *mmod);
    int  (*regalc) (void *mmod);
    int  (*verify) (const void *mmod, int phase);
    int  (*emit)   (const void *mmod, const struct be_cfg *cfg,
                    const char *out_path);
    void (*mfree)  (void *mmod);
} be_desc_t;
```

Return codes come from `be_ret_t`: `BE_OK` on success, negative on failure (`BE_EISEL`, `BE_ERA`, `BE_EEMIT`, etc.), `BE_UNSUP` when the op deliberately does not apply.

`is_on(cfg)` returns non-zero when the backend should run for this invocation. Typically this reads a `mode_*` field from `be_cfg_t`. See `src/backend/backend_cfg.h` for the config layout.

`cfg` is opaque in the header and typed in the backend implementation. Backends cast to `be_cfg_t` and read what they need. If your backend needs a new knob, add a field to `be_cfg_t`, populate it from the CLI parser in `src/main.c`, read it from your descriptor.

## Feature flags

Declare only what you honestly implement. The conformance suite (coming in Phase B) will run the tests matching your declared features and fail loudly if you oversell.

```c
BE_F_SIMT      /* SIMT execution model */
BE_F_SCALAR    /* scalar / stack-everything */
BE_F_ATOMIC    /* atomic RMW family */
BE_F_SHARED    /* per-block shared memory */
BE_F_WARP      /* shfl, ballot, vote */
BE_F_MFMA      /* matrix multiply-accumulate */
BE_F_BARRIER   /* __syncthreads equivalent */
BE_F_DIV       /* per-lane divergence handling */
BE_F_SCRATCH   /* per-thread private */
BE_F_TRANSC    /* sin, cos, exp2, log2 */
BE_F_F16       /* 16-bit float */
BE_F_F64       /* 64-bit float */
BE_F_BF16      /* brain float */
BE_F_MULTIOUT  /* emits multiple files (Tensix) */
```

## The recipe: adding a new backend

Say you are adding support for a new accelerator called `mygpu`.

**1. Copy the skeleton.**

```bash
cp -r src/backend/skeleton src/backend/mygpu
```

**2. Rename inside.** The skeleton uses the prefix `skel_`; rename to `mygpu_` throughout.

**3. Register the descriptor.** In `src/backend/backends.c`:

```c
extern const be_desc_t be_mygpu;    /* add near the other externs */

const be_desc_t * const be_list[] = {
    &be_amd,
    ...
    &be_mygpu,                       /* add to the list */
    NULL
};
```

**4. Wire up the build.** In the top-level `Makefile`, add `src/backend/mygpu/mygpu.c` to `SOURCES`. Also add `-Isrc/backend/mygpu` to `CFLAGS` and `TCFLAGS` if you split into multiple files.

**5. Add a mode flag.** In `src/backend/backend_cfg.h`, add `int mode_mygpu;` to `be_cfg_t`. In `src/main.c`'s CLI parser, add `--mygpu` handling that sets `mode_mygpu = 1`.

**6. Implement `isel` first.** This is where you consume the BIR module and produce your machine module. Look at `src/nvidia/nv_be.c` for the simplest full example, or `src/amdgpu/amd_be.c` for the full-pipeline shape.

**7. Implement `emit`.** Write the output file(s) to disk. Return `BE_OK` or `BE_EEMIT`.

**8. Declare features you have implemented** in the descriptor. Start with zero, add flags as ops start working.

**9. Run the conformance suite.** (Coming in Phase B.)

```bash
make backend-conformance BACKEND=mygpu
```

The suite runs the tests matching your declared features. Green means you are done for that feature set.

## Reference backends by shape

- **Simplest full example**: `src/nvidia/nv_be.c`. Isel + emit only. About 50 lines.
- **Full pipeline including sched, regalloc, and verify**: `src/amdgpu/amd_be.c`. 90 lines of wrappers over the existing AMDGPU internals.
- **Multi-output**: `src/tensix/tensix_be.c`. Emit writes compute + reader + writer + host + several binary artefacts, all derived from one output-path stem. Sets `BE_F_MULTIOUT` in the descriptor.
- **Scalar / stack-everything**: `src/cpu/cpu_be.c`. No sched or regalloc, isel and emit only, `BE_F_SCALAR` in the flags.
- **Stub-level starting point**: `src/intel/intel_be.c`. Same shape as the others but the underlying `intel_compile` returns "not yet a working compiler." Useful as an example of a backend under construction.

## Contract invariants

- Every registered backend must have `name`, `is_on`, `isel`, `emit`. `be_list_ok` in `tests/tbackend.c` asserts this.
- `is_on(cfg)` must be cheap and free of side effects. It is called on every dispatch.
- `isel` allocates the machine module. `mfree` frees it. If your isel does not allocate, `mfree` can be NULL.
- If your `verify` is set, expect it to be called at `BE_VFY_ISEL` and `BE_VFY_RA`. Handle any phase you do not care about by returning `BE_OK`.
- Return `BE_UNSUP` from an op that is intentionally not implemented. The driver treats this as a real error (you registered the backend, you promised to run).

## JPL discipline

Booth's project rules apply to backends too:

- No dynamic allocation in hot paths. Bounded pools where you can.
- No recursion in codegen.
- Bounded loops; if you iterate to a fixpoint, have a guard counter.
- Bounds-check array accesses from external or untrusted indices.
- Check return values. Handle the failure path.
- No floats where integers will do.

See `CONTRIBUTING.md` for the rest.

## Where to go from here

- File a `good-first-backend` issue if you are stuck partway.
- Read `docs/mainframe.md` for the ABEND / SNAP / SYSPRINT diagnostic conventions that every backend inherits.
- The conformance suite (Phase B) and shared substrate (Phase C, `src/be_common/`) are on the roadmap.

Adding a new backend or promoting a stub to first-class should be a weekend project. If it turns into a career, something is wrong and I want to hear about it.
