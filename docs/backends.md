# Backend Authoring

Want to add a new backend or promote one of the stubs to first-class?
Right place. Everything you need is in `src/backend/backend.h` plus
the skeleton in `src/backend/skeleton/`. Copy skeleton, register in
`backends.c`, fill in ops, done. Should be a weekend, not a career.

## Pipeline

```
BIR -> isel -> [sched] -> [regalloc] -> [verify] -> emit -> file(s)
```

Bracketed phases are optional; a NULL op pointer means "driver
skips." Only `isel` and `emit` are required.

## Contract

```c
typedef struct be_desc {
    const char *name;
    const char *triple;
    uint32_t    feats;

    int  (*is_on)  (const struct be_cfg *);
    int  (*isel)   (const struct bir_module *, const struct be_cfg *,
                    void **out_mmod);
    int  (*sched)  (void *);
    int  (*regalc) (void *);
    int  (*verify) (const void *, int phase);
    int  (*emit)   (const void *, const struct be_cfg *, const char *path);
    void (*mfree)  (void *);
} be_desc_t;
```

`is_on(cfg)` is cheap; typically returns a `mode_*` flag from
`be_cfg_t`. Return codes are `be_ret_t` (`BE_OK`, negative on
failure, `BE_UNSUP` when the op deliberately does nothing).

## Features

Declare only what you actually implement. A backend that claims
`BE_F_MFMA` and doesn't emit `mfma` instructions will fail the
conformance suite once that ships.

```
BE_F_SIMT       BE_F_SCALAR    BE_F_ATOMIC   BE_F_SHARED
BE_F_WARP       BE_F_MFMA      BE_F_BARRIER  BE_F_DIV
BE_F_SCRATCH    BE_F_TRANSC    BE_F_F16      BE_F_F64
BE_F_BF16       BE_F_MULTIOUT
```

## Recipe

1. `cp -r src/backend/skeleton src/backend/mygpu`
2. Rename `skel_` -> `mygpu_` in the copy.
3. Add the extern + list entry in `src/backend/backends.c`.
4. Add `src/backend/mygpu/mygpu.c` to `SOURCES` in the Makefile.
5. Add `mode_mygpu` to `be_cfg_t` (in `backend_cfg.h`) and wire the
   CLI flag in `src/main.c`.
6. Implement `isel` first, then `emit`. Add flags to the descriptor
   as ops start working.

## References by shape

- `src/nvidia/nv_be.c` -- simplest full example (isel + emit only).
- `src/amdgpu/amd_be.c` -- full pipeline including sched, regalloc,
  verify.
- `src/tensix/tensix_be.c` -- multi-output (emits compute + reader +
  writer + host + binaries derived from one path stem).
- `src/cpu/cpu_be.c` -- scalar / stack-everything shape.
- `src/intel/intel_be.c` -- stub-level shape for a backend under
  construction.

## Invariants

- Every registered backend has `name`, `is_on`, `isel`, `emit`.
- `is_on` is side-effect free.
- `isel` allocates the module, `mfree` frees it (or `mfree` is NULL
  when nothing was allocated).
- Return `BE_UNSUP` from an op that is deliberately not implemented;
  the driver treats it as a real error, not silent success.

Coding rules in `CONTRIBUTING.md` apply as usual.
