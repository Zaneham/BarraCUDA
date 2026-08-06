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
    uint32_t    opts_size;

    const char *const *flags;
    int  (*parse)  (const char *arg, const char *next, void *opts);

    int  (*is_on)  (const void *opts);
    uint32_t (*warp_size)(const void *opts);
    int  (*isel)   (const struct bir_module *, const struct be_cfg *,
                    const void *opts, void **out_mmod);
    int  (*sched)  (void *);
    int  (*regalc) (void *);
    int  (*verify) (const void *, int phase);
    int  (*emit)   (const void *, const struct be_cfg *, const void *opts,
                    const char *path);
    void (*mfree)  (void *);
} be_desc_t;
```

A backend owns its own command line. `flags` is the NULL-terminated
list it answers for, `parse` is called with each match and returns how
many extra argv entries it swallowed, and `opts` is a fixed slot the
driver hands back to every other op. Nothing about your target belongs
in `be_cfg_t`, which now carries only the settings every backend shares.

`is_on(opts)` is cheap; typically returns a flag your own `parse` set.
`warp_size` is optional and answers 32 by default; the IR asks for it
while lowering `warpSize`, which is what keeps target enums out of the
frontend. Return codes are `be_ret_t` (`BE_OK`, negative on failure,
`BE_UNSUP` when the op deliberately does nothing).

## Features

Declare only what you actually implement. A backend that claims
`BE_F_MFMA` and doesn't emit `mfma` instructions will fail the
conformance suite once that ships.

```
BE_F_SIMT       BE_F_SCALAR    BE_F_ATOMIC   BE_F_SHARED
BE_F_WARP       BE_F_MFMA      BE_F_BARRIER  BE_F_DIV
BE_F_SCRATCH    BE_F_TRANSC    BE_F_F16      BE_F_F64
BE_F_BF16       BE_F_MULTIOUT  BE_F_NOCALL
```

`BE_F_NOCALL` says your emitted code has no calling convention for
`__device__` functions, so the driver inlines them before isel.

## Recipe

1. `cp -r src/backend/skeleton src/backend/mygpu`
2. Rename `skel_` -> `mygpu_` in the copy.
3. Add the extern + list entry in `src/backend/backends.c`.
4. Add `src/backend/mygpu/mygpu.c` to `SOURCES` in the Makefile.
5. List your flags in the descriptor and set them in `parse`. The
   driver routes them to you; `src/main.c` needs no edit.
6. Document them in `usage()`, which the suite checks.
7. Implement `isel` first, then `emit`. Add feature bits to the
   descriptor as ops start working.

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
