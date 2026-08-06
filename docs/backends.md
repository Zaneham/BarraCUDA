# Adding a backend

[← back to README](../README.md)

If you want to add a target, or take one of the stubs and make it real,
this is the file for it. Everything a backend has to provide is in
`src/backend/backend.h`, and there is a skeleton in
`src/backend/skeleton/` you can copy and start filling in.

The reason this document exists at all is that the AMD backend is about
420 KB of source, and for a long time the only way to work out what a
backend actually had to do was to read it. That is a miserable way to
start. So the shape got pulled out into one struct with one page of
explanation, and now the AMD backend is just an example of the shape
rather than the definition of it.

## The pipeline

```
BIR -> isel -> [sched] -> [regalloc] -> [verify] -> emit -> file(s)
```

Only `isel` and `emit` are required. The bracketed phases are optional
and a NULL pointer means the driver skips that step, so a text-format
target like PTX can go straight from selection to emission without
pretending it has a register allocator.

## The contract

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

A backend owns its own command line, which is the part I would most
like you to notice. `flags` is the NULL-terminated list of options you
answer for, `parse` gets called with each one and returns how many
extra argv entries it swallowed, and `opts` is a fixed slot the driver
holds on to and hands back to every other op. That means nothing about
your target needs to go anywhere near `be_cfg_t`, which now carries
only the handful of settings every backend genuinely shares, and you
should not need to touch `src/main.c` at all.

`is_on` should be cheap and free of side effects. Usually it just
returns a flag your own `parse` set. `warp_size` is optional and
answers 32 if you leave it out. The IR asks for it while lowering
`warpSize`, long before your backend has built anything, and having it
answer a plain number is what keeps target-specific enums from leaking
back into the frontend.

Return codes are `be_ret_t`. `BE_OK` for success, something negative
for failure, and `BE_UNSUP` when an op deliberately does nothing yet.
The driver treats `BE_UNSUP` as a real error rather than quiet success,
which is on purpose. A half-finished backend that silently writes an
empty file is worse than one that says it cannot do this yet.

## Feature bits

```
BE_F_SIMT       BE_F_SCALAR    BE_F_ATOMIC   BE_F_SHARED
BE_F_WARP       BE_F_MFMA      BE_F_BARRIER  BE_F_DIV
BE_F_SCRATCH    BE_F_TRANSC    BE_F_F16      BE_F_F64
BE_F_BF16       BE_F_MULTIOUT  BE_F_NOCALL
```

Only claim what you have actually implemented. These are not
aspirations, they are what the conformance suite will key its tests off
once it lands, so a backend that claims `BE_F_MFMA` without emitting
`mfma` will simply fail.

Two of them do something today rather than just describing you.
`BE_F_NOCALL` says your emitted code has no calling convention for
`__device__` functions, so the driver inlines them before selection
ever runs. `BE_F_MULTIOUT` says you write more than one file, which
Tensix does.

## Roughly how it goes

1. `cp -r src/backend/skeleton src/backend/mygpu`
2. Rename `skel_` to your own prefix.
3. Add the extern and the list entry in `src/backend/backends.c`.
4. Add your file to `SOURCES` in the Makefile.
5. Put your flags in the descriptor and set them in `parse`. The driver
   routes them to you.
6. Document those flags in `usage()`. The test suite checks that every
   flag a backend declares turns up in `--help`, so this one is not
   optional.
7. Get `isel` working, then `emit`, and add feature bits as things
   start actually working rather than up front.

## Existing backends, by shape

Whichever of these is closest to your target is probably the best thing
to read first.

- `src/nvidia/nv_be.c` is the simplest complete one, isel and emit and
  nothing else.
- `src/amdgpu/amd_be.c` is the full pipeline with scheduling, register
  allocation and two verify passes.
- `src/tensix/tensix_be.c` emits several files from one path stem.
- `src/cpu/cpu_be.c` is the scalar, stack-everything shape.
- `src/intel/intel_be.c` is what a backend looks like while it is still
  being built.

## Things the driver assumes

- Every registered backend has a `name`, `is_on`, `isel` and `emit`.
- `is_on` has no side effects.
- `isel` allocates the module and `mfree` frees it, or `mfree` is NULL
  because you never allocated anything.
- An op that is not implemented returns `BE_UNSUP` rather than
  pretending it worked.

The usual coding rules in [CONTRIBUTING.md](../CONTRIBUTING.md) apply
here too.
