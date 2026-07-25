# Numeric regression CI

Compiles Fortran `do concurrent` kernels through LFortran and Booth to every
backend, then checks the numbers against SLATEC known-good values. GPU IEEE
float is not CPU IEEE float, so a mismatch between backends is expected and not
by itself a failure. Two axes decide the verdict:

- **accuracy** — worst output element vs the known-good value. A loose ceiling,
  there only to catch a kernel that is actually broken. The saxpy guard-clause
  bug read 3.0 where 5.0 was due (40% off) and this caught it.
- **regression** — this run's mean vs the committed baseline for the *same*
  backend. Like-for-like, so the constant float differences fall out and only
  codegen drift shows. This is the real gate.

A backend with no baseline entry is recorded, not failed. Bless new numbers with
`--update-baseline` once you have looked at them.

## Layout

    manifest.json     kernels, their signatures, test cases, known-good values
    baseline.json     per-backend measured means; the regression yardstick
    kernels/*.f90      the do-concurrent sources
    run_numeric.py    the orchestrator

## Running

    KATH=./kath \
    LFORTRAN=lfortran \
    TINYGRAD_PATH=/path/to/tinygrad \
    python3 tests/numeric/run_numeric.py --backends cpu,rdna3,rdna4

`cpu`, `rdna3` and `rdna4` gate the build (numeric.yml). NVIDIA has no free CI
runner, so `numeric-nvidia.yml` runs `--backends nvidia` on a self-hosted box
only when a maintainer adds the `run-nvidia` label to a PR. The emulated RDNA
path is where a wave-divergence bug like the guard clause gets caught without a
card.

Only elementwise BLAS-1 lives here (saxpy, sscal, scopy). Reductions (sdot,
sasum, snrm2) are blocked upstream: LFortran's CUDA backend emits an empty
kernel for a `do concurrent ... reduce(+:)`, so there is nothing for kath to
compile yet. They land once that offload path does.

## Adding a kernel

Drop a `do concurrent` subroutine in `kernels/`, add a manifest entry with its
generated signature (check the `.cuda.cu` LFortran emits), its cases and the
known-good value, then `--update-baseline`. Ragged element counts (200, 255)
matter more than round ones: they are the ones that expose wave-mask bugs.
