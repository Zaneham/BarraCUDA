# Usage

[← back to README](../README.md)

## Building

```bash
make
```

### Requirements

- A C99 compiler (gcc, clang, whatever you've got)

## Command reference

The compiled binary is `kath` (after Kathleen Booth); the project is Booth. They differ on purpose, since a `booth` already lives in the Linux HA stack and you may have both on your PATH.

```bash
# Compile to AMD GPU binary (RDNA 3, default)
./kath --amdgpu-bin kernel.cu -o kernel.hsaco

# Compile for RDNA 2
./kath --amdgpu-bin --gfx1030 kernel.cu -o kernel.hsaco

# Compile for RDNA 4
./kath --amdgpu-bin --gfx1200 kernel.cu -o kernel.hsaco

# Compile to NVIDIA PTX
./kath --nvidia-ptx kernel.cu -o kernel.ptx

# Compile to Tenstorrent Metalium C++
./kath --tensix kernel.cu -o kernel_compute.cpp

# Compile to native RV32IM ELF for Tenstorrent baby cores
./kath --rv-elf kernel.cu -o kernel.elf

# Dump the TDF (Tile DataFlow) layout: regions, channels, NoC arcs
./kath --tdf kernel.cu

# HIP frontend (auto-on for .hip files, predefines __HIPCC__ and platform
# macros). Pair with any backend.
./kath --hip --amdgpu-bin kernel.hip -o kernel.hsaco
./kath --hip --nvidia-ptx kernel.hip -o kernel.ptx

# Triton frontend (parses @triton.jit Python through to BIR). Pair with
# any backend, or use --lex / --parse / --sema for inspection.
./kath --triton --amdgpu-bin kernel.py -o kernel.hsaco
./kath --triton --nvidia-ptx kernel.py -o kernel.ptx
./kath --triton --tensix       kernel.py -o kernel_compute.cpp

# CPU backend: compile a kernel to a host-runnable x86-64 object, no GPU
# needed. Link it with a host driver and run it like any other function.
./kath --cpu kernel.cu -o kernel.o
./kath --triton --cpu tests/tri_vadd.py -o vadd.o
# Calling convention: pass the kernel's own params, then one extra arg on
# the end, nthreads. The body runs once per thread_id up to nthreads, so a
# 1-D launch just hands it the element count. See examples/cpu_launch_vadd.c.

# RISC-V backend: same idea, RV64IMFD object you run under qemu-riscv64.
./kath --rv64 kernel.cu -o kernel_rv.o
# These are System V ELF objects, so link and run them on Linux (or WSL),
# not under MinGW (wrong ABI). The rv64 host has to be freestanding, see
# tests/diff for a worked runner.

# Differential testing: run the same kernel on two backends and diff the
# results. The CPU backend is the oracle, so a disagreement points at the
# other backend's codegen. Genuine cross-backend (x86 vs RISC-V), no GPU.
bash tests/diff/run_diff.sh

# Dump the IR (for debugging or curiosity)
./kath --ir kernel.cu

# Just parse and dump the AST
./kath --ast kernel.cu

# Run semantic analysis
./kath --sema kernel.cu

# Error messages in te reo Maori (or any language with a translation file)
./kath --lang lang/mi.txt --amdgpu-bin kernel.cu -o kernel.hsaco
```

## Runtime Launcher

Booth includes a minimal HSA runtime (`src/runtime/`) for dispatching compiled kernels on real AMD hardware. Zero compile-time dependency on ROCm. It loads `libhsa-runtime64.so` at runtime via `dlopen`.

```bash
# Compile the runtime and example together
gcc -std=c99 -O2 -I src/runtime \
    examples/launch_saxpy.c src/runtime/bc_runtime.c \
    -ldl -lm -o launch_saxpy

# Compile a kernel and run it
./kath --amdgpu-bin -o test.hsaco tests/canonical.cu
./launch_saxpy test.hsaco
```

Requires Linux with ROCm installed. See `examples/launch_saxpy.c` for a complete example.

## Fortran

Booth compiles Fortran `do concurrent` kernels by taking the CUDA source
[LFortran](https://lfortran.org/) emits for its GPU offload and compiling that
the rest of the way down. Elementwise arithmetic works today. See the
limitations at the end before planning around it.

Write a kernel the same way you would for LFortran:

```fortran
subroutine saxpy(x, y, a, n)
  integer, intent(in) :: n
  real, intent(in) :: x(n), a
  real, intent(inout) :: y(n)
  integer :: i
  do concurrent (i = 1:n)
    y(i) = a * x(i) + y(i)
  end do
end subroutine
```

Ask LFortran for the kernel source. No nvcc needed, the sidecar is written
beside the object file:

```bash
lfortran --gpu=cuda -c saxpy.f90 -o saxpy.o
# writes saxpy.o.cuda.cu
```

That file ends with a CUDA-runtime registration block, which is host glue for
`cudaLaunchKernel` and means nothing here. Cut it:

```bash
sed '/Auto-generated kernel registration/,$d' saxpy.o.cuda.cu > saxpy.cu
```

Then compile it for whatever you have:

```bash
./kath --amdgpu-bin saxpy.cu -o saxpy.hsaco   # AMD RDNA3
./kath --nvidia-ptx saxpy.cu -o saxpy.ptx     # NVIDIA
./kath --cpu        saxpy.cu -o saxpy.o       # x86-64, no GPU needed
```

### Calling it

Read the generated signature out of the `.cu` before you write a host driver,
because it will not match the Fortran argument order:

```c
extern "C" __global__ void __lfortran_gpu_kernel_0(
    float *x, float *y, float a, int n, int __loop_end_0)
```

Arrays come first, then scalars in symbol-table order rather than declaration
order, then a trailing `__loop_end_0` carrying the loop bound. The `--cpu`
backend adds one more argument on the end, `nthreads`, as it does for any
kernel.

`src/runtime/lf_gpu.c` implements LFortran's GPU offload ABI on top of the
NVIDIA runtime, so a Fortran program can launch kernels Booth produced.
`lf_gpu_hsa.c` is the AMD sibling.

### What is tested

`tests/numeric/` compiles Fortran kernels through this whole chain on every
push and checks the results against SLATEC known-good values, across x86-64,
RDNA3 and RDNA4. Six BLAS-1 routines so far: saxpy, sscal, scopy, sswap, srot
and srotm.

```bash
KATH=./kath LFORTRAN=lfortran \
python3 tests/numeric/run_numeric.py --backends cpu
```

See `tests/numeric/README.md` for adding a kernel.

### Limitations

Elementwise arithmetic only, and the reasons are all upstream of Booth:

- No reductions, so sdot, sasum and snrm2 are absent. A `do concurrent` with a
  `reduce` clause produces an empty kernel.
- Nested loops inside a `do concurrent` are dropped, which blocks Chebyshev
  recurrences and with them the SLATEC special functions
  ([lfortran#12369](https://github.com/lfortran/lfortran/issues/12369)).
- Intrinsics are emitted as `abs` and `exp` rather than `fabsf` and `expf`, so
  transcendentals do not resolve.
- Named constants used in a kernel body come through as uninitialised locals.

Tenstorrent is not supported from Fortran yet. The baby cores are RV32IM with
no FPU, so float has to go through a soft-float runtime, which is in progress.

If something breaks and you cannot tell whether it is an LFortran gap or a
Booth one, raise it here and it will get sorted out from this side.

## MLIR

`--mlir` reads MLIR text. The reader is Ondřej Čertík's, vendored under
`src/mlir/vendor`, and there is no LLVM in the path.

```bash
# Lower MLIR and compile it, same as any other frontend
./kath --mlir kernel.mlir --cpu -o kernel.o
./kath --mlir kernel.mlir --rv64 -o kernel.o

# Dump the BIR the lowering produced
./kath --mlir --ir kernel.mlir

# Reprint what was read instead of lowering it, for telling a misreading
# from a bad file
./kath --mlir --pp kernel.mlir
```

Accepted today: `func.func` with named arguments, `return`, `arith.constant`,
every `arith` integer and float binop, `arith.cmpi` and `arith.cmpf`, and the
`arith` conversions. Anything else is named on stderr and the whole lowering
fails, because an op skipped quietly is a kernel that compiles and computes
something different.

Nothing in MLIR marks a function as a kernel yet, so everything lowers as a
device function. `--cpu` and `--rv64` give you real code; the GPU backends will
take the file and then report zero kernels, which is an honest answer until
the `gpu` dialect arrives.
