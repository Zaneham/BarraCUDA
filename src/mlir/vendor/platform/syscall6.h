#ifndef COREC_PLATFORM_SYSCALL6_H
#define COREC_PLATFORM_SYSCALL6_H

// =============================================================================
// __builtin_syscall6 — the one compiler-independent raw-syscall primitive
// =============================================================================
//
// platform_<os>.c issues raw kernel syscalls (no libc) through a single
// intrinsic:
//
//     long __builtin_syscall6(long num, long a1, long a2,
//                             long a3, long a4, long a5, long a6);
//
// It returns the kernel's raw return value (a negative errno on failure).
//
// The point of routing every syscall through one intrinsic is that the
// platform sources stay BOTH compiler-independent AND architecture-independent:
//
//   * tinyC treats __builtin_syscall6 as a real compiler builtin and lowers it
//     to the target's syscall trap directly (x86_64 `syscall`, aarch64 `svc`).
//     tinyC has no inline asm, so this intrinsic is the only way it can trap
//     into the kernel.
//
//   * clang/gcc do NOT know this builtin, so this header polyfills it as a
//     static inline function whose body is the host architecture's inline-asm
//     syscall sequence. This is the ONLY place in the platform layer that
//     contains per-architecture syscall asm; the .c files never spell it out.
//
// Adding a new architecture means adding one branch here (for the clang/gcc
// build) plus a backend lowering in tinyC — the platform_<os>.c sources are
// untouched.

#ifndef __TINYC__

#if defined(__x86_64__)

// Linux/x86_64 kernel convention: number in rax; args in rdi, rsi, rdx, r10,
// r8, r9; trap via `syscall`; rcx and r11 are clobbered.
static inline long __corec_syscall6(long n, long a1, long a2, long a3,
                                    long a4, long a5, long a6) {
    long ret;
    register long r10 __asm__("r10") = a4;
    register long r8  __asm__("r8")  = a5;
    register long r9  __asm__("r9")  = a6;
    __asm__ volatile(
        "syscall"
        : "=a"(ret)
        : "a"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10), "r"(r8), "r"(r9)
        : "rcx", "r11", "memory");
    return ret;
}

#elif defined(__aarch64__)

// Linux/aarch64 kernel convention: number in x8; args in x0..x5; trap via
// `svc #0`; the result comes back in x0. (Note: this is the Linux ABI; the
// macOS arm64 trap is `svc #0x80` with the number in x16 — but macOS uses
// libSystem, not raw syscalls, so platform_macos.c never includes this header.)
static inline long __corec_syscall6(long n, long a1, long a2, long a3,
                                    long a4, long a5, long a6) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0") = a1;
    register long x1 __asm__("x1") = a2;
    register long x2 __asm__("x2") = a3;
    register long x3 __asm__("x3") = a4;
    register long x4 __asm__("x4") = a5;
    register long x5 __asm__("x5") = a6;
    __asm__ volatile(
        "svc #0"
        : "+r"(x0)
        : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5)
        : "memory");
    return x0;
}

#else
#error "syscall6.h: no __builtin_syscall6 fallback for this architecture"
#endif

#define __builtin_syscall6(n, a1, a2, a3, a4, a5, a6) \
    __corec_syscall6((long)(n), (long)(a1), (long)(a2), (long)(a3), \
                     (long)(a4), (long)(a5), (long)(a6))

#endif // !__TINYC__

#endif // COREC_PLATFORM_SYSCALL6_H
