#pragma once

#include <base/types.h>

// -----------------------------------------------------------------------------
// wasm -> native lift host shim
//
// For the wasm->native lift (tinyC `--from-wasm`), the lifted module already
// contains the wasm-side platform_* (compiled from platform_wasm.c), which sit
// *above* the host copies on the call chain (app -> wasm platform_fd_write ->
// fd_write [wasi_adapter.c] -> host platform_fd_write). To reach the host copy
// without colliding with (or recursing into) the wasm-side one, the lift's host
// pieces -- wasm/wasi_adapter.c and the host platform_*.c -- are compiled with
// -DPLATFORM_HOST_SHIM, which renames the platform I/O entry points to
// __host_platform_*. Normal (non-lift) builds leave PLATFORM_HOST_SHIM undefined
// and are completely unaffected.
#ifdef PLATFORM_HOST_SHIM
#define platform_init      __host_platform_init
#define platform_exit      __host_platform_exit
#define platform_fd_write  __host_platform_fd_write
#define platform_fd_read   __host_platform_fd_read
#define platform_fd_close  __host_platform_fd_close
#define platform_fd_seek   __host_platform_fd_seek
#define platform_fd_tell   __host_platform_fd_tell
#define platform_path_open __host_platform_path_open
#endif
// -----------------------------------------------------------------------------

/*
 * The platform interface: the only way a C program in this project can
 * communicate with the system. Everything else (the `base/` library, any
 * higher-level libraries, applications) is built on top in a strictly
 * platform-independent way.
 *
 * The interface is intentionally small and curated. It is modeled after the
 * WASI capability model (file descriptors, explicit rights, no ambient
 * authority), but it is not WASI-specific: WASI is just one of several
 * backends. Each backend implements the same interface using whatever the
 * host system provides:
 *
 *   * WebAssembly: WASI imports (and Clang intrinsics for memory)
 *   * Linux:       raw syscalls
 *   * macOS:       the `libSystem.dylib` shared library
 *   * Windows:     the `kernel32.dll` shared library
 *
 * Because every system call funnels through this header, porting to a new
 * platform means writing exactly one new `platform_*.c` file.
 */

// Memory Handling
//
// The memory is organized as follows:
// WASM with Clang:
// [reserved] [data] [stack] [ heap of platform_heap_size() bytes ]
//                           ^ platform_heap_base()
//
// In WASM all of this is part of the samme linear contiguous memory. The
// __heap_base is created by wasm-ld (not be at a page boundary), and
// platform_heap_base()+platform_heap_size() starts at a page boundary.
//
// Linux with Clang:
// [reserved] [data] [...] [ heap of platform_heap_size() bytes ]         [stack]
//                         ^ platform_heap_base()
// The [...] are other ELF sections, and any possible shared libraries, maybe
// also some unused space. The [stack] starts at the highest virtual address
// and grows down. The [heap] is our own memory reserved to 4GB via mmap, and
// we commit to more pages when `heap_grow()` is called. The platform_heap_base()
// is the initial pointer returned by mmap.
// There will be regions which are not reserved (will segfault) both before
// heap and after heap.
//
// macOS and Windows work in a similar way.
//
// It is not guaranteed that all addresses below platform_heap_base() are
// addressable. The platform_heap_base() is at a page boundary on native platforms,
// but not in WASM.

#define PLATFORM_WASM_PAGE_SIZE 65536 // 64 KiB

// Returns a pointer to the base of the heap. The base might not be at a system
// page boundary.
void* platform_heap_base();

// Returns the size of the heap in bytes. The heap size is not in general a
// multiple of a page size, because the heap base might not lie on a page
// boundary.
size_t platform_heap_size();

// Grows the heap by `num_bytes` bytes. If not multiple of a system-dependent
// page size (not necessarily equal to PLATFORM_WASM_PAGE_SIZE, although on most systems
// PLATFORM_WASM_PAGE_SIZE is usually a multiple of the system page size), it will round
// up to an even multiple of system page size.
// Returns the pointer to the new region (equal to the last
// `platform_heap_base()+platform_heap_size()`)
void* platform_heap_grow(size_t num_bytes);



// Write multiple buffers to the file descriptor.
typedef struct ciovec_s {
    const void* buf;
    size_t buf_len;
} ciovec_t;
uint32_t platform_fd_write(int fd, const ciovec_t* iovs, size_t iovs_len, size_t* nwritten);


// Terminate the process with exit code `status`
void platform_exit(int status);


// File I/O
//
// File descriptor type - opaque handle to an open file
typedef int platform_fd_t;

// I/O vector for scatter-gather operations (read buffers)
typedef struct iovec_s {
    void* iov_base;
    size_t iov_len;
} iovec_t;

// Rights flags (capabilities for file operations). Values match the WASI
// __WASI_RIGHTS_* constants; non-WASI backends translate as appropriate.
#define PLATFORM_RIGHT_FD_READ   0x2   // (1 << 1)
#define PLATFORM_RIGHT_FD_WRITE  0x40  // (1 << 6)
#define PLATFORM_RIGHT_FD_SEEK   0x4   // (1 << 2)
#define PLATFORM_RIGHT_FD_TELL   0x20  // (1 << 5)

// Common rights combinations
#define PLATFORM_RIGHTS_READ  (PLATFORM_RIGHT_FD_READ | PLATFORM_RIGHT_FD_SEEK | PLATFORM_RIGHT_FD_TELL)
#define PLATFORM_RIGHTS_WRITE (PLATFORM_RIGHT_FD_WRITE | PLATFORM_RIGHT_FD_SEEK | PLATFORM_RIGHT_FD_TELL)
#define PLATFORM_RIGHTS_RDWR  (PLATFORM_RIGHTS_READ | PLATFORM_RIGHTS_WRITE)

// File creation flags. Values match the WASI __WASI_OFLAGS_* constants.
#define PLATFORM_O_CREAT   0x1  // (1 << 0)
#define PLATFORM_O_TRUNC   0x8  // (1 << 3)

// Seek whence values
#define PLATFORM_SEEK_SET 0
#define PLATFORM_SEEK_CUR 1
#define PLATFORM_SEEK_END 2

// Special file descriptors
#define PLATFORM_STDIN_FD  0
#define PLATFORM_STDOUT_FD 1
#define PLATFORM_STDERR_FD 2

// Open a file at the given path with the specified rights and creation flags.
// rights: combination of PLATFORM_RIGHTS_READ, PLATFORM_RIGHTS_WRITE, or PLATFORM_RIGHTS_RDWR
// oflags: combination of PLATFORM_O_CREAT, PLATFORM_O_TRUNC
// Returns a file descriptor on success, or -1 on error.
platform_fd_t platform_path_open(const char* path, size_t path_len, uint64_t rights, int oflags);

// Close a file descriptor.
// Returns 0 on success, or errno on error.
int platform_fd_close(platform_fd_t fd);

// Read from a file descriptor using scatter-gather I/O.
// Returns 0 on success with bytes read in *nread, or errno on error.
int platform_fd_read(platform_fd_t fd, const iovec_t* iovs, size_t iovs_len, size_t* nread);

// Seek to a position in the file.
// Returns 0 on success with new position in *newoffset, or errno on error.
int platform_fd_seek(platform_fd_t fd, int64_t offset, int whence, uint64_t* newoffset);

// Get the current position in the file.
// Returns 0 on success with position in *offset, or errno on error.
int platform_fd_tell(platform_fd_t fd, uint64_t* offset);


// Command Line Arguments
//
// Get the sizes of the command line arguments.
// Returns 0 on success with:
//   *argc: number of arguments
//   *argv_buf_size: total size needed to store all argument strings (including null terminators)
// Returns errno on error.
int platform_args_sizes_get(size_t* argc, size_t* argv_buf_size);

// Get the command line arguments.
// Parameters:
//   argv: array of pointers to be filled with argument string pointers
//   argv_buf: buffer to store the actual argument strings
// The caller must allocate:
//   - argv array of at least argc pointers
//   - argv_buf buffer of at least argv_buf_size bytes
// Returns 0 on success, or errno on error.
int platform_args_get(char** argv, char* argv_buf);


// Environment Variables
//
// Get the sizes of the environment variables. Each environment variable is a
// UTF-8 "KEY=VALUE" string terminated by a NUL byte.
// Returns 0 on success with:
//   *environ_count: number of environment variables
//   *environ_buf_size: total size needed to store all environment strings
//                      (including null terminators)
// Returns errno on error.
int platform_environ_sizes_get(size_t* environ_count, size_t* environ_buf_size);

// Get the environment variables.
// Parameters:
//   environ: array of pointers to be filled with "KEY=VALUE" string pointers
//   environ_buf: buffer to store the actual environment strings
// The caller must allocate:
//   - environ array of at least environ_count pointers
//   - environ_buf buffer of at least environ_buf_size bytes
// Returns 0 on success, or errno on error.
int platform_environ_get(char** environ, char* environ_buf);


//=============================================================================
// Platform Initialization
//=============================================================================
//
// Initialize the platform runtime (heap, buddy allocator, command line args,
// environment variables).
//
// USAGE PATTERNS:
//
// 1. When PLATFORM_SKIP_ENTRY is DEFINED (platform skips entry point):
//    - The platform does NOT provide _start or any entry point implementation
//    - You MUST provide your own entry point (main, SDL_main, etc.)
//    - You MUST call platform_init(argc, argv, envp) manually in your entry point
//    - Do NOT implement app_main()
//    - Example: SDL apps call this in SDL_AppInit()
//
// 2. When PLATFORM_SKIP_ENTRY is NOT defined (platform provides entry point):
//    - The platform provides _start which calls platform_init() then app_main()
//    - You MUST implement app_main()
//    - Do NOT call platform_init() - it's called automatically by _start
//
// Parameters:
//   argc: argument count (may be 0 for platforms without argc/argv)
//   argv: argument vector (may be NULL for platforms without argc/argv)
//   envp: environment vector, NULL-terminated array of "KEY=VALUE" strings
//         (may be NULL for platforms that source the environment through
//          their host API rather than through the entry point, e.g. Windows
//          and WebAssembly/WASI)
//
void platform_init(int argc, char** argv, char** envp);


// Math Functions
//
// Square root functions using platform-specific builtins
double fast_sqrt(double x);
float fast_sqrtf(float x);

//=============================================================================
// File mapping (read-only, private)
//=============================================================================
//
// Attempts to map a file into memory for read-only, private (COW) access.
// On success:
//   *out_handle is set to an opaque handle that must be passed to
//     platform_file_unmap when done (may be 0 for empty files)
//   *out_data points to the mapped bytes (or NULL for empty files)
//   *out_size is the file size in bytes
// Returns true on success, false on failure.
//
// Platform behavior:
//   - Linux/macOS/Windows: uses mmap/MapViewOfFile. If mapping fails,
//     returns false (no heap copy fallback here).
//   - WASM: returns false immediately (no mmap available).
//
// Callers should fall back to a regular buffered read (e.g., read_file)
// when this returns false.
bool platform_read_file_mmap(const char *filename, uint64_t *out_handle, void **out_data, size_t *out_size);

// Releases a mapping obtained from platform_read_file_mmap.
// Safe to call with handle == 0. Resets/cleans any internal state for that handle.
void platform_file_unmap(uint64_t handle);
