#include <platform/platform.h>
#include <base/types.h>
#include <base/buddy.h>

// =============================================================================
// == macOS Implementation
// =============================================================================

// Define off_t as long long (macOS uses 64-bit off_t).
typedef long long off_t;

// Define struct iovec since we can't include headers
struct iovec {
    void *iov_base;
    size_t iov_len;
};

// Extern declarations for libSystem functions.
extern void* mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset);
extern int mprotect(void *addr, size_t len, int prot);
extern ssize_t writev(int fd, const struct iovec *iov, int iovcnt);
extern void _exit(int status);
extern int * __error(); // Returns pointer to errno
extern int open(const char *path, int flags, ...);
extern int close(int fd);
extern int dup(int fd);
extern int dup2(int oldfd, int newfd);
extern int fcntl(int fd, int cmd, ...);
extern ssize_t readv(int fd, const struct iovec *iov, int iovcnt);
extern off_t lseek(int fd, off_t offset, int whence);
extern int munmap(void *addr, size_t len);

// Protection and mapping flags (macOS-specific values)
#define PROT_NONE  0x00
#define PROT_READ  0x01
#define PROT_WRITE 0x02

#define MAP_PRIVATE 0x0002
#define MAP_ANONYMOUS 0x1000  // Different from Linux
#define MAP_FAILED ((void*)-1)

// fcntl commands
#define F_DUPFD 0

// Emulated heap state for macOS.
static uint8_t* linux_heap_base = NULL; // Reuse name for consistency
static size_t committed_pages = 0;
// 32 GiB virtual reservation. macOS arm64 has 47-bit user VA (128 TiB),
// so this is comfortably small relative to available address space.
// mmap(PROT_NONE) does not commit physical memory until mprotect makes
// a page RW, so the on-startup cost is just a single VM range entry
// in the kernel. Large enough for tinyc selfhost: the wasm-pipeline's
// wasm -> wasmstack -> wasmssa -> wmir lift on a ~3 MiB linked.wasm
// peaks at ~5 GiB of MLIR ops in the shared arena, well under 32 GiB.
static const size_t RESERVED_SIZE = 32ULL << 30; // 32 GiB

// Command line arguments storage
static int stored_argc = 0;
static char** stored_argv = NULL;

// Environment variables storage. `stored_envp` points to a NULL-terminated
// array of "KEY=VALUE" UTF-8 strings; `stored_environ_count` is the number
// of entries (excluding the terminating NULL).
static char** stored_envp = NULL;
static size_t stored_environ_count = 0;

typedef struct {
    void* addr;
    size_t size;
    bool in_use;
} MmapHandle;

#define MMAP_HANDLE_CAP 10
static MmapHandle g_mmap_handles[MMAP_HANDLE_CAP] = {0};

static void ensure_heap_initialized(void) {
    if (linux_heap_base == NULL) {
        linux_heap_base = (uint8_t*)mmap(
            NULL,
            RESERVED_SIZE,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0
        );
        if (linux_heap_base == (void*)-1) {
            // TODO: abort here if we cannot reserve the memory
            linux_heap_base = NULL;
        }
    }
}

// Implementation of fd_write using writev.
uint32_t platform_fd_write(int fd, const ciovec_t* iovs, size_t iovs_len, size_t* nwritten) {
    ssize_t ret = writev(fd, (const struct iovec *)iovs, (int)iovs_len);
    if (ret < 0) {
        *nwritten = 0;
        return (uint32_t)*__error(); // Get errno value
    }
    *nwritten = (size_t)ret;
    return 0;
}

void platform_exit(int status) {
    _exit(status);
}

void* platform_heap_base() {
    return linux_heap_base;
}

size_t platform_heap_size() {
    return committed_pages * PLATFORM_WASM_PAGE_SIZE;
}

static inline uintptr_t align(uintptr_t val, uintptr_t alignment) {
  return (val + alignment - 1) & ~(alignment - 1);
}

// Implementation of platform_heap_grow using mprotect to commit pages.
void* platform_heap_grow(size_t num_bytes) {
    size_t num_pages = align(num_bytes, PLATFORM_WASM_PAGE_SIZE) / PLATFORM_WASM_PAGE_SIZE;
    if (linux_heap_base == NULL) {
        return NULL;
    }

    size_t new_total_pages = committed_pages + num_pages;
    if ((new_total_pages * PLATFORM_WASM_PAGE_SIZE) > RESERVED_SIZE) {
        return NULL;
    }

    int ret = mprotect(
        (void*)(linux_heap_base + (committed_pages * PLATFORM_WASM_PAGE_SIZE)),
        num_pages * PLATFORM_WASM_PAGE_SIZE,
        PROT_READ | PROT_WRITE
    );

    if (ret != 0) {
        return NULL;
    }

    void* old_top = (void*)(linux_heap_base + (committed_pages * PLATFORM_WASM_PAGE_SIZE));
    committed_pages = new_total_pages;
    return old_top;
}

// Math functions using compiler builtins
double fast_sqrt(double x) {
    return __builtin_sqrt(x);
}

float fast_sqrtf(float x) {
    return __builtin_sqrtf(x);
}

// Public initialization function for hosts that provide their own entry
// point (PLATFORM_SKIP_ENTRY); the default _start path below calls this
// itself.
void platform_init(int argc, char** argv, char** envp) {
    stored_argc = argc;
    stored_argv = argv;
    stored_envp = envp;
    stored_environ_count = 0;
    if (envp) {
        while (envp[stored_environ_count]) stored_environ_count++;
    }
    ensure_heap_initialized();
    buddy_init();
}

// macOS open() flags
#define O_RDONLY   0x0000
#define O_WRONLY   0x0001
#define O_RDWR     0x0002
#define O_CREAT    0x0200
#define O_TRUNC    0x0400

// File I/O implementations
platform_fd_t platform_path_open(const char* path, size_t path_len, uint64_t rights, int oflags) {
    (void)path_len; /* path is NUL-terminated on this backend */
    // Extract access mode from rights
    int os_flags = 0;
    int has_read = (rights & PLATFORM_RIGHT_FD_READ) != 0;
    int has_write = (rights & PLATFORM_RIGHT_FD_WRITE) != 0;

    if (has_read && has_write) {
        os_flags |= O_RDWR;
    } else if (has_write) {
        os_flags |= O_WRONLY;
    } else {
        os_flags |= O_RDONLY;
    }

    // Map oflags to macOS creation flags
    if (oflags & PLATFORM_O_CREAT) os_flags |= O_CREAT;
    if (oflags & PLATFORM_O_TRUNC) os_flags |= O_TRUNC;

    // Default mode for created files (0644)
    int fd = open(path, os_flags, 0644);

    // On macOS, open() can return 0, 1, or 2 if those file descriptors were closed.
    // This would collide with stdin/stdout/stderr. To prevent this, we use fcntl()
    // with F_DUPFD to find the lowest available FD >= 3, then close the original low FD.
    // See: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/open.2.html
    // "The file descriptor returned by a successful call will be the lowest-numbered
    //  file descriptor not currently open for the process."
    // See: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html
    // "F_DUPFD: Find the lowest numbered available file descriptor greater than or
    //  equal to arg and make it be a copy of fd."
    if (fd >= 0 && fd <= PLATFORM_STDERR_FD) {
        // Use fcntl with F_DUPFD to get lowest available FD >= 3
        int new_fd = fcntl(fd, F_DUPFD, 3);
        if (new_fd < 0) {
            // fcntl failed, close the original FD and return error
            close(fd);
            return -1;
        }
        // Verify new_fd is not a reserved FD (should be > 2)
        if (new_fd <= PLATFORM_STDERR_FD) {
            // This should never happen, but if it does, close both and fail
            close(new_fd);
            close(fd);
            return -1;
        }
        close(fd);
        fd = new_fd;
    }

    return fd;
}

int platform_fd_close(platform_fd_t fd) {
    int result = close(fd);
    return (result < 0) ? *__error() : 0;
}

int platform_fd_read(platform_fd_t fd, const iovec_t* iovs, size_t iovs_len, size_t* nread) {
    ssize_t result = readv(fd, (const struct iovec*)iovs, (int)iovs_len);
    if (result < 0) {
        *nread = 0;
        return *__error();  // Return errno
    }
    *nread = (size_t)result;
    return 0;  // Success
}

int platform_fd_seek(platform_fd_t fd, int64_t offset, int whence, uint64_t* newoffset) {
    off_t result = lseek(fd, (off_t)offset, whence);
    if (result < 0) {
        *newoffset = 0;
        return *__error();  // Return errno
    }
    *newoffset = (uint64_t)result;
    return 0;  // Success
}

int platform_fd_tell(platform_fd_t fd, uint64_t* offset) {
    off_t result = lseek(fd, 0, PLATFORM_SEEK_CUR);
    if (result < 0) {
        *offset = 0;
        return *__error();  // Return errno
    }
    *offset = (uint64_t)result;
    return 0;  // Success
}

// Command line arguments implementation
int platform_args_sizes_get(size_t* argc, size_t* argv_buf_size) {
    *argc = (size_t)stored_argc;

    // Calculate total buffer size needed
    size_t total_size = 0;
    for (int i = 0; i < stored_argc; i++) {
        const char* arg = stored_argv[i];
        while (*arg++) total_size++;  // strlen
        total_size++;  // null terminator
    }
    *argv_buf_size = total_size;
    return 0;
}

int platform_args_get(char** argv, char* argv_buf) {
    char* buf_ptr = argv_buf;
    for (int i = 0; i < stored_argc; i++) {
        argv[i] = buf_ptr;
        const char* src = stored_argv[i];
        while (*src) {
            *buf_ptr++ = *src++;
        }
        *buf_ptr++ = '\0';
    }
    return 0;
}

// Environment variables implementation
int platform_environ_sizes_get(size_t* environ_count, size_t* environ_buf_size) {
    *environ_count = stored_environ_count;

    size_t total_size = 0;
    for (size_t i = 0; i < stored_environ_count; i++) {
        const char* e = stored_envp[i];
        while (*e++) total_size++;
        total_size++;
    }
    *environ_buf_size = total_size;
    return 0;
}

int platform_environ_get(char** environ, char* environ_buf) {
    char* buf_ptr = environ_buf;
    for (size_t i = 0; i < stored_environ_count; i++) {
        environ[i] = buf_ptr;
        const char* src = stored_envp[i];
        while (*src) {
            *buf_ptr++ = *src++;
        }
        *buf_ptr++ = '\0';
    }
    return 0;
}

bool platform_read_file_mmap(const char *filename, uint64_t *out_handle, void **out_data, size_t *out_size) {
    if (!filename || !out_handle || !out_data || !out_size) return false;
    *out_handle = 0;
    *out_data = NULL;
    *out_size = 0;

    int fd = open(filename, O_RDONLY, 0);
    if (fd < 0) {
        return false;
    }

    off_t end = lseek(fd, 0, PLATFORM_SEEK_END);
    if (end < 0) {
        close(fd);
        return false;
    }

    size_t file_size = (size_t)end;
    if (lseek(fd, 0, PLATFORM_SEEK_SET) < 0) {
        close(fd);
        return false;
    }

    if (file_size == 0) {
        close(fd);
        *out_size = 0;
        return true;
    }

    void *addr = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        return false;
    }

    int slot = -1;
    for (int i = 0; i < MMAP_HANDLE_CAP; i++) {
        if (!g_mmap_handles[i].in_use) {
            slot = i;
            break;
        }
    }
    if (slot == -1) {
        munmap(addr, file_size);
        return false;
    }

    g_mmap_handles[slot].addr = addr;
    g_mmap_handles[slot].size = file_size;
    g_mmap_handles[slot].in_use = true;

    *out_handle = (uint64_t)(slot + 1);
    *out_data = addr;
    *out_size = file_size;
    return true;
}

void platform_file_unmap(uint64_t handle) {
    if (handle == 0) return;
    uint64_t idx = handle - 1;
    if (idx >= MMAP_HANDLE_CAP) return;
    MmapHandle *h = &g_mmap_handles[idx];
    if (!h->in_use) return;
    if (h->addr && h->size > 0) {
        munmap(h->addr, h->size);
    }
    h->addr = NULL;
    h->size = 0;
    h->in_use = false;
}

#ifndef PLATFORM_SKIP_ENTRY
// Forward declaration for application entry point (only when platform provides entry)
int app_main();

// Initialize the platform and call the application
static int platform_init_and_run(int argc, char** argv, char** envp) {
    platform_init(argc, argv, envp);
    int status = app_main();
    return status;
}

// Entry point for macOS.
// macOS passes argc, argv, and envp to the entry point. The full ABI is
// `start(int argc, char** argv, char** envp, char** apple)`; we only need
// the first three.
void _start(int argc, char** argv, char** envp) {
    int status = platform_init_and_run(argc, argv, envp);
    platform_exit(status);
}
#endif
