#include <platform/platform.h>
#include <base/types.h>
#include <base/buddy.h>

// =============================================================================
// == Windows Implementation (MSVC)
// =============================================================================

// Windows API declarations - we'll define only what we need
typedef void* HANDLE;
typedef unsigned long DWORD;
typedef DWORD* LPDWORD;
typedef void* LPVOID;
typedef const void* LPCVOID;
typedef unsigned long long SIZE_T;
typedef long long LONGLONG;
typedef unsigned short wchar_t;
typedef union {
    struct {
        DWORD LowPart;
        DWORD HighPart;
    };
    LONGLONG QuadPart;
} LARGE_INTEGER;

// Windows constants
#define STD_INPUT_HANDLE ((DWORD)-10)
#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define STD_ERROR_HANDLE ((DWORD)-12)
#define MEM_COMMIT 0x1000
#define MEM_RESERVE 0x2000
#define PAGE_READWRITE 0x04
#define INVALID_HANDLE_VALUE ((HANDLE)(long long)-1)
#define GENERIC_READ 0x80000000
#define GENERIC_WRITE 0x40000000
#define CREATE_NEW 1
#define CREATE_ALWAYS 2
#define OPEN_EXISTING 3
#define OPEN_ALWAYS 4
#define TRUNCATE_EXISTING 5
#define FILE_SHARE_READ 0x00000001
#define FILE_SHARE_WRITE 0x00000002
#define FILE_BEGIN 0
#define FILE_CURRENT 1
#define FILE_END 2
#define FILE_ATTRIBUTE_NORMAL 0x00000080
#define FILE_FLAG_SEQUENTIAL_SCAN 0x08000000
#define FILE_MAP_COPY 0x00000001
#define FILE_MAP_WRITE 0x00000002
#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

// Windows API function declarations
__declspec(dllimport) HANDLE __stdcall GetStdHandle(DWORD nStdHandle);
__declspec(dllimport) int __stdcall WriteFile(HANDLE hFile, LPCVOID lpBuffer, DWORD nNumberOfBytesToWrite, LPDWORD lpNumberOfBytesWritten, void* lpOverlapped);
__declspec(dllimport) LPVOID __stdcall VirtualAlloc(LPVOID lpAddress, SIZE_T dwSize, DWORD flAllocationType, DWORD flProtect);
__declspec(dllimport) void __stdcall ExitProcess(unsigned int uExitCode);
__declspec(dllimport) HANDLE __stdcall CreateFileA(const char* lpFileName, DWORD dwDesiredAccess, DWORD dwShareMode, void* lpSecurityAttributes, DWORD dwCreationDisposition, DWORD dwFlagsAndAttributes, HANDLE hTemplateFile);
__declspec(dllimport) int __stdcall CloseHandle(HANDLE hObject);
__declspec(dllimport) int __stdcall ReadFile(HANDLE hFile, LPVOID lpBuffer, DWORD nNumberOfBytesToRead, LPDWORD lpNumberOfBytesRead, void* lpOverlapped);
__declspec(dllimport) int __stdcall SetFilePointerEx(HANDLE hFile, LARGE_INTEGER liDistanceToMove, LARGE_INTEGER* lpNewFilePointer, DWORD dwMoveMethod);
__declspec(dllimport) wchar_t* __stdcall GetCommandLineW(void);
__declspec(dllimport) wchar_t** __stdcall CommandLineToArgvW(const wchar_t* lpCmdLine, int* pNumArgs);
__declspec(dllimport) HANDLE __stdcall LocalFree(HANDLE hMem);
__declspec(dllimport) wchar_t* __stdcall GetEnvironmentStringsW(void);
__declspec(dllimport) int __stdcall FreeEnvironmentStringsW(wchar_t* penv);
__declspec(dllimport) HANDLE __stdcall CreateFileMappingA(HANDLE hFile, void* lpFileMappingAttributes, DWORD flProtect, DWORD dwMaximumSizeHigh, DWORD dwMaximumSizeLow, const char* lpName);
__declspec(dllimport) LPVOID __stdcall MapViewOfFile(HANDLE hFileMappingObject, DWORD dwDesiredAccess, DWORD dwFileOffsetHigh, DWORD dwFileOffsetLow, size_t dwNumberOfBytesToMap);
__declspec(dllimport) int __stdcall UnmapViewOfFile(LPCVOID lpBaseAddress);
__declspec(dllimport) int __stdcall GetFileSizeEx(HANDLE hFile, LARGE_INTEGER* lpFileSize);
__declspec(dllimport) void __stdcall Sleep(DWORD dwMilliseconds);

// Our emulated heap state for Windows
static uint8_t* windows_heap_base = NULL;
static size_t committed_pages = 0;
static const size_t RESERVED_SIZE = 1ULL << 32; // Reserve 4GB of virtual address space

// Command line arguments storage (UTF-8 converted)
static int stored_argc = 0;
static char** stored_argv = NULL;
static char* stored_argv_buf = NULL;

// Environment variables storage (UTF-8 converted). `stored_environ` is a
// table of `stored_environ_count` pointers into `stored_environ_buf`, each
// pointing to a "KEY=VALUE" UTF-8 string terminated by a NUL.
static size_t stored_environ_count = 0;
static char** stored_environ = NULL;
static char* stored_environ_buf = NULL;

typedef struct {
    void   *view;
    HANDLE  hFile;
    HANDLE  hMapping;
    size_t  size;
    bool    in_use;
} MmapHandle;

#define MMAP_HANDLE_CAP 10
static MmapHandle g_mmap_handles[MMAP_HANDLE_CAP] = {0};

// Emulation of `fd_write` using Windows WriteFile API
uint32_t platform_fd_write(int fd, const ciovec_t* iovs, size_t iovs_len, size_t* nwritten) {
    HANDLE hOutput;

    // Handle standard streams specially
    if (fd == PLATFORM_STDOUT_FD) {
        hOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    } else if (fd == PLATFORM_STDERR_FD) {
        hOutput = GetStdHandle(STD_ERROR_HANDLE);
    } else {
        // Treat as a file handle returned from platform_path_open
        hOutput = (HANDLE)(long long)fd;
    }

    if (hOutput == INVALID_HANDLE_VALUE) {
        *nwritten = 0;
        return 1; // Error
    }

    size_t total_written = 0;
    for (size_t i = 0; i < iovs_len; i++) {
        DWORD bytes_written = 0;
        int result = WriteFile(hOutput, iovs[i].buf, (DWORD)iovs[i].buf_len, &bytes_written, NULL);
        if (!result) {
            *nwritten = total_written;
            return 1; // Error
        }
        total_written += bytes_written;
    }

    *nwritten = total_written;
    return 0; // Success
}

// Initialize the heap by reserving and committing initial memory
static void ensure_heap_initialized() {
    if (windows_heap_base == NULL) {
        // Reserve a large virtual address space
        windows_heap_base = (uint8_t*)VirtualAlloc(NULL, RESERVED_SIZE, MEM_RESERVE, PAGE_READWRITE);
        if (windows_heap_base == NULL) {
            platform_exit(1); // Failed to reserve memory
        }

        // Commit the first page
        void* committed = VirtualAlloc(windows_heap_base, PLATFORM_WASM_PAGE_SIZE, MEM_COMMIT, PAGE_READWRITE);
        if (committed == NULL) {
            platform_exit(1); // Failed to commit memory
        }
        committed_pages = 1;
    }
}

static inline uintptr_t align(uintptr_t val, uintptr_t alignment) {
  return (val + alignment - 1) & ~(alignment - 1);
}

// Windows platform_heap_grow implementation using VirtualAlloc
void* platform_heap_grow(size_t num_bytes) {
    size_t num_pages = align(num_bytes, PLATFORM_WASM_PAGE_SIZE) / PLATFORM_WASM_PAGE_SIZE;

    if (num_pages == 0) {
        // TODO: what should be returned here?
        return (void*)(committed_pages * PLATFORM_WASM_PAGE_SIZE);
    }

    size_t bytes_to_commit = num_pages * PLATFORM_WASM_PAGE_SIZE;
    void* new_memory = VirtualAlloc(
        windows_heap_base + (committed_pages * PLATFORM_WASM_PAGE_SIZE),
        bytes_to_commit,
        MEM_COMMIT,
        PAGE_READWRITE
    );

    if (new_memory == NULL) {
        return (void*)-1; // Growth failed
    }

    size_t prev_size = committed_pages;
    committed_pages += num_pages;
    return (void*)(windows_heap_base + prev_size * PLATFORM_WASM_PAGE_SIZE);
}

void* platform_heap_base() {
    return windows_heap_base;
}

// Windows platform_heap_size implementation
size_t platform_heap_size() {
    return committed_pages * PLATFORM_WASM_PAGE_SIZE;
}

#ifndef PLATFORM_SKIP_ENTRY
// Stub for __chkstk which is normally provided by the C runtime
// Since we're using /kernel flag, we need to provide this ourselves.
// When PLATFORM_SKIP_ENTRY is defined we are linking against the real
// CRT (e.g. /MD), which provides a correct __chkstk that probes /
// commits stack pages; replacing it with this no-op stub would cause
// silent stack-guard-page faults (0xC0000005) for any function whose
// frame exceeds one page (4KB on x64).
void __chkstk() {
    // Do nothing - we're not using large stack allocations
}

// _fltused is required by MSVC when using floating-point operations
// This symbol must be present when using floating-point without the CRT
int _fltused = 1;
#endif

// Process exit function
void platform_exit(int status) {
    ExitProcess((unsigned int)status);
}

// Forward declaration for command line argument initialization
static void init_args();
static void init_environ();

// Math functions using compiler builtins
double fast_sqrt(double x) {
    if (x == 0.0) return 0.0; // Handle zero for correctness
    double xhalf = 0.5 * x;

    // Use union for standard-compliant type punning (avoids strict-aliasing issues)
    union {
        double d;
        uint64_t ui;
    } u;
    u.d = x;
    uint64_t i = u.ui;

    i = 0x5fe6eb50c7b537a9ULL - (i >> 1); // Magic number for initial inverse sqrt guess (double precision)
    u.ui = i;
    double y = u.d;

    y = y * (1.5 - xhalf * y * y); // First Newton-Raphson refinement
    // Optional: Uncomment for better accuracy (~full double precision with two iterations), adds ~10-20% runtime
    y = y * (1.5 - xhalf * y * y); // Second refinement

    return x * y; // Convert inverse sqrt to sqrt
}

float fast_sqrtf(float x) {
    if (x == 0.0f) return 0.0f;  // Handle zero for correctness

    float xhalf = 0.5f * x;
    union { float f; int i; } u;  // Reinterpret float bits as int
    u.f = x;
    int i = 0x5f3759df - (u.i >> 1); // Magic number for initial inverse sqrt guess
    u.i = i;
    float y = u.f;                // Reinterpret back to float
    y = y * (1.5f - xhalf * y * y);  // First Newton-Raphson refinement

    // Optional: Uncomment for better accuracy (~full float precision), adds ~10-20% runtime
    y = y * (1.5f - xhalf * y * y);  // Second refinement

    return x * y;  // Convert inverse sqrt to sqrt
}

// Public initialization function for hosts that provide their own entry
// point (PLATFORM_SKIP_ENTRY); the default _start path below calls this
// itself.
void platform_init(int argc, char** argv, char** envp) {
    (void)argc; (void)argv; (void)envp;
    init_args();
    init_environ();
    ensure_heap_initialized();
    buddy_init();
}

// File I/O implementations
platform_fd_t platform_path_open(const char* path, size_t path_len, uint64_t rights, int oflags) {
    (void)path_len; /* path is NUL-terminated on this backend */
    // Extract access mode from rights
    DWORD access = 0;
    DWORD creation = OPEN_EXISTING;
    int has_read = (rights & PLATFORM_RIGHT_FD_READ) != 0;
    int has_write = (rights & PLATFORM_RIGHT_FD_WRITE) != 0;

    if (has_read && has_write) {
        access = GENERIC_READ | GENERIC_WRITE;
    } else if (has_write) {
        access = GENERIC_WRITE;
    } else {
        access = GENERIC_READ;
    }

    // Map oflags to Windows creation disposition.
    //
    // Note: we treat O_TRUNC alone the same as O_CREAT|O_TRUNC and use
    // CREATE_ALWAYS, rather than TRUNCATE_EXISTING. TRUNCATE_EXISTING has
    // been observed to fail intermittently on CI runners (sharing/scanning
    // races right after we close the same path). CREATE_ALWAYS reliably
    // produces the desired end state — a zero-length file open for writing
    // — and matches POSIX O_TRUNC semantics whenever the file already
    // exists, which is the normal use case.
    if (oflags & PLATFORM_O_TRUNC) {
        creation = CREATE_ALWAYS;  // Create new or truncate existing.
    } else if (oflags & PLATFORM_O_CREAT) {
        creation = OPEN_ALWAYS;    // Open existing or create new.
    } else {
        creation = OPEN_EXISTING;  // Open existing file.
    }

    HANDLE handle = INVALID_HANDLE_VALUE;
    // Retry CreateFileA briefly to tolerate transient sharing violations on
    // Windows. After we close a file, an antivirus or the search indexer can
    // briefly open it with restrictive share modes, causing our next open of
    // the same path to fail with ERROR_SHARING_VIOLATION. The window is
    // typically a few milliseconds.
    for (int attempt = 0; attempt < 20; attempt++) {
        handle = CreateFileA(
            path,
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL,
            creation,
            0,
            NULL
        );
        if (handle != INVALID_HANDLE_VALUE) break;
        Sleep(50);
    }

    if (handle == INVALID_HANDLE_VALUE) {
        return -1;
    }

    // Return handle cast to int (platform_fd_t)
    // Note: Windows HANDLEs are pointers (typically large values) and will never
    // collide with the special file descriptor values 0, 1, or 2 (stdin/stdout/stderr).
    // See: https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea
    // HANDLEs are kernel object handles, not POSIX-style small integer file descriptors.
    return (platform_fd_t)(long long)handle;
}

int platform_fd_close(platform_fd_t fd) {
    HANDLE handle = (HANDLE)(long long)fd;
    return CloseHandle(handle) ? 0 : 1;  // Return 0 on success, non-zero on error
}

int platform_fd_read(platform_fd_t fd, const iovec_t* iovs, size_t iovs_len, size_t* nread) {
    HANDLE handle;

    // Handle standard input specially
    if (fd == PLATFORM_STDIN_FD) {
        handle = GetStdHandle(STD_INPUT_HANDLE);
    } else {
        // Treat as a file handle returned from platform_path_open
        handle = (HANDLE)(long long)fd;
    }

    if (handle == INVALID_HANDLE_VALUE) {
        *nread = 0;
        return 1;  // Error
    }

    size_t total_read = 0;

    // Windows doesn't have readv, so loop over iovecs
    for (size_t i = 0; i < iovs_len; i++) {
        DWORD bytes_read = 0;
        int result = ReadFile(handle, iovs[i].iov_base, (DWORD)iovs[i].iov_len, &bytes_read, NULL);
        if (!result) {
            *nread = total_read;
            return 1;  // Error
        }
        total_read += bytes_read;
        if (bytes_read < iovs[i].iov_len) {
            // Short read, stop here
            break;
        }
    }

    *nread = total_read;
    return 0;  // Success
}

int platform_fd_seek(platform_fd_t fd, int64_t offset, int whence, uint64_t* newoffset) {
    HANDLE handle = (HANDLE)(long long)fd;
    LARGE_INTEGER distance;
    LARGE_INTEGER new_position;
    distance.QuadPart = offset;

    DWORD method = FILE_BEGIN;
    if (whence == PLATFORM_SEEK_CUR) method = FILE_CURRENT;
    else if (whence == PLATFORM_SEEK_END) method = FILE_END;

    int result = SetFilePointerEx(handle, distance, &new_position, method);
    if (!result) {
        *newoffset = 0;
        return 1;  // Error
    }
    *newoffset = (uint64_t)new_position.QuadPart;
    return 0;  // Success
}

int platform_fd_tell(platform_fd_t fd, uint64_t* offset) {
    HANDLE handle = (HANDLE)(long long)fd;
    LARGE_INTEGER distance;
    LARGE_INTEGER position;
    distance.QuadPart = 0;

    int result = SetFilePointerEx(handle, distance, &position, FILE_CURRENT);
    if (!result) {
        *offset = 0;
        return 1;  // Error
    }
    *offset = (uint64_t)position.QuadPart;
    return 0;  // Success
}

// Helper: Convert UTF-16 wide char to UTF-8
// Returns number of bytes written (1-3 for BMP), or 0 on error
// Note: This handles Basic Multilingual Plane only (wchar_t is 16-bit on Windows)
static int wchar_to_utf8(wchar_t wc, char* out) {
    if (wc < 0x80) {
        out[0] = (char)wc;
        return 1;
    } else if (wc < 0x800) {
        out[0] = (char)(0xC0 | (wc >> 6));
        out[1] = (char)(0x80 | (wc & 0x3F));
        return 2;
    } else {
        // BMP character (up to 0xFFFF)
        out[0] = (char)(0xE0 | (wc >> 12));
        out[1] = (char)(0x80 | ((wc >> 6) & 0x3F));
        out[2] = (char)(0x80 | (wc & 0x3F));
        return 3;
    }
}

// Helper: Get length of wide string
static size_t wcslen(const wchar_t* str) {
    size_t len = 0;
    while (*str++) len++;
    return len;
}

// Helper: Convert wide string to UTF-8
// Returns number of bytes written (excluding null terminator)
static size_t widestr_to_utf8(const wchar_t* wstr, char* out, size_t out_size) {
    size_t written = 0;
    while (*wstr && written + 4 < out_size) {
        int bytes = wchar_to_utf8(*wstr, out + written);
        if (bytes == 0) break;
        written += (size_t)bytes;
        wstr++;
    }
    if (written < out_size) {
        out[written] = '\0';
    }
    return written;
}

// Initialize command line arguments from Windows API
static void init_args() {
    if (stored_argv != NULL) return;  // Already initialized

    wchar_t* cmd_line = GetCommandLineW();
    int argc = 0;
    wchar_t** wargv = CommandLineToArgvW(cmd_line, &argc);

    if (wargv == NULL || argc == 0) {
        stored_argc = 0;
        stored_argv = NULL;
        return;
    }

    // Calculate total buffer size needed for UTF-8 conversion
    size_t total_size = 0;
    for (int i = 0; i < argc; i++) {
        // Worst case: each wide char becomes 3 UTF-8 bytes (BMP only)
        total_size += wcslen(wargv[i]) * 3 + 1;
    }

    // Allocate storage using VirtualAlloc
    stored_argv = (char**)VirtualAlloc(NULL, (size_t)argc * sizeof(char*), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    stored_argv_buf = (char*)VirtualAlloc(NULL, total_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (!stored_argv || !stored_argv_buf) {
        stored_argc = 0;
        LocalFree((HANDLE)wargv);
        return;
    }

    // Convert each argument to UTF-8
    char* buf_ptr = stored_argv_buf;
    for (int i = 0; i < argc; i++) {
        stored_argv[i] = buf_ptr;
        size_t bytes = widestr_to_utf8(wargv[i], buf_ptr, total_size - (size_t)(buf_ptr - stored_argv_buf));
        buf_ptr += bytes + 1;  // +1 for null terminator
    }

    stored_argc = argc;
    LocalFree((HANDLE)wargv);
}

// Command line arguments implementation
int platform_args_sizes_get(size_t* argc, size_t* argv_buf_size) {
    init_args();

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
    init_args();

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

// Initialize environment variables from Windows API. GetEnvironmentStringsW
// returns a pointer to a block of "KEY=VALUE\0KEY=VALUE\0...\0" UTF-16
// strings, terminated by an extra NUL (double-NUL termination).
static void init_environ() {
    if (stored_environ != NULL) return;  // Already initialized

    wchar_t* env_block = GetEnvironmentStringsW();
    if (env_block == NULL) {
        stored_environ_count = 0;
        return;
    }

    // First pass: count entries and worst-case UTF-8 byte count.
    size_t count = 0;
    size_t total_size = 0;
    wchar_t* p = env_block;
    while (*p) {
        size_t wlen = wcslen(p);
        // Worst case: each wide char becomes 3 UTF-8 bytes (BMP only).
        total_size += wlen * 3 + 1;
        count++;
        p += wlen + 1;
    }

    if (count == 0) {
        FreeEnvironmentStringsW(env_block);
        stored_environ_count = 0;
        return;
    }

    // Allocate storage using VirtualAlloc (same pattern as init_args).
    stored_environ = (char**)VirtualAlloc(NULL, count * sizeof(char*), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    stored_environ_buf = (char*)VirtualAlloc(NULL, total_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (!stored_environ || !stored_environ_buf) {
        FreeEnvironmentStringsW(env_block);
        stored_environ_count = 0;
        stored_environ = NULL;
        return;
    }

    // Second pass: convert each entry to UTF-8.
    char* buf_ptr = stored_environ_buf;
    p = env_block;
    for (size_t i = 0; i < count; i++) {
        stored_environ[i] = buf_ptr;
        size_t remaining = total_size - (size_t)(buf_ptr - stored_environ_buf);
        size_t bytes = widestr_to_utf8(p, buf_ptr, remaining);
        buf_ptr += bytes + 1;  // +1 for null terminator
        p += wcslen(p) + 1;
    }

    stored_environ_count = count;
    FreeEnvironmentStringsW(env_block);
}

// Environment variables implementation
int platform_environ_sizes_get(size_t* environ_count, size_t* environ_buf_size) {
    init_environ();

    *environ_count = stored_environ_count;

    size_t total_size = 0;
    for (size_t i = 0; i < stored_environ_count; i++) {
        const char* e = stored_environ[i];
        while (*e++) total_size++;
        total_size++;
    }
    *environ_buf_size = total_size;
    return 0;
}

int platform_environ_get(char** environ, char* environ_buf) {
    init_environ();

    char* buf_ptr = environ_buf;
    for (size_t i = 0; i < stored_environ_count; i++) {
        environ[i] = buf_ptr;
        const char* src = stored_environ[i];
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

    HANDLE hFile = CreateFileA(
        filename,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        return false;
    }
    if (fileSize.QuadPart > (LONGLONG)SIZE_MAX) {
        CloseHandle(hFile);
        return false;
    }
    size_t file_size = (size_t)fileSize.QuadPart;
    if (file_size == 0) {
        CloseHandle(hFile);
        *out_size = 0;
        return true;
    }

    HANDLE hMapping = CreateFileMappingA(
        hFile,
        NULL,
        PAGE_READWRITE,
        0, 0,
        NULL
    );
    if (!hMapping) {
        CloseHandle(hFile);
        return false;
    }

    void *view = MapViewOfFile(
        hMapping,
        FILE_MAP_COPY | FILE_MAP_WRITE,
        0, 0, 0
    );
    if (!view) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
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
        UnmapViewOfFile(view);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return false;
    }

    g_mmap_handles[slot].view = view;
    g_mmap_handles[slot].hFile = hFile;
    g_mmap_handles[slot].hMapping = hMapping;
    g_mmap_handles[slot].size = file_size;
    g_mmap_handles[slot].in_use = true;

    *out_handle = (uint64_t)(slot + 1);
    *out_data = view;
    *out_size = file_size;
    return true;
}

void platform_file_unmap(uint64_t handle) {
    if (handle == 0) return;

    uint64_t idx = handle - 1;
    if (idx >= MMAP_HANDLE_CAP) return;

    MmapHandle *internal = &g_mmap_handles[idx];
    if (!internal->in_use) return;

    if (internal->view) {
        UnmapViewOfFile(internal->view);
    }
    if (internal->hMapping) {
        CloseHandle(internal->hMapping);
    }
    if (internal->hFile) {
        CloseHandle(internal->hFile);
    }
    internal->view = NULL;
    internal->hFile = NULL;
    internal->hMapping = NULL;
    internal->size = 0;
    internal->in_use = false;
}

#ifndef PLATFORM_SKIP_ENTRY
// Forward declaration for application entry point (only when platform provides entry)
int app_main();

// Initialize the platform and call the application
static int platform_init_and_run() {
    platform_init(0, NULL, NULL);
    int status = app_main();
    return status;
}

// Entry point for Windows - MSVC uses _start but we need to set it up correctly
void _start() {
    int status = platform_init_and_run();
    platform_exit(status);
}
#endif
