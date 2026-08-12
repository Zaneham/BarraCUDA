#pragma once

#include <platform/platform.h>
#include <base/arena.h>
#include <base/string.h>
#include <base/format.h>

// The functions below this comment do not allocate memory (no arenas), so they
// are safe to use anywhere, including in arena / buddy allocator code, or in
// asserts.

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Writes all data from the iovecs to the specified file descriptor.
 *
 * This function repeatedly calls fd_write until all data is written or an error occurs.
 * It updates the iovecs to skip already-written data.
 *
 * @param fd The file descriptor to write to.
 * @param iovs Array of ciovec_t structures containing the data to write.
 * @param iovs_len Number of iovecs in the array.
 * @return 0 on success, or an error code if fd_write fails.
 */
uint32_t write_all(int fd, ciovec_t* iovs, size_t iovs_len);

// Prints a single line, appends `\n`
void writeln(int fd, const char* text);

// Prints: text + ' ' + int + '\n'
void writeln_int(int fd, const char* text, int n);

// Prints text with location information, appends '\n'
void writeln_loc(int fd, const char *text, const char *file, unsigned int line, const char *function);

#define PRINT_ERR(x) writeln_loc(PLATFORM_STDERR_FD, (x), __FILE__, __LINE__, __func__)

// The functions below allocate via arenas/scratch and depend on more of base/.

// Returns the file contents as a null-terminated string in `text`.
// Returns `true` on success, otherwise `false`.
bool read_file(Arena *arena, const string filename, string *text);
string read_file_ok(Arena *arena, const string filename);

void println_explicit(string fmt, size_t arg_count, ...);

#define println(fmt, ...) \
    println_explicit(fmt, COUNT_ARGS(__VA_ARGS__) __VA_OPT__(,) APPLY_WITH_COUNT(COUNT_ARGS(__VA_ARGS__) __VA_OPT__(,) __VA_ARGS__))

#define PRINT_LOG(x) println(str_lit("{}:{} in {}(): {}"), str_lit(__FILE__), __LINE__, str_lit(__func__), str_lit(x))
#ifdef __cplusplus
}
#endif
