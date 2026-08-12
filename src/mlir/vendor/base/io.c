#include <base/io.h>
#include <base/types.h>
#include <base/exit.h>
#include <base/mem.h>
#include <base/scratch.h>
#include <platform/platform.h>
#include <base/numconv.h>

uint32_t write_all(int fd, ciovec_t* iovs, size_t iovs_len) {
    size_t i;
    size_t nwritten;
    uint32_t ret;

    for (i = 0; i < iovs_len; ) {
        ret = platform_fd_write(fd, &iovs[i], iovs_len - i, &nwritten);
        if (ret != 0) {
            return ret; // Return error code
        }

        // Advance through the iovecs based on how much was written
        while (nwritten > 0 && i < iovs_len) {
            if (nwritten >= iovs[i].buf_len) {
                nwritten -= iovs[i].buf_len;
                i++;
            } else {
                iovs[i].buf = (const uint8_t*)iovs[i].buf + nwritten;
                iovs[i].buf_len -= nwritten;
                nwritten = 0;
            }
        }
    }
    return 0; // Success
}

void writeln(int fd, const char* text) {
    const char *msg1 = text;
    const char *msg2 = "\n";

    ciovec_t iovs[2];
    iovs[0].buf = msg1;
    iovs[0].buf_len = base_strlen(msg1);
    iovs[1].buf = msg2;
    iovs[1].buf_len = base_strlen(msg2);

    write_all(fd, iovs, 2);
}

void writeln_int(int fd, const char* text, int n) {
    (void)fd; /* upstream writes to stderr regardless. Left alone deliberately. */
    const char *msg1 = text;
    const char *msg2 = " ";
    char p[32]; size_t p_len = int_to_str(n, p); p[p_len] = '\0';
    const char *msg3 = "\n";

    ciovec_t iovs[4];
    iovs[0].buf = msg1;
    iovs[0].buf_len = base_strlen(msg1);
    iovs[1].buf = msg2;
    iovs[1].buf_len = base_strlen(msg2);
    iovs[2].buf = p;
    iovs[2].buf_len = base_strlen(p);
    iovs[3].buf = msg3;
    iovs[3].buf_len = base_strlen(msg3);

    write_all(PLATFORM_STDERR_FD, iovs, 4);
}

void writeln_loc(int fd, const char *text, const char *file, unsigned int line, const char *function) {
    char line_str[32]; size_t p_len = int_to_str((int)line, line_str);
    line_str[p_len] = '\0';

    const char *msg[] = {file, ":", line_str, " in ",
        function, "(): ", text, "\n"};

    ciovec_t iovs[array_size(msg)];
    for (size_t i=0; i<array_size(msg); i++) {
        iovs[i].buf = msg[i];
        iovs[i].buf_len = base_strlen(msg[i]);
    }

    write_all(fd, iovs, array_size(msg));
}

// Returns the file contents as a null-terminated string in `text`.
// Returns `true` on success, otherwise `false`.
// The size of `text` includes the null character, which is inserted
// to allow tokenizing the text and use a null character as a "file end"
// condition.
bool read_file(Arena *arena, const string filename, string *text) {
    Scratch scratch = scratch_begin_avoid_conflict(arena);
    platform_fd_t fd = platform_path_open(str_to_cstr_copy(scratch.arena, filename),
            filename.size, PLATFORM_RIGHTS_READ, 0);
    if (fd < 0) {
        scratch_end(scratch);
        return false;
    }

    // Get file size by seeking to end
    uint64_t filesize_u64;
    if (platform_fd_seek(fd, 0, PLATFORM_SEEK_END, &filesize_u64) != 0) {
        platform_fd_close(fd);
        scratch_end(scratch);
        return false;
    }

    // Seek back to beginning
    uint64_t dummy;
    if (platform_fd_seek(fd, 0, PLATFORM_SEEK_SET, &dummy) != 0) {
        platform_fd_close(fd);
        scratch_end(scratch);
        return false;
    }

    size_t filesize = (size_t)filesize_u64;

    // Allocate buffer
    char *bytes = arena_new_array(arena, char, filesize+1);

    // Read file contents using iovec
    iovec_t iov = { .iov_base = bytes, .iov_len = filesize };
    size_t nread;
    int ret = platform_fd_read(fd, &iov, 1, &nread);
    platform_fd_close(fd);

    if (ret != 0 || nread != filesize) {
        scratch_end(scratch);
        return false;
    }
    bytes[nread] = '\0';
    text->str = bytes;
    text->size = filesize+1;
    scratch_end(scratch);
    return true;
}


string read_file_ok(Arena *arena, const string filename) {
    string text;
    if (read_file(arena, filename, &text)) {
        return text;
    } else {
        FATAL_ERROR("File cannot be opened.");
        return text;
    }
}

void println_explicit(string fmt, size_t arg_count, ...) {
    Scratch scratch = scratch_begin();
    va_list varg;
    va_start(varg, arg_count);

    string text = format_explicit_varg(scratch.arena, fmt, arg_count, varg);
    va_end(varg);
    text = str_concat(scratch.arena, text, str_lit("\n"));
    ciovec_t iov = {.buf = text.str, .buf_len = text.size};
    write_all(PLATFORM_STDOUT_FD, &iov, 1);

    scratch_end(scratch);
}
