#include <base/types.h>
#include <base/stdarg.h>
#include <base/mem.h>
#include <base/format.h>
#include <base/scratch.h>
#include <base/strbuf.h>
#include <base/exit.h>
#include <base/io.h>

// Inline implementation of isdigit
static inline int fmt_isdigit(int c) {
    return c >= '0' && c <= '9';
}


typedef struct {
    char alignment;  // '<', '>', '^', or '\0'
    int width;       // -1 if not specified
    int precision;   // -1 if not specified
} FormatSpec;

// Parse format specifier
static FormatSpec parse_format_spec(string spec) {
    FormatSpec fs = {.alignment = '\0', .width = -1, .precision = -1};
    const char *p = spec.str;
    const char *end = spec.str + spec.size;
    if (p < end) {
        if (*p == '<' || *p == '>' || *p == '^') {
            fs.alignment = *p++;
        }
    }
    if (p < end && fmt_isdigit(*p)) {
        fs.width = 0;
        while (p < end && fmt_isdigit(*p)) {
            fs.width = fs.width * 10 + (*p++ - '0');
        }
    }
    if (p < end && *p == '.') {
        p++;
        if (p < end && fmt_isdigit(*p)) {
            fs.precision = 0;
            while (p < end && fmt_isdigit(*p)) {
                fs.precision = fs.precision * 10 + (*p++ - '0');
            }
        }
    }
    return fs;
}

// Core formatting function with variadic arguments
string format_explicit_varg(Arena *arena, string fmt, size_t arg_count,
        va_list ap) {
    Scratch scratch = scratch_begin_avoid_conflict(arena);
    strbuf result = strbuf_make_cap(scratch.arena, fmt.size + 16);
    const char *p = fmt.str;
    const char *end = fmt.str + fmt.size;
    size_t arg_index = 0;
    while (p < end) {
        const char *open_brace = base_memchr(p, '{', (size_t)(end - p));
        if (open_brace == NULL) {
            string remaining = {.str = (char*)p, .size = (uint64_t)(end - p)};
            strbuf_append(scratch.arena, &result, remaining);
            break;
        }
        if (open_brace > p) {
            string part = {.str = (char*)p, .size = (uint64_t)(open_brace - p)};
            strbuf_append(scratch.arena, &result, part);
        }
        p = open_brace + 1;
        if (p >= end) {
            strbuf_append_char(scratch.arena, &result, '{');
            break;
        }
        if (*p == '{') {
            strbuf_append_char(scratch.arena, &result, '{');
            p++;
            continue;
        }
        const char *close_brace = base_memchr(p, '}', (size_t)(end - p));
        if (close_brace == NULL) {
            string error = str_lit("Error: missing closing brace");
            strbuf_append(scratch.arena, &result, error);
            break;
        }
        const char *colon = base_memchr(p, ':', (size_t)(close_brace - p));
        FormatSpec spec;
        if (colon) {
            string spec_str = {.str = (char*)colon + 1, .size = (uint64_t)(close_brace - (colon + 1))};
            spec = parse_format_spec(spec_str);
        } else {
            if (p != close_brace) {
                string error = str_lit("Error: invalid format specifier");
                strbuf_append(scratch.arena, &result, error);
                p = close_brace + 1;
                continue;
            }
            spec = (FormatSpec){.alignment = '\0', .width = -1, .precision = -1};
        }
        if (arg_index >= arg_count) {
            FATAL_ERROR("Missing argument");
        }
        ArgType type = (ArgType)va_arg(ap, int);
        string s;
        switch (type) {
            case ARG_INT8: {
                int8_t value = (int8_t)va_arg(ap, int);
                s = int_to_string(scratch.arena, value);
                break;
            }
            case ARG_UINT8: {
                uint8_t value = (uint8_t)va_arg(ap, int);
                s = uint_to_string(scratch.arena, value);
                break;
            }
            case ARG_INT16: {
                int16_t value = (int16_t)va_arg(ap, int);
                s = int_to_string(scratch.arena, value);
                break;
            }
            case ARG_UINT16: {
                uint16_t value = (uint16_t)va_arg(ap, int);
                s = uint_to_string(scratch.arena, value);
                break;
            }
            case ARG_INT32: {
                int32_t value = va_arg(ap, int32_t);
                s = int_to_string(scratch.arena, value);
                break;
            }
            case ARG_UINT32: {
                uint32_t value = va_arg(ap, uint32_t);
                s = uint_to_string(scratch.arena, value);
                break;
            }
            case ARG_INT64: {
                int64_t value = va_arg(ap, int64_t);
                s = int_to_string(scratch.arena, value);
                break;
            }
            case ARG_UINT64: {
                uint64_t value = va_arg(ap, uint64_t);
                s = uint_to_string(scratch.arena, value);
                break;
            }
            case ARG_DOUBLE: {
                double value = va_arg(ap, double);
                s = double_to_string(scratch.arena, value, spec.precision);
                break;
            }
            case ARG_STRING: {
                char* value = va_arg(ap, char*);
                s = str_from_cstr_view(value);
                if (spec.precision >= 0 && (uint64_t)spec.precision < s.size) {
                    s.size = (uint64_t)spec.precision;
                }
                break;
            }
            case ARG_STRING2: {
#if defined(_WIN64)
                // On Windows x64, structs > 8 bytes are passed by reference in varargs
                string value = *va_arg(ap, string*);
#else
                string value = va_arg(ap, string);
#endif
                s = value;
                if (spec.precision >= 0 && (uint64_t)spec.precision < s.size) {
                    s.size = (uint64_t)spec.precision;
                }
                break;
            }
            case ARG_POINTER: {
                void* value = va_arg(ap, void*);
                s = uint_to_string(scratch.arena, (uint64_t)value);
                break;
            }
            case ARG_VECTOR_INT64: {
#if defined(_WIN64)
                // On Windows x64, structs > 8 bytes are passed by reference in varargs
                vector_i64 value = *va_arg(ap, vector_i64*);
#else
                vector_i64 value = va_arg(ap, vector_i64);
#endif
                strbuf vec_buf = strbuf_make_cap(scratch.arena, 32);
                strbuf_append_char(scratch.arena, &vec_buf, '{');
                for (size_t i=0; i<value.size; i++) {
                    strbuf_append(scratch.arena, &vec_buf,
                            int_to_string(scratch.arena, value.data[i]));
                    if (i + 1 < value.size) {
                        strbuf_append(scratch.arena, &vec_buf, str_lit(", "));
                    }
                }
                strbuf_append_char(scratch.arena, &vec_buf, '}');
                s = strbuf_to_string(vec_buf);
                if (spec.precision >= 0 && (uint64_t)spec.precision < s.size) {
                    s.size = (uint64_t)spec.precision;
                }
                break;
            }
            default:
                s = str_lit("Unknown type");
        }
        arg_index++;
        // Apply width and alignment
        if (spec.alignment == '\0') {
            // Right-align numeric types, left-align everything else
            if (type == ARG_INT8 || type == ARG_UINT8 ||
                type == ARG_INT16 || type == ARG_UINT16 ||
                type == ARG_INT32 || type == ARG_UINT32 ||
                type == ARG_INT64 || type == ARG_UINT64 ||
                type == ARG_DOUBLE) {
                spec.alignment = '>';
            } else {
                spec.alignment = '<';
            }
        }
        if (spec.width > 0 && s.size < (uint64_t)spec.width) {
            size_t pad_size = (size_t)spec.width - s.size;
            char pad_char = ' ';
            if (spec.alignment == '<') {
                strbuf_append(scratch.arena, &result, s);
                for (size_t k = 0; k < pad_size; k++) {
                    strbuf_append_char(scratch.arena, &result, pad_char);
                }
            } else if (spec.alignment == '^') {
                size_t left_pad = pad_size / 2;
                size_t right_pad = pad_size - left_pad;
                for (size_t k = 0; k < left_pad; k++) {
                    strbuf_append_char(scratch.arena, &result, pad_char);
                }
                strbuf_append(scratch.arena, &result, s);
                for (size_t k = 0; k < right_pad; k++) {
                    strbuf_append_char(scratch.arena, &result, pad_char);
                }
            } else {  // '>' or default
                for (size_t k = 0; k < pad_size; k++) {
                    strbuf_append_char(scratch.arena, &result, pad_char);
                }
                strbuf_append(scratch.arena, &result, s);
            }
        } else {
            strbuf_append(scratch.arena, &result, s);
        }
        p = close_brace + 1;
    }
    if (arg_index != arg_count) {
        FATAL_ERROR("Arguments do not match the format string");
    }

    // Copy final result to the supplied arena
    string final_result = str_copy(arena, strbuf_to_string(result));
    scratch_end(scratch);
    return final_result;
}

string format_explicit(Arena *arena, string fmt, size_t arg_count, ...) {
    va_list ap;
    va_start(ap, arg_count);
    string result = format_explicit_varg(arena, fmt, arg_count, ap);
    va_end(ap);
    return result;
}
