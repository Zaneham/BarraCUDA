#include <base/types.h>
#include <base/string.h>
#include <base/mem.h>
#include <base/numconv.h>

string str_from_cstr_view(char *cstr) {
    return (string){cstr, base_strlen(cstr)};
}

string str_from_cstr_len_view(char *cstr, uint64_t size) {
    return (string){cstr, size};
}

string str_from_cstr_len_view_const(const char *cstr, uint64_t size) {
    return (string){(char*)cstr, size};
}

char *str_to_cstr_copy(Arena *arena, string str) {
    char *cstr = arena_new_array(arena, char, str.size+1);
    base_memcpy(cstr, str.str, str.size);
    cstr[str.size] = '\0';
    return cstr;
}

bool str_eq(string a, string b) {
    if (a.size == b.size) {
        return (base_memcmp(a.str, b.str, a.size) == 0);
    } else {
        return false;
    }
}

string str_substr(string str, uint64_t min, uint64_t size) {
    return (string){str.str+min, size};
}

string int_to_string(Arena *arena, int64_t value) {
    char buf[32];
    size_t len = int64_to_str(value, buf);
    char *str = arena_new_array(arena, char, len);
    base_memcpy(str, buf, len);
    return (string){str, len};
}

string uint_to_string(Arena *arena, uint64_t value) {
    char buf[32];
    size_t len = uint64_to_str(value, buf);
    char *str = arena_new_array(arena, char, len);
    base_memcpy(str, buf, len);
    return (string){str, len};
}

string double_to_string(Arena *arena, double value, int precision) {
    char buf[32];
    size_t len = double_to_str(value, buf, precision);
    char *str = arena_new_array(arena, char, len);
    base_memcpy(str, buf, len);
    return (string){str, len};
}

string char_to_string(Arena *arena, char c) {
    char *buf = arena_new_array(arena, char, 1);
    *buf = c;
    return (string){buf, 1};
}

string str_concat(Arena *arena, string a, string b) {
    if (b.size == 0) return a;
    if (a.size == 0) return str_copy(arena, b);
    size_t total = a.size + b.size;
    char *str = arena_new_array(arena, char, total);
    base_memcpy(str, a.str, a.size);
    base_memcpy(str + a.size, b.str, b.size);
    return (string){str, total};
}

string str_copy(Arena *arena, string a) {
    char *str = NULL;
    if (a.size > 0) {
        str = arena_new_array(arena, char, a.size);
        base_memcpy(str, a.str, a.size);
    }
    return (string){str, a.size};
}

uint32_t str_hash(string str) {
    // FNV-1a hash
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < str.size; i++) {
        hash ^= (uint32_t)(unsigned char)str.str[i];
        hash *= 16777619u;
    }
    return hash;
}
