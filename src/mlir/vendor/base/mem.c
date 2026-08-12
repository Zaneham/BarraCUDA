#include <base/mem.h>

size_t base_strlen(const char* str) {
    const char* s;
    for (s = str; *s; ++s);
    return (size_t)(s - str);
}

char* base_strcpy(char* dest, const char* src) {
    char* d = dest;
    while ((*d++ = *src++) != '\0');
    return dest;
}

int base_strcmp(const char* s1, const char* s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}

void* base_memcpy(void* dest, const void* src, size_t n) {
    unsigned char* d = (unsigned char*)dest;
    const unsigned char* s = (const unsigned char*)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    return dest;
}

void* base_memmove(void* dest, const void* src, size_t n) {
    unsigned char* d = (unsigned char*)dest;
    const unsigned char* s = (const unsigned char*)src;

    if (d == s || n == 0) {
        return dest;
    }

    // Copy forward when regions do not overlap or destination is before source
    if (d < s || d >= s + n) {
        for (size_t i = 0; i < n; i++) {
            d[i] = s[i];
        }
        return dest;
    }

    // Copy backward to handle overlapping regions safely
    for (size_t i = n; i != 0; i--) {
        d[i - 1] = s[i - 1];
    }
    return dest;
}

int base_memcmp(const void* s1, const void* s2, size_t n) {
    const unsigned char* p1 = (const unsigned char*)s1;
    const unsigned char* p2 = (const unsigned char*)s2;
    for (size_t i = 0; i < n; i++) {
        if (p1[i] != p2[i]) {
            return p1[i] - p2[i];
        }
    }
    return 0;
}

void* base_memset(void* s, int c, size_t n) {
    unsigned char* p = (unsigned char*)s;
    for (size_t i = 0; i < n; i++) {
        p[i] = (unsigned char)c;
    }
    return s;
}

void* base_memchr(const void* s, int c, size_t n) {
    const unsigned char* p = (const unsigned char*)s;
    for (size_t i = 0; i < n; i++) {
        if (p[i] == (unsigned char)c) {
            return (void*)(p + i);
        }
    }
    return (void*)0;
}

char* base_strchr(const char* s, int c) {
    while (*s) {
        if (*s == (char)c) {
            return (char*)s;
        }
        s++;
    }
    // Check for null terminator match
    if ((char)c == '\0') {
        return (char*)s;
    }
    return (void*)0;
}

char* base_strrchr(const char* s, int c) {
    const char* last = (void*)0;
    while (*s) {
        if (*s == (char)c) {
            last = s;
        }
        s++;
    }
    // Check for null terminator match
    if ((char)c == '\0') {
        return (char*)s;
    }
    return (char*)last;
}

char* base_strncpy(char* dest, const char* src, size_t n) {
    size_t i;
    for (i = 0; i < n && src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    // Pad with null bytes if src is shorter than n
    for (; i < n; i++) {
        dest[i] = '\0';
    }
    return dest;
}

size_t base_strcspn(const char* s, const char* reject) {
    const char* p;
    const char* r;
    size_t count = 0;

    for (p = s; *p != '\0'; p++) {
        for (r = reject; *r != '\0'; r++) {
            if (*p == *r) {
                return count;
            }
        }
        count++;
    }
    return count;
}

int base_strncmp(const char* s1, const char* s2, size_t n) {
    if (n == 0) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        if (s1[i] != s2[i]) {
            return (unsigned char)s1[i] - (unsigned char)s2[i];
        }
        if (s1[i] == '\0') {
            return 0;
        }
    }
    return 0;
}

char* base_strstr(const char* haystack, const char* needle) {
    if (*needle == '\0') {
        return (char*)haystack;
    }

    for (; *haystack != '\0'; haystack++) {
        const char* h = haystack;
        const char* n = needle;

        while (*h != '\0' && *n != '\0' && *h == *n) {
            h++;
            n++;
        }

        if (*n == '\0') {
            return (char*)haystack;
        }
    }

    return (void*)0;
}
