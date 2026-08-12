#include <base/numconv.h>
#include <base/stdarg.h>

size_t uint64_to_str(uint64_t val, char* buf) {
    if (val == 0) {
        buf[0] = '0';
        return 1;
    }

    size_t len = 0;
    // Convert digits in reverse order
    while (val > 0) {
        buf[len++] = (char)('0' + (val % 10));
        val /= 10;
    }
    // Reverse
    for (size_t i = 0; i < len / 2; i++) {
        char t = buf[i];
        buf[i] = buf[len - 1 - i];
        buf[len - 1 - i] = t;
    }
    return len;
}

size_t int64_to_str(int64_t val, char* buf) {
    if (val < 0) {
        buf[0] = '-';
        size_t len = uint64_to_str((uint64_t)(-val), buf + 1);
        return len + 1;
    } else {
        return uint64_to_str((uint64_t)val, buf);
    }
}

size_t int_to_str(int val, char* buf) {
    return int64_to_str((int64_t)val, buf);
}

size_t double_to_str(double val, char* buf, int precision) {
    // Simple implementation: handle sign, integer part, decimal point, fractional part
    size_t pos = 0;

    if (val < 0) {
        buf[pos++] = '-';
        val = -val;
    }

    // Integer part
    int64_t int_part = (int64_t)val;
    pos += int64_to_str(int_part, buf + pos);

    // Default precision is 6 if not specified
    if (precision < 0) precision = 6;

    if (precision > 0) {
        buf[pos++] = '.';

        // Fractional part
        double frac_part = val - (double)int_part;
        for (int i = 0; i < precision; i++) {
            frac_part *= 10;
            int digit = (int)frac_part;
            buf[pos++] = (char)('0' + digit);
            frac_part -= digit;
        }
    }

    return pos;
}

static size_t double_to_str_g(double val, char* buf, int precision) {
    if (precision < 0) precision = 6;
    size_t len = double_to_str(val, buf, precision);
    while (len > 0 && buf[len - 1] == '0') len--;
    if (len > 0 && buf[len - 1] == '.') len--;
    if (len == 0) buf[len++] = '0';
    return len;
}

// Format `val` as `[-]d.ddde[+-]NN` (C's "%e"), with `precision` fractional
// digits in the mantissa. Always at least two exponent digits.
size_t double_to_str_e(double val, char* buf, int precision) {
    size_t pos = 0;

    int neg = 0;
    if (val < 0) { neg = 1; val = -val; }
    if (precision < 0) precision = 6;
    // Clamp precision so the integer mantissa fits in int64_t.
    if (precision > 17) precision = 17;

    // Normalize to [1, 10) and compute the decimal exponent.
    int exp = 0;
    if (val != 0.0) {
        while (val >= 10.0) { val /= 10.0; exp++; }
        while (val < 1.0)   { val *= 10.0; exp--; }
    }

    // Compute the integer mantissa with `precision+1` digits, rounded.
    double scale = 1.0;
    for (int i = 0; i < precision; i++) scale *= 10.0;
    int64_t mantissa = (int64_t)(val * scale + 0.5);
    int64_t cutoff = 10;
    for (int i = 0; i < precision; i++) cutoff *= 10;
    if (mantissa >= cutoff) { mantissa /= 10; exp++; }

    if (neg) buf[pos++] = '-';

    // First digit (integer part).
    int64_t scale_i = (int64_t)scale;
    int first = (int)(mantissa / scale_i);
    int64_t frac = mantissa % scale_i;
    if (first > 9) first = 9;
    buf[pos++] = (char)('0' + first);

    if (precision > 0) {
        buf[pos++] = '.';
        char digbuf[32];
        int n = 0;
        int64_t r = frac;
        while (r > 0) { digbuf[n++] = (char)('0' + (r % 10)); r /= 10; }
        while (n < precision) digbuf[n++] = '0';
        while (n > 0) buf[pos++] = digbuf[--n];
    }

    buf[pos++] = 'e';
    if (exp < 0) { buf[pos++] = '-'; exp = -exp; }
    else         { buf[pos++] = '+'; }

    char digits[8];
    int n = 0;
    if (exp == 0) digits[n++] = '0';
    while (exp > 0) { digits[n++] = (char)('0' + (exp % 10)); exp /= 10; }
    while (n < 2) digits[n++] = '0';
    while (n > 0) buf[pos++] = digits[--n];

    return pos;
}

size_t uint64_to_hex_str(uint64_t val, char* buf, int uppercase) {
    if (val == 0) {
        buf[0] = '0';
        return 1;
    }

    const char* digits = uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
    size_t len = 0;
    
    // Convert digits in reverse order
    while (val > 0) {
        buf[len++] = digits[val & 0xF];
        val >>= 4;
    }
    
    // Reverse
    for (size_t i = 0; i < len / 2; i++) {
        char t = buf[i];
        buf[i] = buf[len - 1 - i];
        buf[len - 1 - i] = t;
    }
    return len;
}

// vsnprintf/snprintf implementations (only for nostdlib builds)
int base_vsnprintf(char *str, size_t size, const char *format, va_list args) {
    if (size == 0) return 0;

    size_t pos = 0;
    const char* p = format;
    char temp_buf[32];

    while (*p && pos < size - 1) {
        if (*p == '%' && *(p + 1)) {
            p++;

            // Check for length modifiers: l, ll, z
            int is_long = 0;
            int is_long_long = 0;
            int is_size_t = 0;
            
            if (*p == 'l') {
                p++;
                is_long = 1;
                if (*p == 'l') {
                    p++;
                    is_long_long = 1;
                    is_long = 0;
                }
            } else if (*p == 'z') {
                p++;
                is_size_t = 1;
            }

            // Check for precision specifier (e.g., %.2f, %.*s)
            int precision = -1;
            if (*p == '.') {
                p++;
                if (*p == '*') {
                    // %.*X form: precision read from a preceding int arg.
                    precision = va_arg(args, int);
                    if (precision < 0) precision = 0;
                    p++;
                } else {
                    precision = 0;
                    while (*p >= '0' && *p <= '9') {
                        precision = precision * 10 + (*p - '0');
                        p++;
                    }
                }
            }

            switch (*p) {
                case 'd':
                case 'i': {
                    if (is_long_long) {
                        int64_t val = va_arg(args, int64_t);
                        size_t len = int64_to_str(val, temp_buf);
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    } else if (is_long) {
                        long val = va_arg(args, long);
                        size_t len = int64_to_str((int64_t)val, temp_buf);
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    } else {
                        int val = va_arg(args, int);
                        size_t len = int_to_str(val, temp_buf);
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    }
                    break;
                }
                case 'u': {
                    if (is_long_long) {
                        uint64_t val = va_arg(args, uint64_t);
                        size_t len = uint64_to_str(val, temp_buf);
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    } else if (is_long || is_size_t) {
                        unsigned long val = va_arg(args, unsigned long);
                        size_t len = uint64_to_str((uint64_t)val, temp_buf);
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    } else {
                        unsigned int val = va_arg(args, unsigned int);
                        size_t len = uint64_to_str((uint64_t)val, temp_buf);
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    }
                    break;
                }
                case 'x':
                case 'X': {
                    if (is_long_long) {
                        uint64_t val = va_arg(args, uint64_t);
                        size_t len = uint64_to_hex_str(val, temp_buf, *p == 'X');
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    } else if (is_long || is_size_t) {
                        unsigned long val = va_arg(args, unsigned long);
                        size_t len = uint64_to_hex_str((uint64_t)val, temp_buf, *p == 'X');
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    } else {
                        unsigned int val = va_arg(args, unsigned int);
                        size_t len = uint64_to_hex_str(val, temp_buf, *p == 'X');
                        size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                        for (size_t i = 0; i < copy_len; i++) {
                            str[pos++] = temp_buf[i];
                        }
                    }
                    break;
                }
                case 'p': {
                    void* ptr = va_arg(args, void*);
                    if (pos < size - 2) {
                        str[pos++] = '0';
                        str[pos++] = 'x';
                    }
                    size_t len = uint64_to_hex_str((uint64_t)(uintptr_t)ptr, temp_buf, 0);
                    size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                    for (size_t i = 0; i < copy_len; i++) {
                        str[pos++] = temp_buf[i];
                    }
                    break;
                }
                case 'f': {
                    double val = va_arg(args, double);
                    if (precision < 0) precision = 6;
                    size_t len = double_to_str(val, temp_buf, precision);
                    size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                    for (size_t i = 0; i < copy_len; i++) {
                        str[pos++] = temp_buf[i];
                    }
                    break;
                }
                case 'g': {
                    double val = va_arg(args, double);
                    size_t len = double_to_str_g(val, temp_buf, precision);
                    size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                    for (size_t i = 0; i < copy_len; i++) {
                        str[pos++] = temp_buf[i];
                    }
                    break;
                }
                case 'e': {
                    double val = va_arg(args, double);
                    if (precision < 0) precision = 6;
                    size_t len = double_to_str_e(val, temp_buf, precision);
                    size_t copy_len = (pos + len < size - 1) ? len : (size - 1 - pos);
                    for (size_t i = 0; i < copy_len; i++) {
                        str[pos++] = temp_buf[i];
                    }
                    break;
                }
                case 's': {
                    const char* s = va_arg(args, char*);
                    if (s == NULL) s = "(null)";
                    if (precision >= 0) {
                        // %.*s / %.Ns: emit at most `precision` chars (do not
                        // require NUL termination — needed for slice-style
                        // strings like the corec `string` { ptr, size } view).
                        int n = 0;
                        while (n < precision && pos < size - 1) {
                            str[pos++] = s[n++];
                        }
                    } else {
                        while (*s && pos < size - 1) {
                            str[pos++] = *s++;
                        }
                    }
                    break;
                }
                case 'c': {
                    char c = (char)va_arg(args, int);
                    if (pos < size - 1) {
                        str[pos++] = c;
                    }
                    break;
                }
                case '%': {
                    str[pos++] = '%';
                    break;
                }
                default:
                    // Unknown format specifier, just skip it
                    break;
            }
            p++;
        } else {
            str[pos++] = *p++;
        }
    }

    str[pos] = '\0';
    return (int)pos;
}

// Simple snprintf implementation for base/
// Supports: %d, %u, %f, %.Nf, %s
int base_snprintf(char *str, size_t size, const char *format, ...) {
    va_list args;
    va_start(args, format);
    int result = base_vsnprintf(str, size, format, args);
    va_end(args);
    return result;
}
