#pragma once

#include <base/types.h>
#include <base/assert.h>
#include <base/arena.h>
#include <base/mem.h>

// --- Helper Macros (internal use) ---

#ifdef __cplusplus
extern "C" {
#endif

#define _GV_CONCAT_IMPL(a, b) a##b
#define _GV_CONCAT(a, b) _GV_CONCAT_IMPL(a, b)

#define _GV_CONCAT3_IMPL(a, b, c) a##b##c
#define _GV_CONCAT3(a, b, c) _GV_CONCAT3_IMPL(a, b, c)

// --- Conditional Compilation Helper for WITH_BASE_ASSERT ---
#if defined(WITH_BASE_ASSERT)
    #define IF_GENERIC_VECTOR_WITH_BASE_ASSERT(code) code
    // This constant is used for assertion checks.
    // Declared static const for internal linkage, avoiding multiple definition errors.
    static const int GV_INTERNAL_RESERVE_CALLED_MAGIC = 0xDEADBEEF;
#else
    #define IF_GENERIC_VECTOR_WITH_BASE_ASSERT(code)
#endif

// --- Main Macro to Define a Vector Type and its Functions ---
// TYPE: The data type to be stored (e.g., int, MyStructA).
// NAME: A prefix for the generated struct and function names (e.g., IntVec, MyStructAVec).
//       - Struct type will be: NAME
//       - Functions will be: NAME_init, NAME_reserve, NAME_push_back.
#define DEFINE_VECTOR_FOR_TYPE(TYPE, NAME) \
    \
    /* Vector Struct Definition */ \
    typedef struct NAME { \
        TYPE *data; \
        size_t size; \
        size_t max; \
        /* This field is conditionally compiled based on WITH_BASE_ASSERT */ \
        IF_GENERIC_VECTOR_WITH_BASE_ASSERT(int reserve_called_flag;) \
    } NAME; \
    \
    /* Reserves memory for at least 'new_max_capacity' elements. */ \
    /* Resets size to 0. Old data is not preserved. */ \
    static inline void _GV_CONCAT3(NAME, _, reserve)(Arena *arena, NAME *vec, size_t new_max_capacity) { \
        vec->size = 0; \
        if (new_max_capacity <= 0) new_max_capacity = 1; /* Minimum capacity of 1 */ \
        vec->data = arena_new_array(arena, TYPE, new_max_capacity); \
        vec->max = new_max_capacity; \
        IF_GENERIC_VECTOR_WITH_BASE_ASSERT(vec->reserve_called_flag = GV_INTERNAL_RESERVE_CALLED_MAGIC;) \
    } \
    \
    /* Adds an element to the end of the vector, resizing if necessary. */ \
    static inline void _GV_CONCAT3(NAME, _, push_back)(Arena *arena, NAME *vec, TYPE value) { \
        IF_GENERIC_VECTOR_WITH_BASE_ASSERT( \
            assert(vec->reserve_called_flag == GV_INTERNAL_RESERVE_CALLED_MAGIC && \
                   "Vector reserve() not called before push_back()."); \
        ) \
        if (vec->size == vec->max) { \
            size_t new_max_capacity = 2 * vec->max; \
            TYPE* new_data = arena_new_array(arena, TYPE, new_max_capacity); \
            base_memcpy(new_data, vec->data, sizeof(TYPE) * vec->size); \
            vec->data = new_data; \
            vec->max = new_max_capacity; \
        } \
        vec->data[vec->size] = value; \
        vec->size++; \
    }



DEFINE_VECTOR_FOR_TYPE(int64_t, vector_i64)
#ifdef __cplusplus
}
#endif
