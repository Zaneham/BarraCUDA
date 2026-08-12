#pragma once

#include <base/types.h>
#include <base/arena.h>

// Helper Macros (internal use)

#ifdef __cplusplus
extern "C" {
#endif

#define _GV_CONCAT_IMPL(a, b) a##b
#define _GV_CONCAT(a, b) _GV_CONCAT_IMPL(a, b)

#define _GV_CONCAT3_IMPL(a, b, c) a##b##c
#define _GV_CONCAT3(a, b, c) _GV_CONCAT3_IMPL(a, b, c)

// Hashtable Definition Macro with Inlined Hash and Equality
#define DEFINE_HASHTABLE_FOR_TYPES(KEY_TYPE, VALUE_TYPE, NAME) \
    typedef struct _GV_CONCAT3(NAME, _, Entry) { \
        KEY_TYPE key; \
        VALUE_TYPE value; \
        int occupied; \
    } _GV_CONCAT3(NAME, _, Entry); \
    \
    typedef struct NAME { \
        _GV_CONCAT3(NAME, _, Entry) *buckets; \
        size_t num_buckets; \
        size_t size; \
    } NAME; \
    \
    static inline void _GV_CONCAT3(NAME, _, init)(Arena *arena, NAME *ht, size_t initial_buckets) { \
        ht->num_buckets = initial_buckets; \
        ht->size = 0; \
        ht->buckets = arena_new_array(arena, _GV_CONCAT3(NAME, _, Entry), initial_buckets); \
        for (size_t i = 0; i < initial_buckets; i++) { \
            ht->buckets[i].occupied = 0; \
        } \
    } \
    \
    static inline void _GV_CONCAT3(NAME, _, insert)(Arena *arena, NAME *ht, KEY_TYPE key, VALUE_TYPE value) { \
        if ((double)ht->size >= 0.75 * (double)ht->num_buckets) { \
            size_t new_num_buckets = ht->num_buckets * 2; \
            _GV_CONCAT3(NAME, _, Entry) *new_buckets = arena_new_array(arena, _GV_CONCAT3(NAME, _, Entry), new_num_buckets); \
            for (size_t i = 0; i < new_num_buckets; i++) { \
                new_buckets[i].occupied = 0; \
            } \
            for (size_t i = 0; i < ht->num_buckets; i++) { \
                if (ht->buckets[i].occupied) { \
                    KEY_TYPE existing_key = ht->buckets[i].key; \
                    VALUE_TYPE existing_value = ht->buckets[i].value; \
                    size_t hash_value = _GV_CONCAT3(NAME, _, HASH)(existing_key); \
                    size_t index = hash_value % new_num_buckets; \
                    while (new_buckets[index].occupied) { \
                        index = (index + 1) % new_num_buckets; \
                    } \
                    new_buckets[index].key = existing_key; \
                    new_buckets[index].value = existing_value; \
                    new_buckets[index].occupied = 1; \
                } \
            } \
            ht->buckets = new_buckets; \
            ht->num_buckets = new_num_buckets; \
        } \
        size_t hash_value = _GV_CONCAT3(NAME, _, HASH)(key); \
        size_t index = hash_value % ht->num_buckets; \
        while (ht->buckets[index].occupied) { \
            if (_GV_CONCAT3(NAME, _, EQUAL)(ht->buckets[index].key, key)) { \
                ht->buckets[index].value = value; \
                return; \
            } \
            index = (index + 1) % ht->num_buckets; \
        } \
        ht->buckets[index].key = key; \
        ht->buckets[index].value = value; \
        ht->buckets[index].occupied = 1; \
        ht->size++; \
    } \
    \
    static inline VALUE_TYPE* _GV_CONCAT3(NAME, _, get)(NAME *ht, KEY_TYPE key) { \
        size_t hash_value = _GV_CONCAT3(NAME, _, HASH)(key); \
        size_t index = hash_value % ht->num_buckets; \
        size_t start_index = index; \
        while (ht->buckets[index].occupied) { \
            if (_GV_CONCAT3(NAME, _, EQUAL)(ht->buckets[index].key, key)) { \
                return &ht->buckets[index].value; \
            } \
            index = (index + 1) % ht->num_buckets; \
            if (index == start_index) break; \
        } \
        return NULL; \
    }
#ifdef __cplusplus
}
#endif
