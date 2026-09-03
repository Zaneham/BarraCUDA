    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
    .text

    .globl mf16
    .p2align 8
    .type mf16,@function
mf16:
    ; 1 SGPRs, 248 VGPRs, 0 LDS bytes, 0 scratch bytes
.LBB0:
    s_load_dwordx2 s[4:5], s[0:1], 0
    s_load_dwordx2 s[6:7], s[0:1], 8
    s_load_dwordx2 s[8:9], s[0:1], 16
    v_mov_b32 v6, 0
    v_mov_b32 v5, 0
    v_mov_b32 v0, 0
    v_add_u32 v4, 4, v5
    v_add_u32 v3, 4, v0
    s_waitcnt lgkmcnt(0)
    global_load_dword v224, v6, s[8:9]
    v_add_u32 v2, 4, v6
    global_load_dword v240, v0, s[4:5]
    v_add_u32 v1, 8, v6
    global_load_dword v244, v5, s[6:7]
    v_add_u32 v0, 12, v6
    global_load_dword v245, v4, s[6:7]
    global_load_dword v227, v0, s[8:9]
    global_load_dword v241, v3, s[4:5]
    global_load_dword v225, v2, s[8:9]
    global_load_dword v226, v1, s[8:9]
    s_waitcnt vmcnt(0)
    v_mfma_f32_16x16x16f16 v[208:211], v[240:241], v[244:245], v[224:227]
    v_add_u32 v1, 4, v6
    global_store_dword v6, v208, s[8:9]
    v_add_u32 v0, 8, v6
    global_store_dword v1, v209, s[8:9]
    v_add_u32 v1, 12, v6
    global_store_dword v0, v210, s[8:9]
    global_store_dword v1, v211, s[8:9]
    s_endpgm

