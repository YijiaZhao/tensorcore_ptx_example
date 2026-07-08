/**
 * verify_random.h — 随机数据 + CPU 参考 的可复用正确性验证工具
 *
 * 用法(见同目录三个 .cu):
 *   1. fill_random() 生成随机 A/B: 编码写进 GPU 要吃的 buffer, float 值写进 ref 数组
 *   2. GPU 跑 tensor core 指令
 *   3. cpu_gemm_ref() 算参考 D[m][n] = Σ_k A[m][k]*B[n][k]
 *   4. check() 逐元素精确比对 (==, 不是容差)
 *
 * 为什么可以用 == :
 *   取值集合刻意限制在 {0, ±0.5, ±1, ±1.5, ±2} —— 两两乘积 ∈ ±[0.25,4],
 *   K≤64 时 |Σ| ≤ 256 且都是 0.25 的整数倍, 在 fp32 里精确可表,
 *   任何累加顺序结果完全一致 → CPU float == tensor core fp32 累加, 逐bit相等。
 *   (全1.0测试查不出的布局/映射/转置错误, 随机数据全能现形)
 *
 * 布局约定(与仓库所有示例一致):
 *   A: M行 × K列, K-major(行内连续), 每行恒 32 字节 (K×ebits=256bit)
 *   B: N行 × K列, 同上 (即 mma 的 col-major B / desc 的 K-major)
 *   4-bit 低半字节在前, 多字节小端。
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

struct EncVal { uint32_t enc; float val; };

// ---- 各格式 {0, ±0.5, ±1, ±1.5, ±2} 的编码表 ----
static const EncVal FP16_SET[] = {
    {0x0000,0.f},{0x3800,0.5f},{0x3C00,1.f},{0x3E00,1.5f},{0x4000,2.f},
    {0xB800,-0.5f},{0xBC00,-1.f},{0xBE00,-1.5f},{0xC000,-2.f}};
static const EncVal BF16_SET[] = {
    {0x0000,0.f},{0x3F00,0.5f},{0x3F80,1.f},{0x3FC0,1.5f},{0x4000,2.f},
    {0xBF00,-0.5f},{0xBF80,-1.f},{0xBFC0,-1.5f},{0xC000,-2.f}};
static const EncVal TF32_SET[] = {
    {0x00000000,0.f},{0x3F000000,0.5f},{0x3F800000,1.f},{0x3FC00000,1.5f},{0x40000000,2.f},
    {0xBF000000,-0.5f},{0xBF800000,-1.f},{0xBFC00000,-1.5f},{0xC0000000,-2.f}};
static const EncVal E4M3_SET[] = {
    {0x00,0.f},{0x30,0.5f},{0x38,1.f},{0x3C,1.5f},{0x40,2.f},
    {0xB0,-0.5f},{0xB8,-1.f},{0xBC,-1.5f},{0xC0,-2.f}};
static const EncVal E2M1_SET[] = {   // fp4: 0=0x0 0.5=0x1 1.0=0x2 1.5=0x3 2.0=0x4, 负数|0x8
    {0x0,0.f},{0x1,0.5f},{0x2,1.f},{0x3,1.5f},{0x4,2.f},
    {0x9,-0.5f},{0xA,-1.f},{0xB,-1.5f},{0xC,-2.f}};
static const EncVal S8_SET[] = {
    {0x00,0.f},{0x01,1.f},{0x02,2.f},{0xFF,-1.f},{0xFE,-2.f}};
static const EncVal S4_SET[] = {
    {0x0,0.f},{0x1,1.f},{0x2,2.f},{0xF,-1.f},{0xE,-2.f}};
#define SETN(s) (int)(sizeof(s)/sizeof(EncVal))

// ---- 确定性随机(可传seed复跑) ----
static inline uint32_t xorshift(uint32_t* s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}

// 把编码写到 buf 的第 idx 个元素位置 (ebits ∈ {4,8,16,32})
static inline void pack_set(uint8_t* buf, int idx, uint32_t enc, int ebits) {
    if (ebits == 4) {
        int b = idx / 2;
        if (idx % 2 == 0) buf[b] = (buf[b] & 0xF0) | (enc & 0xF);
        else              buf[b] = (buf[b] & 0x0F) | ((enc & 0xF) << 4);
    } else if (ebits == 8)  { buf[idx] = (uint8_t)enc; }
    else if (ebits == 16)   { ((uint16_t*)buf)[idx] = (uint16_t)enc; }
    else                    { ((uint32_t*)buf)[idx] = enc; }
}

// 随机填充 rows×K: 编码 → enc_buf(行连续), float值 → ref[row*K+k]
static inline void fill_random(uint8_t* enc_buf, float* ref, int rows, int K,
                               const EncVal* set, int setn, int ebits, uint32_t* seed) {
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < K; k++) {
            const EncVal& e = set[xorshift(seed) % setn];
            pack_set(enc_buf, r * K + k, e.enc, ebits);
            ref[r * K + k] = e.val;
        }
}

// 指定位置写一个精确值(one-hot probe用), 值必须在set里
static inline void set_elem(uint8_t* enc_buf, float* ref, int K, int row, int k,
                            float v, const EncVal* set, int setn, int ebits) {
    for (int i = 0; i < setn; i++)
        if (set[i].val == v) {
            pack_set(enc_buf, row * K + k, set[i].enc, ebits);
            ref[row * K + k] = v; return;
        }
    fprintf(stderr, "set_elem: %f not in value set\n", v); exit(1);
}

// CPU 参考: D[m*N+n] = Σ_k A[m*K+k] * B[n*K+k]
static inline void cpu_gemm_ref(const float* A, const float* B, float* D,
                                int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += A[m * K + k] * B[n * K + k];
            D[m * N + n] = acc;
        }
}

// 逐元素精确比对
static inline int check_exact(const float* got, const float* want, int n, const char* tag) {
    int bad = 0;
    for (int i = 0; i < n; i++)
        if (got[i] != want[i]) {
            if (bad < 4) printf("  [%s] MISMATCH @%d: got %g want %g\n", tag, i, got[i], want[i]);
            bad++;
        }
    printf("  %-10s %s (%d/%d exact)\n", tag, bad ? "FAIL" : "PASS", n - bad, n);
    return bad;
}

// ============================================================
// 供各示例 .cu 内嵌「阶段2: 随机 vs CPU」使用的公共件
// ============================================================

// ---- mma.sync m16n8kX fragment 下标 (PTX row.col 布局, 行恒32B=8×u32, 全形状统一) ----
#ifdef __CUDACC__
__device__ __forceinline__ int mma_a_idx(int lane, int r) {   // A[16][8] 的 u32 下标
    return (lane / 4 + (r & 1) * 8) * 8 + lane % 4 + (r >> 1) * 4;
}
__device__ __forceinline__ int mma_b_idx(int lane, int r) {   // B[8][8] 的 u32 下标
    return (lane / 4) * 8 + lane % 4 + r * 4;
}
__device__ __forceinline__ int mma_d_idx(int lane, int r) {   // D[16][8] 的 float 下标
    return (lane / 4 + (r >> 1) * 8) * 8 + (lane % 4) * 2 + (r & 1);
}
#endif

// ---- block scale 的 scale 编码表 ----
// 范围由精确性约束推出: 各项数量级差 × 项数 必须落在 fp32 24bit 尾数预算内,
// 否则大数吃小数、累加顺序敏感, == 比对失效。
// ue8m0 纯2的幂(乘法零舍入): 指数 ±3 → 7档
static const EncVal UE8M0_SET[] = {
    {0x7C,0.125f},{0x7D,0.25f},{0x7E,0.5f},{0x7F,1.f},{0x80,2.f},{0x81,4.f},{0x82,8.f}};
// ue4m3 带尾数(尾数相乘吃精度预算): 指数{0.5,1,2} × 偶尾数{1,1.25,1.5,1.75} → 12档
static const EncVal UE4M3_SET[] = {
    {0x30,0.5f},{0x32,0.625f},{0x34,0.75f},{0x36,0.875f},
    {0x38,1.f},{0x3A,1.25f},{0x3C,1.5f},{0x3E,1.75f},
    {0x40,2.f},{0x42,2.5f},{0x44,3.f},{0x46,3.5f}};

// CPU 参考(带 per-K-segment scale, scale 对所有行广播):
//   D[m][n] = Σ_seg sfa[seg]*sfb[seg] * Σ_{k∈seg} A[m][k]*B[n][k]
static inline void cpu_gemm_ref_bs(const float* A, const float* B, float* D,
                                   int M, int N, int K,
                                   const float* sfa, const float* sfb, int nseg) {
    int segK = K / nseg;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int seg = 0; seg < nseg; seg++) {
                float s = 0.f;
                for (int k = seg * segK; k < (seg + 1) * segK; k++)
                    s += A[m * K + k] * B[n * K + k];
                acc += sfa[seg] * sfb[seg] * s;
            }
            D[m * N + n] = acc;
        }
}
