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

// ============================================================
// vr_verify(): 通用「随机 vs CPU」host 驱动 — 每个.cu只需 kernel + launch适配器
// ============================================================
// GMMA/UMMA no-swizzle core-matrix 重排: (m,kb) → (m/8)*256+(kb/16)*128+(m%8)*16+kb%16
static inline void vr_to_core_matrix(uint8_t* buf, int rows) {
    static uint8_t tmp[256 * 32];
    memcpy(tmp, buf, rows * 32);
    for (int m = 0; m < rows; m++)
        for (int kb = 0; kb < 32; kb++)
            buf[(m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16] = tmp[m*32 + kb];
}
static inline float vr_u32f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// {0,1,2,3,4,6} 的各格式编码表 (探针值, fp4也可精确编码)
static const uint32_t VR_ENC_FP16[6] = {0x0000,0x3C00,0x4000,0x4200,0x4400,0x4600};
static const uint32_t VR_ENC_BF16[6] = {0x0000,0x3F80,0x4000,0x4040,0x4080,0x40C0};
static const uint32_t VR_ENC_TF32[6] = {0,0x3F800000,0x40000000,0x40400000,0x40800000,0x40C00000};
static const uint32_t VR_ENC_E4M3[6] = {0x00,0x38,0x40,0x44,0x48,0x4C};
static const uint32_t VR_ENC_E2M1[6] = {0x0,0x2,0x4,0x5,0x6,0x7};
static const uint32_t VR_ENC_S8[6]   = {0,1,2,3,4,6};
static inline uint32_t vr_enc_small(const uint32_t* tab, int v) {
    static const int vals[6] = {0,1,2,3,4,6};
    for (int i = 0; i < 6; i++) if (vals[i] == v) return tab[i];
    return 0;
}

struct VrSpec {
    int M, N, K;                        // 逻辑形状
    int ebits, is_int;
    const EncVal* dset; int dsetn;      // 数据取值集
    const EncVal* sset; int ssetn;      // scale取值集 (NULL=无block scale)
    int nseg;                           // scale段数
    uint32_t sf_one;                    // scale=1.0 的编码 (探针用)
    int core_matrix;                    // A/B 是否需 core-matrix 重排
    int probe;                          // 1=D布局未知(tcgen05), 探针实测; 0=run返回线性float D[M*N]
    int nslot;                          // probe模式下 run 返回的 raw 槽位数
    const uint32_t* enc_tab;            // 探针小整数编码表 (probe=1 时必填)
};
// run: hA/hB 为已按最终布局排好的编码, sf*_enc 每段一个编码(无scale时忽略),
//      out: probe模式填 nslot 个 raw u32; 否则填 M*N 个 float
typedef void (*VrRunFn)(const uint8_t* hA, const uint8_t* hB,
                        const uint32_t* sfa_enc, const uint32_t* sfb_enc, void* out);

static inline int vr_verify(const VrSpec& sp, VrRunFn run, uint32_t seed) {
    static uint8_t hA[256 * 32], hB[256 * 32];
    static float refA[256 * 64], refB[256 * 64], refD[256 * 256];
    static uint32_t draw[256]; static float df[256 * 256];
    uint32_t sf1[8]; for (int g = 0; g < 8; g++) sf1[g] = sp.sf_one;
    const int ROUNDS = 3;
    printf("\n====== 随机数据%s vs CPU参考 (seed=%u) ======\n",
           sp.sset ? "+随机scale" : "", seed);

    static int mrow[256], mcol[256]; static bool valid[256];
    int ncheck;
    if (sp.probe) {                          // base-4 探针实测 D 布局
        static int pr[4][256], pc[4][256], pchk[256];
        auto probe1 = [&](int ad, int bd, int dbl, int* out_) {
            memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
            for (int m = 0; m < sp.M; m++) {
                int v = ad < 0 ? 1 : ((m >> (2*ad)) & 3) * (dbl ? 2 : 1);
                pack_set(hA, m * sp.K, vr_enc_small(sp.enc_tab, v), sp.ebits);
            }
            for (int n = 0; n < sp.N; n++) {
                int v = bd < 0 ? 1 : ((n >> (2*bd)) & 3);
                pack_set(hB, n * sp.K, vr_enc_small(sp.enc_tab, v), sp.ebits);
            }
            if (sp.core_matrix) { vr_to_core_matrix(hA, sp.M); vr_to_core_matrix(hB, sp.N); }
            run(hA, hB, sf1, sf1, draw);
            for (int s = 0; s < sp.nslot; s++)
                out_[s] = sp.is_int ? (int)(int32_t)draw[s] : (int)vr_u32f(draw[s]);
        };
        for (int d = 0; d < 4; d++) probe1(d, -1, 0, pr[d]);
        for (int d = 0; d < 4; d++) probe1(-1, d, 0, pc[d]);
        probe1(0, -1, 1, pchk);
        static bool seen[256][256]; memset(seen, 0, sizeof(seen));
        int nvalid = 0;
        for (int s = 0; s < sp.nslot; s++) {
            bool ok = true; int row = 0, col = 0;
            for (int d = 0; d < 4; d++) {
                if (pr[d][s] < 0 || pr[d][s] > 3 || pc[d][s] < 0 || pc[d][s] > 3) ok = false;
                else { row += pr[d][s] << (2*d); col += pc[d][s] << (2*d); }
            }
            ok = ok && row < sp.M && col < sp.N && pchk[s] == 2*pr[0][s] && !seen[row][col];
            valid[s] = ok; mrow[s] = row; mcol[s] = col;
            if (ok) { seen[row][col] = true; nvalid++; }
        }
        printf("D布局探针: %d/%d 槽位一致有效\n", nvalid, sp.nslot);
        if (nvalid < sp.nslot / 4) { printf("====== FAIL (探针不足) ======\n"); return 1; }
        ncheck = nvalid;
    } else ncheck = sp.M * sp.N;

    int bad = 0;
    for (int r = 0; r < ROUNDS; r++) {
        uint32_t sfa[8], sfb[8]; float sfaf[8], sfbf[8];
        for (int g = 0; g < 8; g++) { sfa[g] = sfb[g] = sp.sf_one; sfaf[g] = sfbf[g] = 1.f; }
        if (sp.sset)
            for (int g = 0; g < sp.nseg; g++) {
                const EncVal& ea = sp.sset[xorshift(&seed) % sp.ssetn];
                const EncVal& eb = sp.sset[xorshift(&seed) % sp.ssetn];
                sfa[g] = ea.enc; sfaf[g] = ea.val; sfb[g] = eb.enc; sfbf[g] = eb.val;
            }
        memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
        fill_random(hA, refA, sp.M, sp.K, sp.dset, sp.dsetn, sp.ebits, &seed);
        fill_random(hB, refB, sp.N, sp.K, sp.dset, sp.dsetn, sp.ebits, &seed);
        if (sp.sset) cpu_gemm_ref_bs(refA, refB, refD, sp.M, sp.N, sp.K, sfaf, sfbf, sp.nseg);
        else         cpu_gemm_ref(refA, refB, refD, sp.M, sp.N, sp.K);
        if (sp.core_matrix) { vr_to_core_matrix(hA, sp.M); vr_to_core_matrix(hB, sp.N); }
        run(hA, hB, sfa, sfb, sp.probe ? (void*)draw : (void*)df);
        static float got[65536], want[65536]; int nv = 0;
        if (sp.probe) {
            for (int s = 0; s < sp.nslot; s++) {
                if (!valid[s]) continue;
                got[nv]  = sp.is_int ? (float)(int32_t)draw[s] : vr_u32f(draw[s]);
                want[nv] = refD[mrow[s] * sp.N + mcol[s]]; nv++;
            }
        } else {
            for (int i = 0; i < sp.M * sp.N; i++) { got[nv] = df[i]; want[nv] = refD[i]; nv++; }
        }
        char tag[16]; snprintf(tag, sizeof(tag), "random r%d", r);
        bad += check_exact(got, want, nv, tag);
    }
    printf("====== %s ======\n", bad ? "FAIL" : "全部PASS");
    return bad ? 1 : 0;
}
