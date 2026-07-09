/**
 * tcgen05.mma 最大tile — MXFP4 block32 (sm_100)
 *
 * cta_group::1 (4 warps = 128 threads), M=128, N=256, K=64
 * 单条指令: 128 × 256 × 64 = 2,097,152 FMA (~2M FMA)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_mxfp4_block32_ue8m0_maxtile tcgen05_mxfp4_block32_ue8m0_maxtile.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

__device__ uint32_t smem_u32(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    // ⚠️ no-swizzle 实际是 8行×16B core-matrix 分块(全1.0输入不敏感, 真实数据排布见 random_cpu_ref/)
    (void)stride_bytes;
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    desc |= (uint64_t)(1) << 46;
    return desc;
}

// MXFP4: scale_format=1 (UE8M0)
__device__ uint32_t make_idesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    d |= (1 << 7);                    // a_format = E2M1
    d |= (1 << 10);                   // b_format = E2M1
    d |= (32 << 17);                  // n_dim = N/8 = 32 (N=256)
    d |= (1 << 23);                   // scale_format = UE8M0
    d |= (8 << 24);                   // m_dim = M/16 = 8 (M=128)
    d |= ((tsfa >> 30) & 3) << 29;
    d |= ((tsfb >> 30) & 3) << 4;
    return d;
}

constexpr int M = 128, N = 256, K = 64;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K / 2;  // 4096
constexpr int B_BYTES = N * K / 2;  // 8192

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 阶段2: 随机数据 + 随机scale vs CPU 参考 (逐bit精确, 自包含)
//   与本文件阶段1同指令/同tile形状; D 布局 base-4 探针实测(行/列各4位, 支持N=256);
//   K方向 1 个tile依次累加(p=0清零/p=1累加), 与阶段1的K循环语义一致。
//   scale 每轮随机、段间一致(段独立需先摸TMEM sf字节→K段映射, 不做假设)。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

namespace p2 {
constexpr int M = 128, N = 256, K = 64, KLOOP = 1, EBITS = 4, THREADS = 128;
constexpr int AT_BYTES = M * 32, BT_BYTES = N * 32;      // 每个K-tile的A/B字节数
constexpr int NSLOT = 128;

__device__ __forceinline__ uint32_t su32(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t mkdesc(void const* p) {
    uint64_t d = 0;
    d |= (uint64_t)((su32(p) >> 4) & 0x3FFF);
    d |= (uint64_t)8  << 16;   // LBO = 128B >> 4 (core-matrix K方向)
    d |= (uint64_t)16 << 32;   // SBO = 256B >> 4 (M/N方向)
    d |= (uint64_t)1  << 46;
    return d;
}
__device__ __forceinline__ uint32_t mkidesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    d |= (1u << 7) | (1u << 10);          // a/b = E2M1
    d |= (uint32_t)(N / 8) << 17;
    d |= 1u << 23;               // scale_format: 1=UE8M0, 0=UE4M3
    d |= 8u << 24;
    d |= ((tsfa >> 30) & 3) << 29;
    d |= ((tsfb >> 30) & 3) << 4;
    return d;
}

__global__ void kern(const uint8_t* A, const uint8_t* B, uint32_t* D_raw,
                     uint32_t sfa_splat, uint32_t sfb_splat) {
    extern __shared__ char smem[];
    uint32_t* s_tmem = (uint32_t*)(smem + KLOOP * (AT_BYTES + BT_BYTES) + 64);
    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;
    for (int i = tid; i < KLOOP * (AT_BYTES + BT_BYTES) / 4; i += THREADS)
        ((uint32_t*)smem)[i] = ((const uint32_t*)A)[i];   // A/B所有tile连续打包传入
    __syncthreads();
    (void)B;
    if (warp_id == 0)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(su32(s_tmem)), "r"(512));
    __syncthreads();
    uint32_t tmem_c = *s_tmem;
    uint32_t tmem_sfa = tmem_c + 256, tmem_sfb = tmem_c + 300;
    if (warp_id == 0) {
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfa), "r"(sfa_splat), "r"(sfa_splat), "r"(sfa_splat), "r"(sfa_splat));
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfb), "r"(sfb_splat), "r"(sfb_splat), "r"(sfb_splat), "r"(sfb_splat));
    }
    __syncthreads();
    uint32_t idesc = mkidesc(tmem_sfa, tmem_sfb);
    __syncthreads();
    if (tid == 0) {
        for (int t = 0; t < KLOOP; t++) {
            uint64_t da = mkdesc(smem + t * (AT_BYTES + BT_BYTES));
            uint64_t db = mkdesc(smem + t * (AT_BYTES + BT_BYTES) + AT_BYTES);
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
                : : "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"((uint32_t)t),
                    "r"(tmem_sfa), "r"(tmem_sfb));
        }
    }
    __syncthreads();
    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n\t"
                     "tcgen05.wait::ld.sync.aligned;\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));
        D_raw[lane*4+0]=r0; D_raw[lane*4+1]=r1; D_raw[lane*4+2]=r2; D_raw[lane*4+3]=r3;
    }
    __syncthreads();
    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_c), "r"(512));
}

inline uint32_t enc_small(int v) {   // e2m1: {0,1,2,3,4,6}
    static const int      vals[] = {0, 1, 2, 3, 4, 6};
    static const uint32_t encs[] = {0x0, 0x2, 0x4, 0x5, 0x6, 0x7};
    for (int i = 0; i < 6; i++) if (vals[i] == v) return encs[i];
    return 0;
}
inline void to_core(uint8_t* buf, int rows) {
    static uint8_t tmp[256 * 32];
    memcpy(tmp, buf, rows * 32);
    for (int m = 0; m < rows; m++)
        for (int kb = 0; kb < 32; kb++)
            buf[(m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16] = tmp[m*32 + kb];
}
inline float u32f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// host缓冲: 每tile [A|B] 连续打包
static uint8_t hAB[KLOOP * (AT_BYTES + BT_BYTES)];
static uint32_t hD[NSLOT];
inline uint8_t* tileA(int t) { return hAB + t * (AT_BYTES + BT_BYTES); }
inline uint8_t* tileB(int t) { return hAB + t * (AT_BYTES + BT_BYTES) + AT_BYTES; }

inline void run(uint8_t* dAB, uint32_t* dD, uint32_t sfa, uint32_t sfb) {
    for (int t = 0; t < KLOOP; t++) { to_core(tileA(t), M); to_core(tileB(t), N); }
    cudaMemcpy(dAB, hAB, sizeof(hAB), cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, sizeof(hD));
    int smem_bytes = KLOOP * (AT_BYTES + BT_BYTES) + 256;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kern<<<1, THREADS, smem_bytes>>>(dAB, nullptr, dD, sfa, sfb);
    cudaDeviceSynchronize();
    cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);
}
inline void probe(uint8_t* dAB, uint32_t* dD, int adigit, int bdigit, int dbl, int* out) {
    memset(hAB, 0, sizeof(hAB));
    for (int m = 0; m < M; m++) {
        int v = adigit < 0 ? 1 : ((m >> (2*adigit)) & 3) * (dbl ? 2 : 1);
        pack_set(tileA(0), m * K, enc_small(v), EBITS);   // 只在tile0放信号
    }
    for (int n = 0; n < N; n++) {
        int v = bdigit < 0 ? 1 : ((n >> (2*bdigit)) & 3);
        pack_set(tileB(0), n * K, enc_small(v), EBITS);
    }
    run(dAB, dD, 0x7Fu * 0x01010101u, 0x7Fu * 0x01010101u);
    for (int s = 0; s < NSLOT; s++) out[s] = (int)u32f(hD[s]);
}

inline int random_vs_cpu(uint32_t seed) {
    const int ROUNDS = 3;
    static float refA[M * K * KLOOP], refB[N * K * KLOOP], refD[M * N];
    uint8_t* dAB; uint32_t* dD;
    cudaMalloc(&dAB, sizeof(hAB)); cudaMalloc(&dD, sizeof(hD));
    printf("\n====== 阶段2: 随机数据+随机scale vs CPU参考 (seed=%u, K_total=%d) ======\n",
           seed, K * KLOOP);
    static int pr[4][NSLOT], pc[4][NSLOT], pchk[NSLOT], mrow[NSLOT], mcol[NSLOT];
    static bool valid[NSLOT];
    for (int d = 0; d < 4; d++) probe(dAB, dD, d, -1, 0, pr[d]);
    for (int d = 0; d < 4; d++) probe(dAB, dD, -1, d, 0, pc[d]);
    probe(dAB, dD, 0, -1, 1, pchk);
    int nvalid = 0; static bool seen[128][256]; memset(seen, 0, sizeof(seen));
    for (int s = 0; s < NSLOT; s++) {
        bool ok = true; int row = 0, col = 0;
        for (int d = 0; d < 4; d++) {
            if (pr[d][s] < 0 || pr[d][s] > 3 || pc[d][s] < 0 || pc[d][s] > 3) ok = false;
            else { row += pr[d][s] << (2*d); col += pc[d][s] << (2*d); }
        }
        ok = ok && row < M && col < N && pchk[s] == 2*pr[0][s] && !seen[row][col];
        valid[s] = ok; mrow[s] = row; mcol[s] = col;
        if (ok) { seen[row][col] = true; nvalid++; }
    }
    printf("D布局探针: %d/%d 槽位一致有效\n", nvalid, NSLOT);
    if (nvalid < 32) { printf("====== 阶段2 FAIL (探针不足) ======\n"); return 1; }
    int bad = 0;
    for (int r = 0; r < ROUNDS; r++) {
        const EncVal& ea = UE8M0_SET[xorshift(&seed) % SETN(UE8M0_SET)];
        const EncVal& eb = UE8M0_SET[xorshift(&seed) % SETN(UE8M0_SET)];
        memset(hAB, 0, sizeof(hAB));
        for (int t = 0; t < KLOOP; t++) {
            fill_random(tileA(t), refA + t * M * K, M, K, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
            fill_random(tileB(t), refB + t * N * K, N, K, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
        }
        for (int m = 0; m < M; m++)                     // CPU参考: K_total = K*KLOOP
            for (int n = 0; n < N; n++) {
                float acc = 0.f;
                for (int t = 0; t < KLOOP; t++)
                    for (int k = 0; k < K; k++)
                        acc += refA[t*M*K + m*K + k] * refB[t*N*K + n*K + k];
                refD[m*N + n] = ea.val * eb.val * acc;
            }
        run(dAB, dD, ea.enc * 0x01010101u, eb.enc * 0x01010101u);
        float got[NSLOT], want[NSLOT]; int nv = 0;
        for (int s = 0; s < NSLOT; s++) {
            if (!valid[s]) continue;
            got[nv] = u32f(hD[s]); want[nv] = refD[mrow[s]*N + mcol[s]]; nv++;
        }
        char tag[16]; snprintf(tag, sizeof(tag), "random r%d", r);
        bad += check_exact(got, want, nv, tag);
    }
    cudaFree(dAB); cudaFree(dD);
    printf("====== 阶段2 %s ======\n", bad ? "FAIL" : "全部PASS");
    return bad ? 1 : 0;
}
}  // namespace p2

int main() {
    return p2::random_vs_cpu(/*seed=*/1);   // 随机全在CPU侧生成, 同seed可复现
}
