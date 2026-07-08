/**
 * tcgen05.mma 最小单元 — MXFP4 block32 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
 *       [tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p
 *
 * MXFP4: e2m1 data + ue8m0 scale, 每32个FP4共享1个scale
 * Tile:  M=128, N=8, K=64, 128线程 (4 warp, cta_group::1)
 * FMA:   128 × 8 × 64 = 65,536
 *
 * 数据流:
 *   1. tcgen05.alloc  → 分配TMEM (累加器 + scale)
 *   2. 填充smem        → A[128×64] FP4, B[8×64] FP4
 *   3. tcgen05.st     → 写ue8m0 scale到TMEM
 *   4. make_desc()    → 构造smem descriptor (uint64)
 *   5. tcgen05.mma    → 1个线程发射 (elect_one)
 *   6. tcgen05.ld     → 从TMEM读结果
 *   7. tcgen05.dealloc → 释放TMEM
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_mxfp4_block32_ue8m0 tcgen05_mxfp4_block32_ue8m0.cu
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

// SmemDescriptor: [0:14) addr>>4, [16:30) LBO>>4, [32:46) SBO>>4, [46:48) version=1
// ⚠️ no-swizzle 布局是 8行×16B core-matrix 分块(不是线性行主): K方向间距LBO=128B, M/N方向SBO=256B
//    本例全1.0输入对排布不敏感; 喂真实数据时 smem 必须按 core-matrix 排, 见 random_cpu_ref/
__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    (void)stride_bytes;
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    desc |= (uint64_t)(1) << 46;
    return desc;
}

// InstrDescriptorBlockScaled for MXFP4: a/b=E2M1, scale=UE8M0, M=128, N=8
__device__ uint32_t make_idesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    d |= (1 << 7);                    // a_format = E2M1
    d |= (1 << 10);                   // b_format = E2M1
    d |= (1 << 17);                   // n_dim = N/8 = 1
    d |= (1 << 23);                   // scale_format = 1 (UE8M0)
    d |= (8 << 24);                   // m_dim = M/16 = 8
    d |= ((tsfa >> 30) & 3) << 29;    // a_sf_id
    d |= ((tsfb >> 30) & 3) << 4;     // b_sf_id
    return d;
}

constexpr int M = 128, N = 8, K = 64;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K / 2;  // 4096
constexpr int B_BYTES = N * K / 2;  // 256

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 阶段2: 随机数据 + 随机scale vs CPU 参考 (逐bit精确)
//   数据每元素独立随机(e2m1可精确编码集); scale 每轮随机、各K段取同值
//   (段间独立随机需先摸清TMEM内sf字节→K段映射, 此处不假设; 段一致时CPU参考不受影响)
//   D 布局用 base-4 探针实测 + 一致性过滤(见 tcgen05 常规精度文件, 同一框架)。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

static const int EBITS = 4;
static uint32_t phase2_enc_small(int v) {   // e2m1: {0,1,2,3,4,6}
    static const int      vals[] = {0, 1, 2, 3, 4, 6};
    static const uint32_t encs[] = {0x0, 0x2, 0x4, 0x5, 0x6, 0x7};
    for (int i = 0; i < 6; i++) if (vals[i] == v) return encs[i];
    return 0;
}

__global__ void phase2_random_kernel(const uint8_t* A, const uint8_t* B, uint32_t* D_raw,
                                     uint32_t sfa_splat, uint32_t sfb_splat) {
    extern __shared__ char smem[];
    uint32_t* sA = (uint32_t*)smem;
    uint32_t* sB = (uint32_t*)(smem + A_BYTES);
    uint32_t* s_tmem = (uint32_t*)(smem + A_BYTES + B_BYTES + 64);
    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;
    for (int i = tid; i < A_BYTES / 4; i += THREADS) sA[i] = ((uint32_t*)A)[i];
    for (int i = tid; i < B_BYTES / 4; i += THREADS) sB[i] = ((uint32_t*)B)[i];
    __syncthreads();
    if (warp_id == 0)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(smem_u32(s_tmem)), "r"(32));
    __syncthreads();
    uint32_t tmem_c = *s_tmem;
    uint32_t tmem_sfa = tmem_c + 8, tmem_sfb = tmem_c + 12;
    if (warp_id == 0) {
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfa), "r"(sfa_splat), "r"(sfa_splat), "r"(sfa_splat), "r"(sfa_splat));
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfb), "r"(sfb_splat), "r"(sfb_splat), "r"(sfb_splat), "r"(sfb_splat));
    }
    __syncthreads();
    uint64_t da = make_desc(sA, 32), db = make_desc(sB, 32);
    uint32_t idesc = make_idesc(tmem_sfa, tmem_sfb);
    __syncthreads();
    if (tid == 0) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
            : : "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(0u),
                "r"(tmem_sfa), "r"(tmem_sfb));
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
                     : : "r"(tmem_c), "r"(32));
}

static void phase2_to_core_matrix(uint8_t* buf, int rows) {
    static uint8_t tmp[128 * 32];
    memcpy(tmp, buf, rows * 32);
    for (int m = 0; m < rows; m++)
        for (int kb = 0; kb < 32; kb++)
            buf[(m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16] = tmp[m*32 + kb];
}
static float phase2_u32f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static uint8_t p2_hA[A_BYTES], p2_hB[B_BYTES];
static uint32_t p2_hD[128];
static void phase2_run(uint8_t* dA, uint8_t* dB, uint32_t* dD, uint32_t sfa, uint32_t sfb) {
    phase2_to_core_matrix(p2_hA, M); phase2_to_core_matrix(p2_hB, N);
    CHECK_CUDA(cudaMemcpy(dA, p2_hA, A_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, p2_hB, B_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dD, 0, 512));
    phase2_random_kernel<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD, sfa, sfb);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(p2_hD, dD, 512, cudaMemcpyDeviceToHost));
}
static void phase2_probe(uint8_t* dA, uint8_t* dB, uint32_t* dD,
                         int adigit, int bdigit, int dbl, int* out) {
    memset(p2_hA, 0, A_BYTES); memset(p2_hB, 0, B_BYTES);
    for (int m = 0; m < M; m++) {
        int v = adigit < 0 ? 1 : ((m >> (2*adigit)) & 3) * (dbl ? 2 : 1);
        pack_set(p2_hA, m * K, phase2_enc_small(v), EBITS);
    }
    for (int n = 0; n < N; n++) {
        int v = bdigit < 0 ? 1 : ((n >> (2*bdigit)) & 3);
        pack_set(p2_hB, n * K, phase2_enc_small(v), EBITS);
    }
    phase2_run(dA, dB, dD, 0x7Fu * 0x01010101u, 0x7Fu * 0x01010101u);
    for (int s = 0; s < 128; s++) out[s] = (int)phase2_u32f(p2_hD[s]);
}

static int phase2_random_vs_cpu(uint32_t seed) {
    const int ROUNDS = 3;
    static float refA[128*64], refB[8*64], refD[128*8];
    uint8_t *dA, *dB; uint32_t* dD;
    CHECK_CUDA(cudaMalloc(&dA, A_BYTES)); CHECK_CUDA(cudaMalloc(&dB, B_BYTES));
    CHECK_CUDA(cudaMalloc(&dD, 512));
    printf("\n====== 阶段2: 随机数据+随机scale vs CPU参考 (seed=%u) ======\n", seed);
    static int pr[4][128], pc[2][128], pchk[128], mrow[128], mcol[128];
    static bool valid[128];
    for (int d = 0; d < 4; d++) phase2_probe(dA, dB, dD, d, -1, 0, pr[d]);
    for (int d = 0; d < 2; d++) phase2_probe(dA, dB, dD, -1, d, 0, pc[d]);
    phase2_probe(dA, dB, dD, 0, -1, 1, pchk);
    int nvalid = 0; static bool seen[128][8]; memset(seen, 0, sizeof(seen));
    for (int s = 0; s < 128; s++) {
        int row = pr[0][s] + 4*pr[1][s] + 16*pr[2][s] + 64*pr[3][s];
        int col = pc[0][s] + 4*pc[1][s];
        bool ok = true;
        for (int d = 0; d < 4; d++) if (pr[d][s] < 0 || pr[d][s] > 3) ok = false;
        for (int d = 0; d < 2; d++) if (pc[d][s] < 0 || pc[d][s] > 3) ok = false;
        ok = ok && row < M && col < N && pchk[s] == 2*pr[0][s] && !seen[row][col%8];
        valid[s] = ok; mrow[s] = row; mcol[s] = col;
        if (ok) { seen[row][col%8] = true; nvalid++; }
    }
    printf("D布局探针: %d/128 槽位一致有效\n", nvalid);
    if (nvalid < 32) { printf("====== 阶段2 FAIL (探针不足) ======\n"); return 1; }
    int bad = 0;
    for (int r = 0; r < ROUNDS; r++) {
        const EncVal& ea = UE8M0_SET[xorshift(&seed) % SETN(UE8M0_SET)];
        const EncVal& eb = UE8M0_SET[xorshift(&seed) % SETN(UE8M0_SET)];
        float sfaf = ea.val, sfbf = eb.val;
        memset(p2_hA, 0, A_BYTES); memset(p2_hB, 0, B_BYTES);
        fill_random(p2_hA, refA, M, K, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
        fill_random(p2_hB, refB, N, K, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
        cpu_gemm_ref_bs(refA, refB, refD, M, N, K, &sfaf, &sfbf, 1);
        phase2_run(dA, dB, dD, ea.enc * 0x01010101u, eb.enc * 0x01010101u);
        float got[128], want[128]; int nv = 0;
        for (int s = 0; s < 128; s++) {
            if (!valid[s]) continue;
            got[nv]  = phase2_u32f(p2_hD[s]);
            want[nv] = refD[mrow[s] * N + mcol[s]]; nv++;
        }
        char tag[16]; snprintf(tag, sizeof(tag), "random r%d", r);
        bad += check_exact(got, want, nv, tag);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    printf("====== 阶段2 %s ======\n", bad ? "FAIL" : "全部PASS");
    return bad ? 1 : 0;
}

int main() {
    return phase2_random_vs_cpu(1);
}
