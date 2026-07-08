/**
 * tcgen05.mma 最小单元 — FP8 e4m3 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::f8f6f4
 *       [tmem_c], desc_a, desc_b, idesc, p
 *
 * FP8 e4m3: [sign|exp(4)|mant(3)], bias=7, 1.0 = 0x38
 * 无 block scale (对比 nvfp4/mxfp4 的 kind::mxf4nvf4.block_scale)
 * Tile:  M=128, N=8, K=32 (K = 32字节/元素1字节), 128线程 (4 warp, cta_group::1)
 * FMA:   128 × 8 × 32 = 32,768
 *
 * 数据流 (比 nvfp4 少了写 scale 到 TMEM 这步):
 *   1. tcgen05.alloc  → 分配TMEM (只有累加器)
 *   2. 填充smem        → A[128×32] e4m3, B[8×32] e4m3
 *   3. make_desc()    → 构造smem descriptor (uint64)
 *   4. tcgen05.mma    → 1个线程发射 (elect_one)
 *   5. tcgen05.ld     → 从TMEM读结果
 *   6. tcgen05.dealloc → 释放TMEM
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_fp8 tcgen05_fp8.cu
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

// InstrDescriptor for kind::f8f6f4 (非block_scale布局, 与mxf4nvf4不同):
//   [4:6) d_format=1(f32), [7:10) a_format=0(E4M3), [10:13) b_format=0(E4M3)
//   [17:23) n_dim=N>>3, [24:29) m_dim=M>>4
__device__ uint32_t make_idesc() {
    uint32_t d = 0;
    d |= (1 << 4);                    // d_format = F32
    // a_format = 0 (E4M3), b_format = 0 (E4M3)
    d |= (1 << 17);                   // n_dim = N/8 = 1
    d |= (8 << 24);                   // m_dim = M/16 = 8
    return d;
}

constexpr int M = 128, N = 8, K = 32;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K;      // 4096
constexpr int B_BYTES = N * K;      // 256

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 阶段2: 随机数据 vs CPU 参考 (逐bit精确)
//   smem 按 8行×16B core-matrix 排布; D 读回布局不做假设 —— base-4 探针实测:
//   行/列按4进制逐位探测(探针值只用0..3, 所有格式可精确编码), 一致性探针(×2)
//   过滤 TMEM 跨kernel残留槽位; 然后随机数据按测得映射与 CPU 参考逐元素 == 。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

static const int EBITS = 8;  // P2EBITS_FIXED
static uint32_t phase2_enc_small(int v) {   // 编码 {0,1,2,3,4,6} → 本格式
    static const int      vals[] = {0, 1, 2, 3, 4, 6};
    static const uint32_t encs[] = {0x00, 0x38, 0x40, 0x44, 0x48, 0x4C};
    for (int i = 0; i < 6; i++) if (vals[i] == v) return encs[i];
    return 0;
}

__global__ void phase2_random_kernel(const uint8_t* A, const uint8_t* B, uint32_t* D_raw) {
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

    uint64_t da = make_desc(sA, 32), db = make_desc(sB, 32);
    __syncthreads();
    if (tid == 0) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n"
            : : "r"(tmem_c), "l"(da), "l"(db), "r"((uint32_t)((1<<4)|(1<<17)|(8<<24))), "r"(0u));
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
static void phase2_run(uint8_t* dA, uint8_t* dB, uint32_t* dD) {
    phase2_to_core_matrix(p2_hA, M); phase2_to_core_matrix(p2_hB, N);
    CHECK_CUDA(cudaMemcpy(dA, p2_hA, A_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, p2_hB, B_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dD, 0, 512));
    phase2_random_kernel<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(p2_hD, dD, 512, cudaMemcpyDeviceToHost));
}
// one-hot 探针: A[m][0]=va(m), B[n][0]=vb(n) → D[m][n]=va*vb
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
    phase2_run(dA, dB, dD);
    for (int s = 0; s < 128; s++) out[s] = (int)phase2_u32f(p2_hD[s]);
}

static int phase2_random_vs_cpu(uint32_t seed) {
    const int ROUNDS = 3, IS_INT = 0;
    static float refA[128*32], refB[8*32], refD[128*8];
    uint8_t *dA, *dB; uint32_t* dD;
    CHECK_CUDA(cudaMalloc(&dA, A_BYTES)); CHECK_CUDA(cudaMalloc(&dB, B_BYTES));
    CHECK_CUDA(cudaMalloc(&dD, 512));
    printf("\n====== 阶段2: 随机数据 vs CPU参考 (seed=%u) ======\n", seed);
    // ---- base-4 探针: 行(4位) + 列(2位, N=8) + 一致性(行位0 ×2) ----
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

        memset(p2_hA, 0, A_BYTES); memset(p2_hB, 0, B_BYTES);
        fill_random(p2_hA, refA, M, K, E4M3_SET, SETN(E4M3_SET), EBITS, &seed);
        fill_random(p2_hB, refB, N, K, E4M3_SET, SETN(E4M3_SET), EBITS, &seed);
        cpu_gemm_ref(refA, refB, refD, M, N, K);
        phase2_run(dA, dB, dD);
        float got[128], want[128]; int nv = 0;
        for (int s = 0; s < 128; s++) {
            if (!valid[s]) continue;
            got[nv]  = IS_INT ? (float)(int32_t)p2_hD[s] : phase2_u32f(p2_hD[s]);
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
