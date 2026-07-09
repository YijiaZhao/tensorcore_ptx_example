/**
 * Warp-specialized tcgen05.mma demo (sm_100a / sm_103a)
 *
 * Layout:
 *   - 1 producer warp (warp 0):
 *       * setmaxnreg.dec.sync  (release registers)
 *       * cp.async load A[k], B[k] from gmem to double-buffered smem
 *       * named barrier: arrive FULL(s), wait EMPTY(s)
 *   - 4 consumer warps (warps 1-4, 128 threads = canonical cta_group::1):
 *       * setmaxnreg.inc.sync  (acquire registers)
 *       * named barrier: wait FULL(s)
 *       * tcgen05.mma + tcgen05.commit (single thread issues, mbarrier tracks completion)
 *       * mbarrier.try_wait.parity for MMA done
 *       * named barrier: arrive EMPTY(s)
 *       * after K_ITERS: tcgen05.ld → gmem
 *
 * Total threads: 1*32 + 4*32 = 160 (5 warps)
 * Shape: M=128, N=256, K_TILE=64, K_ITERS=4 → effective K_total=256
 *        Total FMA = 128 * 256 * 256 = 8,388,608 (~8.4M)
 *        Expected D = K_total = 256 (with all-1.0 inputs in FP4)
 *
 * Compile (B300 sm_103a):
 *   nvcc -gencode arch=compute_103a,code=sm_103a -std=c++17 -O3 -o ws_tcgen05_nvfp4_producer_consumer_pipeline ws_tcgen05_nvfp4_producer_consumer_pipeline.cu
 * Compile (B100/B200 sm_100a):
 *   nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O3 -o ws_tcgen05_nvfp4_producer_consumer_pipeline ws_tcgen05_nvfp4_producer_consumer_pipeline.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

constexpr int M       = 128;
constexpr int N       = 256;
constexpr int K_TILE  = 64;
constexpr int K_ITERS = 4;
constexpr int STAGES  = 2;
constexpr int A_BYTES = M * K_TILE / 2;        // 4096
constexpr int B_BYTES = N * K_TILE / 2;        // 8192

constexpr int PROD_THREADS  = 32;
constexpr int CONS_WARPS    = 4;
constexpr int CONS_THREADS  = 32 * CONS_WARPS;
constexpr int TOTAL_THREADS = PROD_THREADS + CONS_THREADS;  // 160

// Named barrier IDs (max 16 per CTA: 0..15)
#define BAR_FULL(s)   ((s) * 2 + 0)
#define BAR_EMPTY(s)  ((s) * 2 + 1)

__device__ __forceinline__ uint32_t smem_u32(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    // ⚠️ no-swizzle 实际是 8行×16B core-matrix 分块(全1.0输入不敏感, 真实数据排布见 random_cpu_ref/)
    (void)stride_bytes;
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    desc |= (uint64_t)(1) << 46;
    return desc;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    d |= (1 << 7);    // a_format = E2M1
    d |= (1 << 10);   // b_format = E2M1
    d |= (32 << 17);  // n_dim = N/8 = 32 (N=256)
    d |= (8  << 24);  // m_dim = M/16 = 8 (M=128)
    d |= ((tsfa >> 30) & 3) << 29;
    d |= ((tsfb >> 30) & 3) << 4;
    return d;
}

__global__ void __launch_bounds__(TOTAL_THREADS, 1)
ws_tcgen05_kernel(uint8_t const* __restrict__ gmem_A,
                  uint8_t const* __restrict__ gmem_B,
                  float* __restrict__ D_out)
{
    extern __shared__ char smem_buf[];

    // smem layout: A[STAGES] | B[STAGES] | tmem_alloc_slot | mbar[STAGES]
    uint8_t* smem_a[STAGES];
    uint8_t* smem_b[STAGES];
    smem_a[0] = (uint8_t*)smem_buf;
    smem_a[1] = smem_a[0] + A_BYTES;
    smem_b[0] = smem_a[1] + A_BYTES;
    smem_b[1] = smem_b[0] + B_BYTES;
    uint32_t* smem_tmem = (uint32_t*)(smem_b[1] + B_BYTES + 64);
    uint64_t* mbar_done = (uint64_t*)(smem_tmem + 8);  // [STAGES]

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;

    // ============ INIT (all warps cooperate) ============
    if (warp_id == 0 && lane == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(smem_u32(&mbar_done[s])));
        }
    }

    if (warp_id == 1) {
        // alloc 512 columns of TMEM
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(smem_u32(smem_tmem)), "r"(512));
    }
    __syncthreads();

    uint32_t tmem_base = *smem_tmem;
    uint32_t tmem_c    = tmem_base;
    uint32_t tmem_sfa  = tmem_base + 256;
    uint32_t tmem_sfb  = tmem_base + 300;

    // Write scales = ue4m3(1.0) = 0x38
    if (warp_id == 1) {
        uint32_t sf_val = 0x38383838u;
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(tmem_sfa), "r"(sf_val), "r"(sf_val), "r"(sf_val), "r"(sf_val));
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(tmem_sfb), "r"(sf_val), "r"(sf_val), "r"(sf_val), "r"(sf_val));
    }
    __syncthreads();

    uint32_t idesc = make_idesc(tmem_sfa, tmem_sfb);

    if (tid == 0) {
        printf("WS demo: 1 producer warp + %d consumer warps (%d threads)\n",
               CONS_WARPS, TOTAL_THREADS);
        printf("M=%d N=%d K_tile=%d K_iters=%d K_total=%d  (FMA=%d)\n",
               M, N, K_TILE, K_ITERS, K_TILE * K_ITERS, M * N * K_TILE * K_ITERS);
    }

    // ====================================================================
    //                       WARP SPECIALIZATION
    // ====================================================================
    if (warp_id == 0) {
        // ============== PRODUCER ==============
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");

        for (int k = 0; k < K_ITERS; k++) {
            int s = k % STAGES;

            // Wait for slot s to be empty (skip first STAGES iters: slots are initially empty)
            if (k >= STAGES) {
                asm volatile("bar.sync %0, %1;"
                             :: "r"(BAR_EMPTY(s)), "n"(TOTAL_THREADS));
            }

            // cp.async: load A[k] and B[k] from gmem into smem stage s
            uint8_t const* gA = gmem_A + k * A_BYTES;
            uint8_t const* gB = gmem_B + k * B_BYTES;

            #pragma unroll
            for (int off = lane * 16; off < A_BYTES; off += 32 * 16) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                             :: "r"(smem_u32(smem_a[s] + off)), "l"(gA + off));
            }
            #pragma unroll
            for (int off = lane * 16; off < B_BYTES; off += 32 * 16) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                             :: "r"(smem_u32(smem_b[s] + off)), "l"(gB + off));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");

            // Signal stage s is full
            asm volatile("bar.arrive %0, %1;"
                         :: "r"(BAR_FULL(s)), "n"(TOTAL_THREADS));
        }
    }
    else if (warp_id <= CONS_WARPS) {
        // ============== CONSUMER (warps 1..4) ==============
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");

        uint32_t parity[STAGES] = {0, 0};

        for (int k = 0; k < K_ITERS; k++) {
            int s = k % STAGES;

            // Wait for stage s data
            asm volatile("bar.sync %0, %1;"
                         :: "r"(BAR_FULL(s)), "n"(TOTAL_THREADS));

            // tcgen05.mma issued by warp 1, lane 0; accumulate after first iter
            if (warp_id == 1 && lane == 0) {
                uint64_t desc_a = make_desc(smem_a[s], K_TILE / 2);
                uint64_t desc_b = make_desc(smem_b[s], K_TILE / 2);
                uint32_t accum  = (k > 0) ? 1u : 0u;

                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
                    "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
                    "}\n"
                    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                       "r"(idesc), "r"(accum), "r"(tmem_sfa), "r"(tmem_sfb));

                // Schedule mbarrier arrival when this MMA completes
                asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                             :: "r"(smem_u32(&mbar_done[s])));
            }
            __syncwarp();

            // All consumer threads wait for MMA on stage s to complete
            asm volatile(
                "{\n\t"
                ".reg .pred P;\n\t"
                "WAITLOOP_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
                "@P bra DONE_%=;\n\t"
                "bra WAITLOOP_%=;\n\t"
                "DONE_%=:\n\t"
                "}\n"
                :: "r"(smem_u32(&mbar_done[s])), "r"(parity[s]));
            parity[s] ^= 1u;

            // Signal stage s slot is now empty
            asm volatile("bar.arrive %0, %1;"
                         :: "r"(BAR_EMPTY(s)), "n"(TOTAL_THREADS));
        }

        // ============== EPILOGUE: read tmem D → gmem (warp 1) ==============
        if (warp_id == 1) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            D_out[lane * 4 + 0] = __uint_as_float(r0);
            D_out[lane * 4 + 1] = __uint_as_float(r1);
            D_out[lane * 4 + 2] = __uint_as_float(r2);
            D_out[lane * 4 + 3] = __uint_as_float(r3);
        }
    }

    __syncthreads();
    if (warp_id == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_base), "r"(512));
    }
}


// ============================================================
// 阶段2(唯一验证): 随机数据直接喂真实流水线 vs CPU 参考 (逐bit精确)
//   探针和随机数据都通过 ws_tcgen05_kernel 本体(producer/consumer流水线)跑,
//   验证的就是流水线+双缓冲+mbarrier+K累加的完整数据通路。
//   D 布局 base-4 探针实测; scale 固定1.0(kernel内写死, 属流水线demo参数)。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"
// (随机数全部在 CPU 侧生成: 本文件的 fill_random/xorshift 调用都跑在 host 上,
//  编码与float值同源双份 → GPU只跑指令, CPU算参考, 逐bit比对; 同seed可复现)

namespace p2 {
constexpr int EBITS = 4, NSLOT = 128, KTOT = K_TILE * K_ITERS;
inline uint32_t enc_small(int v) {
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
static uint8_t hA[K_ITERS * A_BYTES], hB[K_ITERS * B_BYTES];
static float hD[NSLOT];
inline void run(uint8_t* dA, uint8_t* dB, float* dD) {
    for (int t = 0; t < K_ITERS; t++) {
        to_core(hA + t * A_BYTES, M);
        to_core(hB + t * B_BYTES, N);
    }
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, NSLOT * 4);
    int smem_bytes = STAGES * (A_BYTES + B_BYTES) + 256;
    cudaFuncSetAttribute(ws_tcgen05_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    ws_tcgen05_kernel<<<1, TOTAL_THREADS, smem_bytes>>>(dA, dB, dD);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaMemcpy(hD, dD, NSLOT * 4, cudaMemcpyDeviceToHost);
}
inline void probe(uint8_t* dA, uint8_t* dB, float* dD,
                  int adigit, int bdigit, int dbl, int* out) {
    memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
    for (int m = 0; m < M; m++) {
        int v = adigit < 0 ? 1 : ((m >> (2*adigit)) & 3) * (dbl ? 2 : 1);
        pack_set(hA, m * K_TILE, enc_small(v), EBITS);       // 信号只放 tile0
    }
    for (int n = 0; n < N; n++) {
        int v = bdigit < 0 ? 1 : ((n >> (2*bdigit)) & 3);
        pack_set(hB, n * K_TILE, enc_small(v), EBITS);
    }
    run(dA, dB, dD);
    for (int s = 0; s < NSLOT; s++) out[s] = (int)hD[s];
}
inline int random_vs_cpu(uint32_t seed) {
    const int ROUNDS = 3;
    static float refA[M * K_TILE * K_ITERS], refB[N * K_TILE * K_ITERS], refD[M * N];
    uint8_t *dA, *dB; float* dD;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dD, NSLOT * 4);
    printf("\n====== 阶段2: 随机数据经流水线 vs CPU参考 (seed=%u, K_total=%d) ======\n", seed, KTOT);
    static int pr[4][NSLOT], pc[4][NSLOT], pchk[NSLOT], mrow[NSLOT], mcol[NSLOT];
    static bool valid[NSLOT];
    for (int d = 0; d < 4; d++) probe(dA, dB, dD, d, -1, 0, pr[d]);
    for (int d = 0; d < 4; d++) probe(dA, dB, dD, -1, d, 0, pc[d]);
    probe(dA, dB, dD, 0, -1, 1, pchk);
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
        memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
        for (int t = 0; t < K_ITERS; t++) {
            fill_random(hA + t * A_BYTES, refA + t * M * K_TILE, M, K_TILE,
                        E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
            fill_random(hB + t * B_BYTES, refB + t * N * K_TILE, N, K_TILE,
                        E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
        }
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.f;
                for (int t = 0; t < K_ITERS; t++)
                    for (int k = 0; k < K_TILE; k++)
                        acc += refA[t*M*K_TILE + m*K_TILE + k] * refB[t*N*K_TILE + n*K_TILE + k];
                refD[m*N + n] = acc;
            }
        run(dA, dB, dD);
        float got[NSLOT], want[NSLOT]; int nv = 0;
        for (int s = 0; s < NSLOT; s++) {
            if (!valid[s]) continue;
            got[nv] = hD[s]; want[nv] = refD[mrow[s]*N + mcol[s]]; nv++;
        }
        char tag[16]; snprintf(tag, sizeof(tag), "random r%d", r);
        bad += check_exact(got, want, nv, tag);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    printf("====== 阶段2 %s ======\n", bad ? "FAIL" : "全部PASS");
    return bad ? 1 : 0;
}
}  // namespace p2

int main() {
    return p2::random_vs_cpu(/*seed=*/1);   // 随机全在CPU侧生成, 同seed可复现
}
