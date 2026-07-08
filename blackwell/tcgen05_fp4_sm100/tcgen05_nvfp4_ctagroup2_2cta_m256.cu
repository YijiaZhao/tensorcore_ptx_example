/*
 * Minimal tcgen05.mma cta_group::2 demo for NVFP4 block16 on Blackwell.
 *
 * Instruction:
 *   tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X
 *
 * Shape:
 *   M = 128, N = 16, K = 64
 *   cta_group::2 is a CTA pair: launch a 2-CTA cluster, 128 threads per CTA.
 *
 * Compile:
 *   nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O2 \
 *        -o tcgen05_nvfp4_ctagroup2_2cta_m256 tcgen05_nvfp4_ctagroup2_2cta_m256.cu
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::printf("CUDA error at line %d: %s\n", __LINE__,                \
                        cudaGetErrorString(err));                                \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

constexpr int M = 128;
constexpr int N = 16;
constexpr int K = 64;
constexpr int THREADS = 128;
constexpr int CTA_GROUP = 2;
constexpr int A_BYTES = M * K / 2;
constexpr int B_BYTES = N * K / 2;
constexpr int OUT_FLOATS = 128;

__device__ uint32_t smem_u32(void const* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// SmemDescriptor: [0:14) addr>>4, [16:30) stride>>4, [46:48) version=1.
__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    // ⚠️ no-swizzle 实际是 8行×16B core-matrix 分块(全1.0输入不敏感, 真实数据排布见 random_cpu_ref/)
    (void)stride_bytes;
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    desc |= static_cast<uint64_t>(1) << 46;
    return desc;
}

// InstrDescriptorBlockScaled for NVFP4: A/B=e2m1, scale=ue4m3.
__device__ uint32_t make_idesc(uint32_t tmem_sfa, uint32_t tmem_sfb) {
    uint32_t idesc = 0;
    idesc |= 1u << 7;                         // a_format = e2m1
    idesc |= 1u << 10;                        // b_format = e2m1
    idesc |= static_cast<uint32_t>(N / 8) << 17;  // n_dim = N / 8
    idesc |= static_cast<uint32_t>(M >> 7) << 27; // m_dim = M >> 7
    idesc |= ((tmem_sfa >> 30) & 3u) << 29;   // a_sf_id
    idesc |= ((tmem_sfb >> 30) & 3u) << 4;    // b_sf_id
    return idesc;
}

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 阶段2: 随机数据 + 随机scale vs CPU 参考 (逐bit精确, cta_group::2 版)
//   两个CTA各载自己的一半A(逻辑M=256, 数据互不相同), B共享;
//   D 布局用 base-4 探针在双CTA的256个读回槽位上实测 + 一致性过滤。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

namespace p2 {
constexpr int ML = 256, NN = 16, KK = 64, EBITS = 4, NSLOT = 256;
constexpr int AB = 128 * 32, BB = NN * 32;   // 每CTA的A字节 / B字节

inline uint32_t enc_small(int v) {
    static const int      vals[] = {0, 1, 2, 3, 4, 6};
    static const uint32_t encs[] = {0x0, 0x2, 0x4, 0x5, 0x6, 0x7};
    for (int i = 0; i < 6; i++) if (vals[i] == v) return encs[i];
    return 0;
}

__global__ void __cluster_dims__(2, 1, 1)
kern(const uint8_t* A, const uint8_t* B, uint32_t* D_raw, uint32_t sfa_splat, uint32_t sfb_splat) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int cta_rank = (int)cluster.block_rank();
    extern __shared__ char smem[];
    uint32_t* sA = (uint32_t*)smem;
    uint32_t* sB = (uint32_t*)(smem + AB);
    uint32_t* s_tmem = (uint32_t*)(smem + AB + BB + 64);
    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;
    const uint32_t* gA = (const uint32_t*)(A + cta_rank * AB);   // 本CTA的128行
    for (int i = tid; i < AB / 4; i += 128) sA[i] = gA[i];
    for (int i = tid; i < BB / 4; i += 128) sB[i] = ((const uint32_t*)B)[i];
    __syncthreads();
    if (warp_id == 0)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(smem_u32(s_tmem)), "r"(32));
    __syncthreads(); cluster.sync();
    uint32_t tmem_c = *s_tmem;
    uint32_t tmem_sfa = tmem_c + 16, tmem_sfb = tmem_c + 20;
    if (warp_id == 0) {
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfa), "r"(sfa_splat), "r"(sfa_splat), "r"(sfa_splat), "r"(sfa_splat));
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfb), "r"(sfb_splat), "r"(sfb_splat), "r"(sfb_splat), "r"(sfb_splat));
    }
    __syncthreads(); cluster.sync();
    uint64_t da = make_desc(sA, 32), db = make_desc(sB, 32);
    uint32_t idesc = make_idesc(tmem_sfa, tmem_sfb);
    __shared__ uint64_t mbar[1];
    if (tid == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" : : "r"(smem_u32(mbar)));
    __syncthreads(); cluster.sync();
    if (cta_rank == 0 && tid == 0) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
            : : "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(0u),
                "r"(tmem_sfa), "r"(tmem_sfb));
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
                     : : "r"(smem_u32(mbar)));
        asm volatile("{\n\t.reg .pred p;\n\t"
            "W_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
            "@p bra D_%=;\n\tbra W_%=;\n\tD_%=:\n\t}\n"
            : : "r"(smem_u32(mbar)));
    }
    __syncthreads(); cluster.sync();
    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n\t"
                     "tcgen05.wait::ld.sync.aligned;\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));
        int b = cta_rank * 128 + lane * 4;
        D_raw[b+0]=r0; D_raw[b+1]=r1; D_raw[b+2]=r2; D_raw[b+3]=r3;
    }
    __syncthreads(); cluster.sync();
    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_c), "r"(32));
}

inline void to_core(uint8_t* buf, int rows) {
    static uint8_t tmp[256 * 32];
    memcpy(tmp, buf, rows * 32);
    for (int m = 0; m < rows; m++)
        for (int kb = 0; kb < 32; kb++)
            buf[(m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16] = tmp[m*32 + kb];
}
inline float u32f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static uint8_t hA[2 * AB], hB[BB];
static uint32_t hD[NSLOT];
inline void run(uint8_t* dA, uint8_t* dB, uint32_t* dD, uint32_t sfa, uint32_t sfb) {
    to_core(hA, 128); to_core(hA + AB, 128); to_core(hB, NN);
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemset(dD, 0, sizeof(hD));
    kern<<<2, 128, AB + BB + 256>>>(dA, dB, dD, sfa, sfb);
    cudaDeviceSynchronize();
    cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);
}
inline void probe(uint8_t* dA, uint8_t* dB, uint32_t* dD,
                  int adigit, int bdigit, int dbl, int* out) {
    memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
    for (int m = 0; m < ML; m++) {
        int v = adigit < 0 ? 1 : ((m >> (2*adigit)) & 3) * (dbl ? 2 : 1);
        pack_set(hA + (m / 128) * AB, (m % 128) * KK, enc_small(v), EBITS);
    }
    for (int n = 0; n < NN; n++) {
        int v = bdigit < 0 ? 1 : ((n >> (2*bdigit)) & 3);
        pack_set(hB, n * KK, enc_small(v), EBITS);
    }
    run(dA, dB, dD, 0x38383838u, 0x38383838u);
    for (int s = 0; s < NSLOT; s++) out[s] = (int)u32f(hD[s]);
}

inline int random_vs_cpu(uint32_t seed) {
    const int ROUNDS = 3;
    static float refA[ML * KK], refB[NN * KK], refD[ML * NN];
    uint8_t *dA, *dB; uint32_t* dD;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dD, sizeof(hD));
    printf("\n====== 阶段2: 随机数据+随机scale vs CPU参考 (cta_group::2, 逻辑M=%d, seed=%u) ======\n", ML, seed);
    static int pr[4][NSLOT], pc[2][NSLOT], pchk[NSLOT], mrow[NSLOT], mcol[NSLOT];
    static bool valid[NSLOT];
    for (int d = 0; d < 4; d++) probe(dA, dB, dD, d, -1, 0, pr[d]);
    for (int d = 0; d < 2; d++) probe(dA, dB, dD, -1, d, 0, pc[d]);
    probe(dA, dB, dD, 0, -1, 1, pchk);
    int nvalid = 0; static bool seen[256][16]; memset(seen, 0, sizeof(seen));
    for (int s = 0; s < NSLOT; s++) {
        bool ok = true; int row = 0, col = 0;
        for (int d = 0; d < 4; d++) {
            if (pr[d][s] < 0 || pr[d][s] > 3) ok = false; else row += pr[d][s] << (2*d);
        }
        for (int d = 0; d < 2; d++) {
            if (pc[d][s] < 0 || pc[d][s] > 3) ok = false; else col += pc[d][s] << (2*d);
        }
        ok = ok && row < ML && col < NN && pchk[s] == 2*pr[0][s] && !seen[row][col];
        valid[s] = ok; mrow[s] = row; mcol[s] = col;
        if (ok) { seen[row][col] = true; nvalid++; }
    }
    printf("D布局探针: %d/%d 槽位一致有效\n", nvalid, NSLOT);
    if (nvalid < 64) { printf("====== 阶段2 FAIL (探针不足) ======\n"); return 1; }
    int bad = 0;
    for (int r = 0; r < ROUNDS; r++) {
        const EncVal& ea = UE4M3_SET[xorshift(&seed) % SETN(UE4M3_SET)];
        const EncVal& eb = UE4M3_SET[xorshift(&seed) % SETN(UE4M3_SET)];
        memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
        static float tmpA[128 * KK];
        for (int h = 0; h < 2; h++) {
            fill_random(hA + h * AB, tmpA, 128, KK, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
            memcpy(refA + h * 128 * KK, tmpA, sizeof(tmpA));
        }
        fill_random(hB, refB, NN, KK, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
        for (int m = 0; m < ML; m++)
            for (int n = 0; n < NN; n++) {
                float acc = 0.f;
                for (int k = 0; k < KK; k++) acc += refA[m*KK+k] * refB[n*KK+k];
                refD[m*NN+n] = ea.val * eb.val * acc;
            }
        run(dA, dB, dD, ea.enc * 0x01010101u, eb.enc * 0x01010101u);
        float got[NSLOT], want[NSLOT]; int nv = 0;
        for (int s = 0; s < NSLOT; s++) {
            if (!valid[s]) continue;
            got[nv] = u32f(hD[s]); want[nv] = refD[mrow[s]*NN + mcol[s]]; nv++;
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
    return p2::random_vs_cpu(1);
}
