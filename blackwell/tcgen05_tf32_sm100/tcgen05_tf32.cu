/**
 * tcgen05.mma 最小单元 — TF32 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::tf32
 *       [tmem_c], desc_a, desc_b, idesc, p
 *
 * TF32: 19-bit (1 sign + 8 exp + 10 mant), smem里按4字节/元素存
 *       float 1.0 = 0x3F800000 (tf32是fp32的截断, 1.0精确可表示)
 * Tile:  M=128, N=8, K=8 (K = 32字节/元素4字节), 128线程 (4 warp, cta_group::1)
 * FMA:   128 × 8 × 8 = 8,192
 *
 * 数据流 (无scale, 同fp8):
 *   1. tcgen05.alloc  → 分配TMEM (只有累加器)
 *   2. 填充smem        → A[128×8] tf32, B[8×8] tf32
 *   3. make_desc()    → 构造smem descriptor (uint64)
 *   4. tcgen05.mma    → 1个线程发射 (elect_one)
 *   5. tcgen05.ld     → 从TMEM读结果
 *   6. tcgen05.dealloc → 释放TMEM
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_tf32 tcgen05_tf32.cu
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

// InstrDescriptor for kind::tf32:
//   [4:6) d_format=1(f32), [7:10) a_format=2(TF32), [10:13) b_format=2(TF32)
//   [17:23) n_dim=N>>3, [24:29) m_dim=M>>4
__device__ uint32_t make_idesc() {
    uint32_t d = 0;
    d |= (1 << 4);                    // d_format = F32
    d |= (2 << 7);                    // a_format = TF32
    d |= (2 << 10);                   // b_format = TF32
    d |= (1 << 17);                   // n_dim = N/8 = 1
    d |= (8 << 24);                   // m_dim = M/16 = 8
    return d;
}

constexpr int M = 128, N = 8, K = 8;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K * 4;  // 4096
constexpr int B_BYTES = N * K * 4;  // 256

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 验证: 随机数据 vs CPU 参考 (逐bit精确) — host驱动统一在 verify_random.h::vr_verify()
//   本文件只保留: 被测kernel + launch适配器
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

__global__ void vk(const uint8_t* A, const uint8_t* B, uint32_t* D_raw) {
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
            "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n\t}\n"
            : : "r"(tmem_c), "l"(da), "l"(db), "r"((uint32_t)((1<<4)|(2<<7)|(2<<10)|(1<<17)|(8<<24))), "r"(0u));
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

static void vr_run(const uint8_t* hA, const uint8_t* hB,
                   const uint32_t* sfa, const uint32_t* sfb, void* out) {
    static uint8_t *dA, *dB; static uint32_t* dD;
    if (!dA) { cudaMalloc(&dA, A_BYTES); cudaMalloc(&dB, B_BYTES); cudaMalloc(&dD, 512); }
    cudaMemcpy(dA, hA, A_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_BYTES, cudaMemcpyHostToDevice);
    vk<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dD, 512, cudaMemcpyDeviceToHost);
}

int main() {
    VrSpec sp = {}; sp.M = 128; sp.N = 8; sp.K = K; sp.ebits = 32; sp.is_int = 0;
    sp.dset = TF32_SET; sp.dsetn = SETN(TF32_SET); sp.core_matrix = 1;
    sp.probe = 1; sp.nslot = 128; sp.enc_tab = VR_ENC_TF32; sp.sf_one = 0;
    return vr_verify(sp, vr_run, 1);
}
