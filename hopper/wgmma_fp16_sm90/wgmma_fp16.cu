/**
 * wgmma.mma_async 最小单元 — FP16 (sm_90a, H100/H20 专属)
 *
 * 指令: wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16
 *       {d0..d3}, desc_a, desc_b, scale_d(p), scale_a, scale_b, trans_a, trans_b
 *
 * wgmma = Hopper 的异步 warpgroup MMA (Blackwell 已移除, 换成 tcgen05):
 *   - A/B 从 smem descriptor 直读 (与 tcgen05 同思路)
 *   - 累加器 D 在寄存器, 分摊在整个 warpgroup (4 warp = 128 线程) 上
 *     (tcgen05 则住 TMEM, 单线程发射; wgmma 是 128 线程一起发射)
 *   - 异步: wgmma.fence → mma_async → commit_group → wait_group
 *
 * Tile: M=64, N=8, K=16;  D[64×8] f32 = 128线程 × 4寄存器
 * SmemDescriptor (无swizzle, K-major, core matrix = 8行×16B):
 *   [0:14) addr>>4, [16:30) LBO=16B>>4=1 (K方向相邻core matrix间距)
 *   [32:46) SBO=256B>>4=16 (M/N方向间距, 8行×32B)
 *
 * 编译: nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
 *            -o wgmma_fp16 wgmma_fp16.cu
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

// Hopper wgmma SmemDescriptor: [0:14) addr>>4, [16:30) LBO>>4, [32:46) SBO>>4, [62:64) swizzle=0
// ⚠️ no-swizzle 布局是 8行×16B core-matrix 分块(不是线性行主): LBO=128B, SBO=256B (H20实测)
//    本例全1.0输入对排布不敏感; 喂真实数据时 smem 必须按 core-matrix 排, 见 random_cpu_ref/
__device__ uint64_t make_desc_wgmma(void const* smem_ptr) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    return desc;
}

constexpr int M = 64, N = 8, K = 16;
constexpr int THREADS = 128;                // 1 warpgroup
constexpr int A_BYTES = M * K * 2;          // 2048, row-major, 行=32B
constexpr int B_BYTES = K * N * 2;          // 256,  col-major, 列=32B

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 阶段2: 随机数据 vs CPU 参考 (逐bit精确)
//   smem 按 8行×16B core-matrix 排布(host侧重排), D 按 warpgroup fragment 映射还原,
//   CPU 三重循环参考, 逐元素 == —— descriptor/布局错误这里全兜住 (全1.0查不出)。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

__device__ __forceinline__ int phase2_d_pos(int tid, int r) {   // → D[64][8] 线性下标
    int lane = tid % 32, w = tid / 32;
    return (16 * w + lane / 4 + (r >> 1) * 8) * 8 + (lane % 4) * 2 + (r & 1);
}

__global__ void phase2_random_kernel(const uint8_t* A, const uint8_t* B, float* D) {
    extern __shared__ char smem[];
    uint32_t* sA = (uint32_t*)smem; uint32_t* sB = (uint32_t*)(smem + A_BYTES);
    int tid = threadIdx.x;
    for (int i = tid; i < A_BYTES / 4; i += THREADS) sA[i] = ((uint32_t*)A)[i];
    for (int i = tid; i < B_BYTES / 4; i += THREADS) sB[i] = ((uint32_t*)B)[i];
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;");
    uint64_t da = make_desc_wgmma(sA), db = make_desc_wgmma(sB);
    float d0, d1, d2, d3;
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
        "wgmma.fence.sync.aligned;\n\t"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;\n\t"
        "wgmma.commit_group.sync.aligned;\n\t"
        "wgmma.wait_group.sync.aligned 0;\n\t}\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "l"(da), "l"(db), "r"(0u));
    D[phase2_d_pos(tid,0)]=d0; D[phase2_d_pos(tid,1)]=d1;
    D[phase2_d_pos(tid,2)]=d2; D[phase2_d_pos(tid,3)]=d3;
}

// GMMA no-swizzle: (m,kb) → (m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16
static void phase2_to_core_matrix(uint8_t* buf, int rows) {
    static uint8_t tmp[64 * 32];
    memcpy(tmp, buf, rows * 32);
    for (int m = 0; m < rows; m++)
        for (int kb = 0; kb < 32; kb++)
            buf[(m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16] = tmp[m*32 + kb];
}

static int phase2_random_vs_cpu(uint32_t seed) {
    const int K = 16, EBITS = 16, ROUNDS = 3;
    static uint8_t hA[A_BYTES], hB[B_BYTES];
    static float refA[64*64], refB[8*64], refD[64*8], hD[64*8];
    uint8_t *dA, *dB; float* dD;
    CHECK_CUDA(cudaMalloc(&dA, A_BYTES));
    CHECK_CUDA(cudaMalloc(&dB, B_BYTES));
    CHECK_CUDA(cudaMalloc(&dD, sizeof(hD)));
    printf("\n====== 阶段2: 随机数据 vs CPU参考 (seed=%u) ======\n", seed);
    int bad = 0;
    for (int r = 0; r < ROUNDS; r++) {
        memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
        fill_random(hA, refA, M, K, FP16_SET, SETN(FP16_SET), EBITS, &seed);
        fill_random(hB, refB, N, K, FP16_SET, SETN(FP16_SET), EBITS, &seed);
        cpu_gemm_ref(refA, refB, refD, M, N, K);
        phase2_to_core_matrix(hA, M); phase2_to_core_matrix(hB, N);
        CHECK_CUDA(cudaMemcpy(dA, hA, A_BYTES, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, B_BYTES, cudaMemcpyHostToDevice));
        phase2_random_kernel<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));
        char tag[16]; snprintf(tag, sizeof(tag), "random r%d", r);
        bad += check_exact(hD, refD, 64*8, tag);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    printf("====== 阶段2 %s ======\n", bad ? "FAIL" : "全部PASS");
    return bad ? 1 : 0;
}

int main() {
    return phase2_random_vs_cpu(1);
}
