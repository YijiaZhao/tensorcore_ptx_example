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

__global__ void wgmma_fp16_kernel(float* D_out) {
    extern __shared__ char smem[];
    uint32_t* smem_a = (uint32_t*)smem;
    uint32_t* smem_b = (uint32_t*)(smem + A_BYTES);

    int tid = threadIdx.x;

    // A=B=fp16(1.0)=0x3C00
    for (int i = tid; i < A_BYTES / 4; i += THREADS) smem_a[i] = 0x3C003C00u;
    for (int i = tid; i < B_BYTES / 4; i += THREADS) smem_b[i] = 0x3C003C00u;
    __syncthreads();
    // 普通store对异步代理(wgmma)可见
    asm volatile("fence.proxy.async.shared::cta;");

    uint64_t desc_a = make_desc_wgmma(smem_a);
    uint64_t desc_b = make_desc_wgmma(smem_b);

    // 整个warpgroup(128线程)一起发射; p=0 → 不读入D
    float d0, d1, d2, d3;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "wgmma.fence.sync.aligned;\n\t"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;\n\t"
        "wgmma.commit_group.sync.aligned;\n\t"
        "wgmma.wait_group.sync.aligned 0;\n\t"
        "}\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "l"(desc_a), "l"(desc_b), "r"(0u));

    D_out[tid * 4 + 0] = d0;
    D_out[tid * 4 + 1] = d1;
    D_out[tid * 4 + 2] = d2;
    D_out[tid * 4 + 3] = d3;
}

int main() {
    float *d_out, h[512];
    CHECK_CUDA(cudaMalloc(&d_out, 512 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_out, 0, 512 * sizeof(float)));

    printf("====== FP16 (wgmma.mma_async, sm_90a) ======\n");
    printf("A[64x16]=1.0 (smem desc), B[16x8]=1.0 (smem desc)\n");
    printf("M=%d N=%d K=%d, warpgroup=128线程, expect D=16.0\n\n", M, N, K);

    wgmma_fp16_kernel<<<1, THREADS, 8192>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 512 * sizeof(float), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 512; i++) if (h[i] == 16.0f) pass++;
        printf("Result: %d/512 correct (=16.0)\n", pass);
        printf("D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return 0;
}
