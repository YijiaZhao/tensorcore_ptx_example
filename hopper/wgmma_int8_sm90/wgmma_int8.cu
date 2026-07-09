/**
 * wgmma.mma_async 最小单元 — INT8 s8 (sm_90a, H100/H20 专属)
 *
 * 指令: wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8
 *       {d0..d3}, desc_a, desc_b, scale_d(p)  (整数wgmma无scale_a/b/trans参数)
 *
 * wgmma = Hopper 的异步 warpgroup MMA (Blackwell 已移除, 换成 tcgen05):
 *   - A/B 从 smem descriptor 直读 (与 tcgen05 同思路)
 *   - 累加器 D 在寄存器, 分摊在整个 warpgroup (4 warp = 128 线程) 上
 *     (tcgen05 则住 TMEM, 单线程发射; wgmma 是 128 线程一起发射)
 *   - 异步: wgmma.fence → mma_async → commit_group → wait_group
 *
 * Tile: M=64, N=8, K=32;  D[64×8] s32 = 128线程 × 4寄存器
 * SmemDescriptor (无swizzle, K-major, core matrix = 8行×16B):
 *   [0:14) addr>>4, [16:30) LBO=16B>>4=1 (K方向相邻core matrix间距)
 *   [32:46) SBO=256B>>4=16 (M/N方向间距, 8行×32B)
 *
 * 编译: nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
 *            -o wgmma_int8 wgmma_int8.cu
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

constexpr int M = 64, N = 8, K = 32;
constexpr int THREADS = 128;                // 1 warpgroup
constexpr int A_BYTES = M * K;              // 2048, row-major, 行=32B
constexpr int B_BYTES = K * N;              // 256,  col-major, 列=32B

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 验证: 随机数据 vs CPU 参考 (逐bit精确) — host驱动统一在 verify_random.h::vr_verify()
//   本文件只保留: 被测kernel + launch适配器
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

__device__ __forceinline__ int vr_d_pos(int tid, int r) {
    int lane = tid % 32, w = tid / 32;
    return (16 * w + lane / 4 + (r >> 1) * 8) * 8 + (lane % 4) * 2 + (r & 1);
}
// ======================= 被测 kernel =======================
// wgmma 执行模型 (Hopper 专属, Blackwell 已被 tcgen05 取代):
//   - 发射单位是 warpgroup = 4 个连续 warp = 128 线程, 全体执行同一条指令
//   - A/B 不占寄存器: 硬件按 smem descriptor 直接从 shared memory 读
//   - D 在寄存器, 分摊在整个 warpgroup 上; 指令是异步的, 要显式等完成
__global__ void vk(const uint8_t* A, const uint8_t* B, float* D) {
    extern __shared__ char smem[];
    uint32_t* sA = (uint32_t*)smem; uint32_t* sB = (uint32_t*)(smem + A_BYTES);
    int tid = threadIdx.x;
    // 数据搬进 smem (真实kernel这里是 cp.async/TMA; 注意布局必须是 core-matrix 分块)
    for (int i = tid; i < A_BYTES / 4; i += THREADS) sA[i] = ((uint32_t*)A)[i];
    for (int i = tid; i < B_BYTES / 4; i += THREADS) sB[i] = ((uint32_t*)B)[i];
    __syncthreads();
    // 普通 st 写的 smem 要对"异步代理"(wgmma引擎)可见, 必须过这道 fence
    asm volatile("fence.proxy.async.shared::cta;");
    // smem descriptor: 64bit 编码 {基址>>4, LBO=K方向core-matrix间距, SBO=M/N方向间距}
    uint64_t da = make_desc_wgmma(sA), db = make_desc_wgmma(sB);
    int32_t d0, d1, d2, d3;
    // ---- 异步指令序列(固定四段, 顺序不可少) ----
    //   wgmma.fence        : 保证之前对累加器寄存器的读写已就绪
    //   wgmma.mma_async    : 发射, 立即返回(不等算完!)
    //     操作数: {D 4寄存器}, desc_A, desc_B, scale_d(p: 0=不读入C), 
    //             之后是立即数 scale_a, scale_b(1/-1 取负), trans_a, trans_b(f16/bf16才有)
    //   wgmma.commit_group : 把已发射的 wgmma 打包成一组
    //   wgmma.wait_group 0 : 阻塞到"未完成组数 ≤0", 即本组算完, D 寄存器才可读
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
        "wgmma.fence.sync.aligned;\n\t"
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 {%0,%1,%2,%3}, %4, %5, p;\n\t"
        "wgmma.commit_group.sync.aligned;\n\t"
        "wgmma.wait_group.sync.aligned 0;\n\t}\n"
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        : "l"(da), "l"(db), "r"(0u));
    D[vr_d_pos(tid,0)]=(float)d0; D[vr_d_pos(tid,1)]=(float)d1;
    D[vr_d_pos(tid,2)]=(float)d2; D[vr_d_pos(tid,3)]=(float)d3;
}

static void vr_run(const uint8_t* hA, const uint8_t* hB,
                   const uint32_t*, const uint32_t*, void* out) {
    static uint8_t *dA, *dB; static float* dD;
    if (!dA) { cudaMalloc(&dA, A_BYTES); cudaMalloc(&dB, B_BYTES); cudaMalloc(&dD, 2048); }
    cudaMemcpy(dA, hA, A_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_BYTES, cudaMemcpyHostToDevice);
    vk<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dD, 2048, cudaMemcpyDeviceToHost);
}

int main() {
    VrSpec sp = {}; sp.M = 64; sp.N = 8; sp.K = 32; sp.ebits = 8; sp.is_int = 1;
    sp.dset = S8_SET; sp.dsetn = SETN(S8_SET); sp.core_matrix = 1;
    return vr_verify(sp, vr_run, /*seed=*/1);   // 随机全在CPU侧生成(见verify_random.h数据流注释); 同seed可复现
}
