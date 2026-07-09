/**
 * 最小 INT8 Tensor Core 单元 (sm_80+, 6000D=sm_120a)
 *
 * 指令: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
 *
 * 计算: D[16x8] = A[16x32] × B[32x8] + C[16x8]   (s32 累加)
 *       每条指令 16×8×32 = 4096 IMAD
 *
 * ============ 数据格式 ============
 *
 * S8: 有符号8-bit整数 [-128, 127]
 *   uint32_t 0x01010101 = 4个 s8(1)
 *   累加器/输出是 s32 (不是float!)
 *
 * ============ 寄存器布局 ============
 *
 * A[16×32] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}, 每寄存器 4个 s8
 *   32线程 × 16 = 512 = 16×32 ✓
 *
 * B[32×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}, 每寄存器 4个 s8
 *   32线程 × 8 = 256 = 32×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 s32 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 s32 = 16×8 ✓
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_int8 mma_int8.cu
 *          (sm_80 及以上都支持, 改对应 -gencode 即可)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                  \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                       \
            exit(1);                                               \
        }                                                          \
    } while (0)

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 验证: 随机数据 vs CPU 参考 (逐bit精确) — host驱动统一在 verify_random.h::vr_verify()
//   本文件只保留: 被测kernel + launch适配器
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

// ======================= 被测 kernel =======================
// mma.sync 执行模型: 一个 warp(32线程) 同步协作完成一条 16×8×K 的矩阵乘加,
// A/B/C/D 全部住在寄存器里, 按 PTX 规定的 fragment 布局分摊到 32 个 lane 上。
__global__ void vk(const uint32_t* A, const uint32_t* B, float* D) {
    int lane = threadIdx.x % 32;

    // -- 装载 A fragment: 每 lane 4 个 u32 寄存器 --
    // PTX row.col 布局(mma_a_idx 实现): 寄存器 r 拿的是
    //   行 = lane/4 + (r&1)*8   (每 4 个 lane 一组管一行, r 的奇偶位选上/下半 8 行)
    //   列 = 行内第 lane%4 + (r>>1)*4 个 32bit 块 (r 的高位选 K 的前/后半段)
    uint32_t a0=A[mma_a_idx(lane,0)], a1=A[mma_a_idx(lane,1)],
             a2=A[mma_a_idx(lane,2)], a3=A[mma_a_idx(lane,3)];
    // -- 装载 B fragment: 每 lane 2 个 u32 (B 是 col-major, n = lane/4) --
    uint32_t b0=B[mma_b_idx(lane,0)], b1=B[mma_b_idx(lane,1)];

    int32_t d0,d1,d2,d3;
    // ---- 指令本体 ----
    // mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    //   |        |       |   |    └ 操作数精度: D类型.A类型.B类型.C类型
    //   |        |       |   └ row.col: A行主 × B列主 (唯一支持的组合)
    //   |        |       └ m16n8kK: 单条指令的 tile 形状
    //   |        └ aligned: warp 32 线程必须全部活跃且执行同一条
    //   └ sync: 指令自带 warp 同步语义, 完成后寄存器立即可用
    // 操作数分组: {D 4寄存器} {A 4寄存器} {B 2寄存器} {C 4寄存器}
    // C 传 0 = 纯乘不累加; 想累加就把上一条的 D 喂回 C。
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "r"(0),"r"(0),"r"(0),"r"(0));

    // -- 写回 D: fragment 布局与 A 同族 --
    //   行 = lane/4 + (r>>1)*8, 列 = (lane%4)*2 + (r&1)  → 还原成线性 D[16][8]
    D[mma_d_idx(lane,0)]=(float)d0; D[mma_d_idx(lane,1)]=(float)d1;
    D[mma_d_idx(lane,2)]=(float)d2; D[mma_d_idx(lane,3)]=(float)d3;
}

static void vr_run(const uint8_t* hA, const uint8_t* hB,
                   const uint32_t*, const uint32_t*, void* out) {
    static uint8_t *dA, *dB; static float* dD;
    if (!dA) { cudaMalloc(&dA, 512); cudaMalloc(&dB, 256); cudaMalloc(&dD, 512); }
    cudaMemcpy(dA, hA, 512, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);
    vk<<<1, 32>>>((uint32_t*)dA, (uint32_t*)dB, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dD, 512, cudaMemcpyDeviceToHost);
}

int main() {
    // 验证配置: 形状/元素位宽/取值集 → vr_verify 负责 随机构造→CPU参考→逐bit比对
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 32; sp.ebits = 8; sp.is_int = 1;
    sp.dset = S8_SET; sp.dsetn = SETN(S8_SET);
    return vr_verify(sp, vr_run, /*seed=*/1);   // 随机全在CPU侧生成(见verify_random.h数据流注释); 同seed可复现; 退出码0=全部逐bit相等
}
