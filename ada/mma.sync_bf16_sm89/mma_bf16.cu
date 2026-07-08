/**
 * 最小 BF16 Tensor Core 单元 (sm_80+, Ada L40S/RTX40=sm_89)
 *
 * 指令: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 *
 * 计算: D[16x8] = A[16x16] × B[16x8] + C[16x8]
 *       每条指令 16×8×16 = 2048 FMA
 *       (纯SASS对比用的stub见 bf16_mma_sass.cu, 本文件是可跑的完整验证)
 *
 * ============ 数据格式 ============
 *
 * BF16 (16-bit: [sign|exp(8)|mant(7)]):
 *   fp32 的高16位截断, 动态范围与fp32相同
 *   1.0 = 0x3F80    2.0 = 0x4000    0.5 = 0x3F00
 *   uint32_t 0x3F803F80 = 2个 bf16(1.0)
 *
 * ============ 寄存器布局 ============
 *
 * A[16×16] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}, 每寄存器 2个 bf16
 *   32线程 × 8 = 256 = 16×16 ✓
 *
 * B[16×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}, 每寄存器 2个 bf16
 *   32线程 × 4 = 128 = 16×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 float 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 float = 16×8 ✓
 *
 * Compile: nvcc -gencode arch=compute_89,code=sm_89 -o mma_bf16 mma_bf16.cu
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

    float d0,d1,d2,d3;
    // ---- 指令本体 ----
    // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    //   |        |       |   |    └ 操作数精度: D类型.A类型.B类型.C类型
    //   |        |       |   └ row.col: A行主 × B列主 (唯一支持的组合)
    //   |        |       └ m16n8kK: 单条指令的 tile 形状
    //   |        └ aligned: warp 32 线程必须全部活跃且执行同一条
    //   └ sync: 指令自带 warp 同步语义, 完成后寄存器立即可用
    // 操作数分组: {D 4寄存器} {A 4寄存器} {B 2寄存器} {C 4寄存器}
    // C 传 0 = 纯乘不累加; 想累加就把上一条的 D 喂回 C。
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));

    // -- 写回 D: fragment 布局与 A 同族 --
    //   行 = lane/4 + (r>>1)*8, 列 = (lane%4)*2 + (r&1)  → 还原成线性 D[16][8]
    D[mma_d_idx(lane,0)]=d0; D[mma_d_idx(lane,1)]=d1;
    D[mma_d_idx(lane,2)]=d2; D[mma_d_idx(lane,3)]=d3;
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
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 16; sp.ebits = 16; sp.is_int = 0;
    sp.dset = BF16_SET; sp.dsetn = SETN(BF16_SET);
    return vr_verify(sp, vr_run, 1);   // 退出码 0 = 全部逐bit相等
}
