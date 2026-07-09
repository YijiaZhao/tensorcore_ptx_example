/**
 * 最小 INT4 Tensor Core 单元 — Blackwell 上是模拟路径!
 *
 * 指令: mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
 *
 * 计算: D[16x8] = A[16x64] × B[64x8] + C[16x8]   (s32 累加)
 *
 * ============ 重要: int4 tensor core 在 Blackwell 已移除 ============
 *
 * 实测 SASS (2026-07, CUDA 13.0):
 *   sm_80  (A100):        1× IMMA.16864.S4.S4          ← 原生 int4 IMMA
 *   sm_100a (B200):  36× IMAD.SHL + 2× IMMA.16832.S8   ← 解包成s8, 借int8单元模拟
 *   sm_120a (6000D): 同上模拟
 *
 * 也就是说 s4 在 Blackwell 只是功能兼容, 没有吞吐收益(反而多了解包开销)。
 * tcgen05 没有 kind::i4 —— 4-bit 在大卡上只有 FP4 (kind::mxf4nvf4)。
 * INT4 量化模型在 Blackwell 上应走 w4a8/w4a16 (权重解包) 或改用 nvfp4。
 *
 * ============ 数据格式 ============
 *
 * S4: 有符号4-bit整数 [-8, 7], 每byte装2个 (lower nibble=elem0)
 *   0x11 = 两个 s4(1),  uint32_t 0x11111111 = 8个 s4(1)
 *
 * ============ 寄存器布局 ============
 *
 * A[16×64] (row-major): 每线程 4× uint32_t, 每寄存器 8个 s4, 32线程×32 = 1024 ✓
 * B[64×8]  (col-major): 每线程 2× uint32_t, 32线程×16 = 512 ✓
 * C/D[16×8]: 每线程 4× s32, 32线程×4 = 128 ✓
 *
 * Compile: nvcc -gencode arch=compute_100a,code=sm_100a -o mma_int4_int8emu mma_int4_int8emu.cu
 *          (sm_75+ 都能编; sm_80/sm_89 原生, sm_90+/Blackwell 模拟)
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
// 验证: 随机数据 vs CPU 参考 (逐bit精确)
//
// 随机数在哪生成? —— 全部在 CPU 侧(不在本文件、不在GPU):
//   main 把种子交给 vr_verify()(verify_random.h), 它在 host 上用 xorshift 抽数,
//   每抽一个数同时落两份: 编码(bit形式)进 GPU buffer, float值进 CPU 参考数组
//   —— 一次抽取喂两边, 天然保证 CPU 和 GPU 算的是同一组数。
//
// 完整数据流:  [CPU]抽随机+算参考答案 → cudaMemcpy上卡 → [GPU]只跑下面这条
//   tensor core 指令(kernel里零随机逻辑) → 拷回 → [CPU]逐元素 == 比对。
//   取值集限制在 {0,±0.5,±1,±1.5,±2}: 任意累加顺序 fp32 零舍入, 所以敢用 ==。
//
// 本文件只保留: 被测kernel + launch适配器; 其余(抽数/参考/比对/探针)在公共驱动。
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
    // mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
    //   |        |       |   |    └ 操作数精度: D类型.A类型.B类型.C类型
    //   |        |       |   └ row.col: A行主 × B列主 (唯一支持的组合)
    //   |        |       └ m16n8kK: 单条指令的 tile 形状
    //   |        └ aligned: warp 32 线程必须全部活跃且执行同一条
    //   └ sync: 指令自带 warp 同步语义, 完成后寄存器立即可用
    // 操作数分组: {D 4寄存器} {A 4寄存器} {B 2寄存器} {C 4寄存器}
    // C 传 0 = 纯乘不累加; 想累加就把上一条的 D 喂回 C。
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "r"(0),"r"(0),"r"(0),"r"(0));

    // -- 写回 D: fragment 布局与 A 同族 --
    //   行 = lane/4 + (r>>1)*8, 列 = (lane%4)*2 + (r&1)  → 还原成线性 D[16][8]
    D[mma_d_idx(lane,0)]=(float)d0; D[mma_d_idx(lane,1)]=(float)d1;
    D[mma_d_idx(lane,2)]=(float)d2; D[mma_d_idx(lane,3)]=(float)d3;
}

// launch适配器: 只做 H2D搬运→启动kernel→D2H。不产生任何随机数(那是CPU侧驱动的事),
// 也不做比对 —— 保持被测路径纯净: 出错时嫌疑只剩"这条指令+这段搬运"。
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
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 64; sp.ebits = 4; sp.is_int = 1;
    sp.dset = S4_SET; sp.dsetn = SETN(S4_SET);
    return vr_verify(sp, vr_run, /*seed=*/1);   // 随机全在CPU侧生成(见verify_random.h数据流注释); 同seed可复现; 退出码0=全部逐bit相等
}
