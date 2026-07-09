/**
 * 最小 FP8 Tensor Core 单元 (sm_89+, 6000D=sm_120a)
 *
 * 指令: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 *
 * 计算: D[16x8] = A[16x32] × B[32x8] + C[16x8]
 *       每条指令 16×8×32 = 4096 FMA
 *       无 block scale (对比 mma_mxfp4.cu 的 kind::mxf4nvf4.block_scale)
 *
 * ============ 数据格式 ============
 *
 * FP8 e4m3 (8-bit: [sign|exp(4)|mant(3)], bias=7):
 *   value = 2^(exp-7) × (1 + mant/8)
 *   0x38 = 1.0   0x40 = 2.0   0x30 = 0.5   0x3C = 1.5
 *   uint32_t 0x38383838 = 4个 e4m3(1.0)
 *
 * ============ 寄存器布局 ============
 *
 * A[16×32] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}
 *   每寄存器 4 bytes = 4 e4m3
 *   每线程 4×4 = 16 e4m3
 *   32线程 × 16 = 512 = 16×32 ✓
 *
 * B[32×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}
 *   每寄存器 4 e4m3
 *   32线程 × 8 = 256 = 32×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 float 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 float = 16×8 ✓
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_fp8_sm100fp16emu_sm120native mma_fp8_sm100fp16emu_sm120native.cu
 *          (sm_89/sm_90/sm_100 也支持, 改对应 -gencode 即可)
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

    float d0,d1,d2,d3;
    // ---- 指令本体 ----
    // mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    //   |        |       |   |    └ 操作数精度: D类型.A类型.B类型.C类型
    //   |        |       |   └ row.col: A行主 × B列主 (唯一支持的组合)
    //   |        |       └ m16n8kK: 单条指令的 tile 形状
    //   |        └ aligned: warp 32 线程必须全部活跃且执行同一条
    //   └ sync: 指令自带 warp 同步语义, 完成后寄存器立即可用
    // 操作数分组: {D 4寄存器} {A 4寄存器} {B 2寄存器} {C 4寄存器}
    // C 传 0 = 纯乘不累加; 想累加就把上一条的 D 喂回 C。
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));

    // -- 写回 D: fragment 布局与 A 同族 --
    //   行 = lane/4 + (r>>1)*8, 列 = (lane%4)*2 + (r&1)  → 还原成线性 D[16][8]
    D[mma_d_idx(lane,0)]=d0; D[mma_d_idx(lane,1)]=d1;
    D[mma_d_idx(lane,2)]=d2; D[mma_d_idx(lane,3)]=d3;
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
    if (cudaDeviceSynchronize() != cudaSuccess) {   // 架构不匹配等错误在这里就报清楚, 不带着全零结果去比对
        printf("CUDA错误: %s (编译的-gencode和当前GPU匹配吗? 见文件头编译行)\n",
               cudaGetErrorString(cudaGetLastError()));
        exit(1);
    }
    cudaMemcpy(out, dD, 512, cudaMemcpyDeviceToHost);
}

int main() {
    // 验证配置: 形状/元素位宽/取值集 → vr_verify 负责 随机构造→CPU参考→逐bit比对
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 32; sp.ebits = 8; sp.is_int = 0;
    sp.dset = E4M3_SET; sp.dsetn = SETN(E4M3_SET);
    return vr_verify(sp, vr_run, /*seed=*/1);   // 随机全在CPU侧生成(见verify_random.h数据流注释); 同seed可复现; 退出码0=全部逐bit相等
}
