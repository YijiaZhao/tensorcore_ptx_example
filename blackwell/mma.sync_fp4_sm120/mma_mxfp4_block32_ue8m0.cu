/**
 * 最小 NVFP4 Tensor Core 单元 (sm_120, RTX 6000D)
 *
 * 指令: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X
 *       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
 *
 * 计算: D[16x8] = (sf_A * A[16x64]) × (sf_B * B[64x8]) + C[16x8]
 *       每条指令 16×8×64 = 8192 FMA
 *
 * ============ 数据格式 ============
 *
 * FP4 e2m1 (4-bit: [sign|exp(2)|mant(1)], bias=1):
 *   0x0=0    0x1=0.5   0x2=1.0   0x3=1.5
 *   0x4=2.0  0x5=3.0   0x6=4.0   0x7=6.0
 *   0x8~0xF = 对应负数
 *
 * UE8M0 scale (8-bit unsigned exponent, bias=127):
 *   value = 2^(byte - 127)
 *   0x7F=1.0  0x80=2.0  0x7E=0.5
 *
 * ============ 寄存器布局 ============
 *
 * Native FP4 packing: 每byte装2个FP4 (lower nibble=elem0, upper nibble=elem1)
 *   byte 0x22 = 两个 FP4(1.0)
 *   uint32_t 0x22222222 = 8个 FP4(1.0)
 *
 * A[16×64] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}
 *   每寄存器 4 bytes × 2 FP4/byte = 8 FP4
 *   每线程 4×8 = 32 FP4
 *   32线程 × 32 = 1024 FP4 = 16×64 ✓
 *
 * B[64×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}
 *   每寄存器 8 FP4
 *   每线程 2×8 = 16 FP4
 *   32线程 × 16 = 512 FP4 = 64×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 float 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 float = 16×8 ✓
 *
 * Scale factor (scale_vec::2X, K=64分2段):
 *   sf_A: 16-bit 寄存器, 装2个 ue8m0
 *     bits[7:0]  = K[0:31]  的 scale
 *     bits[15:8] = K[32:63] 的 scale
 *   sf_B: 同上
 *   {byte-id, thread-id}: 指定从warp中哪个线程的哪个byte读取scale
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_nvfp4 mma_mxfp4_block32_ue8m0.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

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

// ======================= 被测 kernel (block scale 版) =======================
// 与普通 mma.sync 的区别: 指令末尾多了 scale 操作数, 硬件在乘加内部
// 对 A/B 每个 K 段乘上一个 8bit scale (这是 Blackwell sm_120a 专属能力)。
__global__ void vk(const uint32_t* A, const uint32_t* B, float* D,
                   uint32_t sfa, uint32_t sfb) {
    int lane = threadIdx.x % 32;
    uint32_t a0=A[mma_a_idx(lane,0)], a1=A[mma_a_idx(lane,1)],
             a2=A[mma_a_idx(lane,2)], a3=A[mma_a_idx(lane,3)];
    uint32_t b0=B[mma_b_idx(lane,0)], b1=B[mma_b_idx(lane,1)];
    float d0,d1,d2,d3;
    // ---- 指令本体 ----
    // mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
    //   kind::mxf4nvf4/mxf8f6f4 = block scale 家族; scale_vec::NX = 每行 K 分 N 段
    //   末尾类型 ue8m0/ue4m3 = scale 的编码格式
    // 操作数: {D}{A}{B}{C} 同普通版, 之后追加
    //   {%14}=sf_A 寄存器(每段1字节, 低段在低字节)
    //   {%15,%16}={byte-id, thread-id}: 从 warp 哪个线程的哪个字节取 scale (最小例全0 → lane0 广播)
    //   {%17}{%18,%19} = sf_B 同理
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
    D[mma_d_idx(lane,0)]=d0; D[mma_d_idx(lane,1)]=d1;
    D[mma_d_idx(lane,2)]=d2; D[mma_d_idx(lane,3)]=d3;
}

// launch适配器: 只做 H2D搬运→启动kernel→D2H。不产生任何随机数(那是CPU侧驱动的事),
// 也不做比对 —— 保持被测路径纯净: 出错时嫌疑只剩"这条指令+这段搬运"。
static void vr_run(const uint8_t* hA, const uint8_t* hB,
                   const uint32_t* sfa, const uint32_t* sfb, void* out) {
    static uint8_t *dA, *dB; static float* dD;
    if (!dA) { cudaMalloc(&dA, 512); cudaMalloc(&dB, 256); cudaMalloc(&dD, 512); }
    uint32_t pa = 0, pb = 0;                    // 每段1字节, 低段在低字节
    for (int g = 0; g < 2; g++) { pa |= (sfa[g] & 0xFF) << (8*g); pb |= (sfb[g] & 0xFF) << (8*g); }
    cudaMemcpy(dA, hA, 512, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);
    vk<<<1, 32>>>((uint32_t*)dA, (uint32_t*)dB, dD, pa, pb);
    if (cudaDeviceSynchronize() != cudaSuccess) {   // 架构不匹配等错误在这里就报清楚, 不带着全零结果去比对
        printf("CUDA错误: %s (编译的-gencode和当前GPU匹配吗? 见文件头编译行)\n",
               cudaGetErrorString(cudaGetLastError()));
        exit(1);
    }
    cudaMemcpy(out, dD, 512, cudaMemcpyDeviceToHost);
}

int main() {
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 64; sp.ebits = 4;
    sp.dset = E2M1_SET; sp.dsetn = SETN(E2M1_SET);
    sp.sset = UE8M0_SET; sp.ssetn = SETN(UE8M0_SET); sp.nseg = 2; sp.sf_one = 0x7F;
    return vr_verify(sp, vr_run, /*seed=*/1);   // 随机全在CPU侧生成(见verify_random.h数据流注释); 同seed可复现
}
