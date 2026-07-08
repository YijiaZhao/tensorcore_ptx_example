/**
 * 最小 NVFP4 Tensor Core 单元 — scale_vec::4X + ue4m3 (sm_120)
 *
 * 指令: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
 *       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
 *
 * 计算: D[16x8] = (sf_A * A[16x64]) × (sf_B * B[64x8]) + C[16x8]
 *
 * ============ 与 2X 版本的区别 ============
 *
 *                    2X (mma_mxfp4.cu)         4X (本文件)
 *   scale个数         2                        4
 *   每个scale覆盖     32个FP4                   16个FP4
 *   scale类型         ue8m0 (纯指数,8bit)       ue4m3 (4bit指数+3bit尾数,8bit)
 *   scale寄存器       16-bit (2×ue8m0)          32-bit (4×ue4m3)
 *   scale精度         只能表示2的幂次            能表示 1.0~480.0 范围的非2幂值
 *
 * ============ ue4m3 编码 ============
 *
 * ue4m3: unsigned, 4-bit exponent, 3-bit mantissa, bias=7
 *   value = 2^(exp - 7) × (1 + mant/8)    (normalized, exp>0)
 *   value = 2^(-6) × (mant/8)             (subnormal, exp=0)
 *
 * 常用值:
 *   0x38 = exp=0111(7), mant=000 → 2^(7-7) × 1.0 = 1.0
 *   0x40 = exp=1000(8), mant=000 → 2^(8-7) × 1.0 = 2.0
 *   0x30 = exp=0110(6), mant=000 → 2^(6-7) × 1.0 = 0.5
 *   0x3C = exp=0111(7), mant=100 → 2^0 × 1.5 = 1.5
 *
 * ============ 寄存器布局 ============
 *
 * A[16×64], B[64×8]: 同2X版本，native FP4 packing (2 nibble/byte)
 *
 * Scale factor (scale_vec::4X, K=64分4段):
 *   sf_A: 32-bit 寄存器, 装4个 ue4m3 (每个8-bit)
 *     byte[0] = K[0:15]   的 scale
 *     byte[1] = K[16:31]  的 scale
 *     byte[2] = K[32:47]  的 scale
 *     byte[3] = K[48:63]  的 scale
 *   sf_B: 同上
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_nvfp4_block16_ue4m3 mma_nvfp4_block16_ue4m3.cu
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
// 验证: 随机数据 vs CPU 参考 (逐bit精确) — host驱动统一在 verify_random.h::vr_verify()
//   本文件只保留: 被测kernel + launch适配器
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
    // mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
    //   kind::mxf4nvf4/mxf8f6f4 = block scale 家族; scale_vec::NX = 每行 K 分 N 段
    //   末尾类型 ue8m0/ue4m3 = scale 的编码格式
    // 操作数: {D}{A}{B}{C} 同普通版, 之后追加
    //   {%14}=sf_A 寄存器(每段1字节, 低段在低字节)
    //   {%15,%16}={byte-id, thread-id}: 从 warp 哪个线程的哪个字节取 scale (最小例全0 → lane0 广播)
    //   {%17}{%18,%19} = sf_B 同理
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
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

static void vr_run(const uint8_t* hA, const uint8_t* hB,
                   const uint32_t* sfa, const uint32_t* sfb, void* out) {
    static uint8_t *dA, *dB; static float* dD;
    if (!dA) { cudaMalloc(&dA, 512); cudaMalloc(&dB, 256); cudaMalloc(&dD, 512); }
    uint32_t pa = 0, pb = 0;                    // 每段1字节, 低段在低字节
    for (int g = 0; g < 4; g++) { pa |= (sfa[g] & 0xFF) << (8*g); pb |= (sfb[g] & 0xFF) << (8*g); }
    cudaMemcpy(dA, hA, 512, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);
    vk<<<1, 32>>>((uint32_t*)dA, (uint32_t*)dB, dD, pa, pb);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dD, 512, cudaMemcpyDeviceToHost);
}

int main() {
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 64; sp.ebits = 4;
    sp.dset = E2M1_SET; sp.dsetn = SETN(E2M1_SET);
    sp.sset = UE4M3_SET; sp.ssetn = SETN(UE4M3_SET); sp.nseg = 4; sp.sf_one = 0x38;
    return vr_verify(sp, vr_run, 1);
}
