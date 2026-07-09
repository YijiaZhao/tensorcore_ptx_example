/**
 * 最小 MXFP8 Tensor Core 单元 (sm_120a, 6000D) — fp8 的硬件block scale
 *
 * 指令: mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
 *       .m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
 *
 * 计算: D[16x8] = (sf_A * A[16x32]) × (sf_B * B[32x8]) + C[16x8]
 *
 * MXFP8: e4m3 data + ue8m0 scale, 每32个元素1个scale (K=32恰好1个block)
 *   scale在MMA内部硬件相乘 —— fp8唯一的硬件scale形态
 *   (Ada/Hopper无此指令; DeepSeek式fp32细粒度scale在任何卡上都只能软件promotion)
 *
 * ue8m0: 8-bit纯指数, value=2^(byte-127); 0x7F=1.0, 0x80=2.0, 0x7E=0.5
 *
 * 本例 sf_A=2.0, sf_B=1.0 → D = 2×1×32 = 64.0 (证明scale硬件生效)
 *
 * ============ 寄存器布局 ============
 *
 * A/B/C/D 与裸fp8版 (mma_fp8.cu) 完全相同:
 *   A 4×u32 (16个e4m3/线程), B 2×u32, C/D 4×f32
 * Scale (scale_vec::1X, K=32只有1段):
 *   sf_A: 32-bit寄存器只用低8位 (1个ue8m0)
 *   {byte-id, thread-id}: 指定从warp哪个线程哪个byte读scale (最小例全0)
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_mxfp8 mma_mxfp8.cu
 *          (sm_120a/121a 专属; sm_100 无此指令, B200 走 tcgen05 版)
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
    // mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
    //   kind::mxf4nvf4/mxf8f6f4 = block scale 家族; scale_vec::NX = 每行 K 分 N 段
    //   末尾类型 ue8m0/ue4m3 = scale 的编码格式
    // 操作数: {D}{A}{B}{C} 同普通版, 之后追加
    //   {%14}=sf_A 寄存器(每段1字节, 低段在低字节)
    //   {%15,%16}={byte-id, thread-id}: 从 warp 哪个线程的哪个字节取 scale (最小例全0 → lane0 广播)
    //   {%17}{%18,%19} = sf_B 同理
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
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
    for (int g = 0; g < 1; g++) { pa |= (sfa[g] & 0xFF) << (8*g); pb |= (sfb[g] & 0xFF) << (8*g); }
    cudaMemcpy(dA, hA, 512, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);
    vk<<<1, 32>>>((uint32_t*)dA, (uint32_t*)dB, dD, pa, pb);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dD, 512, cudaMemcpyDeviceToHost);
}

int main() {
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 32; sp.ebits = 8;
    sp.dset = E4M3_SET; sp.dsetn = SETN(E4M3_SET);
    sp.sset = UE8M0_SET; sp.ssetn = SETN(UE8M0_SET); sp.nseg = 1; sp.sf_one = 0x7F;
    return vr_verify(sp, vr_run, /*seed=*/1);   // 随机全在CPU侧生成(见verify_random.h数据流注释); 同seed可复现
}
