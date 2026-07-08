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

__global__ void vk(const uint32_t* A, const uint32_t* B, float* D) {
    int lane = threadIdx.x % 32;
    uint32_t a0=A[mma_a_idx(lane,0)], a1=A[mma_a_idx(lane,1)],
             a2=A[mma_a_idx(lane,2)], a3=A[mma_a_idx(lane,3)];
    uint32_t b0=B[mma_b_idx(lane,0)], b1=B[mma_b_idx(lane,1)];
    float d0,d1,d2,d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));
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
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 16; sp.ebits = 16; sp.is_int = 0;
    sp.dset = BF16_SET; sp.dsetn = SETN(BF16_SET);
    return vr_verify(sp, vr_run, 1);
}
