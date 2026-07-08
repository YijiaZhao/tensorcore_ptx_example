/**
 * 最小 TF32 Tensor Core 单元 (sm_80+, 6000D=sm_120a)
 *
 * 指令: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
 *
 * 计算: D[16x8] = A[16x8] × B[8x8] + C[16x8]
 *       每条指令 16×8×8 = 1024 FMA
 *
 * ============ 数据格式 ============
 *
 * TF32 (19-bit: [sign|exp(8)|mant(10)]):
 *   fp32 的截断格式, 寄存器里按 32-bit 存 (低13位mantissa被硬件忽略)
 *   1.0f = 0x3F800000 (精确可表示, 不需要 cvt.rna.tf32.f32)
 *   一般数据需先 cvt.rna.tf32.f32 转换 (或 __float_to_tf32)
 *
 * ============ 寄存器布局 ============
 *
 * A[16×8] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}, 每寄存器 1个 tf32
 *   32线程 × 4 = 128 = 16×8 ✓
 *
 * B[8×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}, 每寄存器 1个 tf32
 *   32线程 × 2 = 64 = 8×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 float 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 float = 16×8 ✓
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_tf32 mma_tf32.cu
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
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
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
    VrSpec sp = {}; sp.M = 16; sp.N = 8; sp.K = 8; sp.ebits = 32; sp.is_int = 0;
    sp.dset = TF32_SET; sp.dsetn = SETN(TF32_SET);
    return vr_verify(sp, vr_run, 1);
}
