/**
 * 最小 BF16 Tensor Core 单元 (sm_80+, 6000D=sm_120a)
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
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_bf16 mma_bf16.cu
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
// 阶段2: 随机数据 vs CPU 参考 (逐bit精确)
//   全1.0输入对布局不敏感(排乱了结果也"对"), 这里每元素独立随机,
//   CPU 三重循环算参考, 逐元素 == 比对 —— fragment 映射/格式解码错误全兜住。
//   取值集{0,±0.5,±1,±1.5,±2}保证任意累加顺序 fp32 精确, 允许用 == 而非容差。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

__global__ void phase2_random_kernel(const uint32_t* A, const uint32_t* B, float* D) {
    int lane = threadIdx.x % 32;
    uint32_t a0=A[mma_a_idx(lane,0)], a1=A[mma_a_idx(lane,1)],
             a2=A[mma_a_idx(lane,2)], a3=A[mma_a_idx(lane,3)];
    uint32_t b0=B[mma_b_idx(lane,0)], b1=B[mma_b_idx(lane,1)];
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));
    D[mma_d_idx(lane,0)]=d0; D[mma_d_idx(lane,1)]=d1;
    D[mma_d_idx(lane,2)]=d2; D[mma_d_idx(lane,3)]=d3;
}

static int phase2_random_vs_cpu(uint32_t seed) {
    const int M = 16, N = 8, K = 16, EBITS = 16, ROUNDS = 3;
    static uint8_t hA[16*32], hB[8*32];
    static float refA[16*64], refB[8*64], refD[128], hD[128];
    uint8_t *dA, *dB; float* dD;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(hA)));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(hB)));
    CHECK_CUDA(cudaMalloc(&dD, sizeof(hD)));
    printf("\n====== 阶段2: 随机数据 vs CPU参考 (seed=%u) ======\n", seed);
    int bad = 0;
    for (int r = 0; r < ROUNDS; r++) {
        memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
        fill_random(hA, refA, M, K, BF16_SET, SETN(BF16_SET), EBITS, &seed);
        fill_random(hB, refB, N, K, BF16_SET, SETN(BF16_SET), EBITS, &seed);
        cpu_gemm_ref(refA, refB, refD, M, N, K);
        CHECK_CUDA(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
        phase2_random_kernel<<<1, 32>>>((uint32_t*)dA, (uint32_t*)dB, dD);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));
        char tag[16]; snprintf(tag, sizeof(tag), "random r%d", r);
        bad += check_exact(hD, refD, 128, tag);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    printf("====== 阶段2 %s ======\n", bad ? "FAIL" : "全部PASS");
    return bad ? 1 : 0;
}

int main() {
    return phase2_random_vs_cpu(1);
}
