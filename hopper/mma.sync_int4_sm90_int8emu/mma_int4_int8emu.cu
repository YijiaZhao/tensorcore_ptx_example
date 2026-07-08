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
 * Compile: nvcc -gencode arch=compute_90a,code=sm_90a -o mma_int4_int8emu mma_int4_int8emu.cu
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
    int32_t d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "r"(0),"r"(0),"r"(0),"r"(0));
    D[mma_d_idx(lane,0)]=(float)d0; D[mma_d_idx(lane,1)]=(float)d1;
    D[mma_d_idx(lane,2)]=(float)d2; D[mma_d_idx(lane,3)]=(float)d3;
}

static int phase2_random_vs_cpu(uint32_t seed) {
    const int M = 16, N = 8, K = 64, EBITS = 4, ROUNDS = 3;
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
        fill_random(hA, refA, M, K, S4_SET, SETN(S4_SET), EBITS, &seed);
        fill_random(hB, refB, N, K, S4_SET, SETN(S4_SET), EBITS, &seed);
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
