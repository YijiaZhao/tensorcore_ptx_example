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
 * Compile: nvcc -gencode arch=compute_100a,code=sm_100a -o mma_int4 mma_int4.cu
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

__global__ void mma_int4_kernel(int32_t *D_out) {
    // A[16×64]: 全部填 s4(1), 每byte两个: 0x11
    uint32_t a0 = 0x11111111;
    uint32_t a1 = 0x11111111;
    uint32_t a2 = 0x11111111;
    uint32_t a3 = 0x11111111;

    // B[64×8]: 全部填 s4(1)
    uint32_t b0 = 0x11111111;
    uint32_t b1 = 0x11111111;

    // C[16×8] = 0 (s32)
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));

    int lane = threadIdx.x % 32;
    D_out[lane * 4 + 0] = d0;
    D_out[lane * 4 + 1] = d1;
    D_out[lane * 4 + 2] = d2;
    D_out[lane * 4 + 3] = d3;
}

int main() {
    int32_t *d_out, h[128];
    CHECK_CUDA(cudaMalloc(&d_out, 128 * sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(int32_t)));

    printf("====== INT4 s4 (mma.sync, m16n8k64) ======\n");
    printf("A[16x64]=1, B[64x8]=1, C=0, s32累加\n");
    printf("Blackwell上是模拟路径(解包+int8 IMMA), 仅功能兼容\n");
    printf("expect D = 64\n\n");

    mma_int4_kernel<<<1, 32>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(int32_t), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 64) pass++;
        printf("Result: %d/128 correct (=64)\n", pass);
        printf("D[0:8]: %d %d %d %d %d %d %d %d\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return 0;
}
