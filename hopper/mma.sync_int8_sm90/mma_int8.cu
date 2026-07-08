/**
 * 最小 INT8 Tensor Core 单元 (sm_80+, H100/H20=sm_90a)
 *
 * 指令: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
 *
 * 计算: D[16x8] = A[16x32] × B[32x8] + C[16x8]   (s32 累加)
 *       每条指令 16×8×32 = 4096 IMAD
 *
 * ============ 数据格式 ============
 *
 * S8: 有符号8-bit整数 [-128, 127]
 *   uint32_t 0x01010101 = 4个 s8(1)
 *   累加器/输出是 s32 (不是float!)
 *
 * ============ 寄存器布局 ============
 *
 * A[16×32] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}, 每寄存器 4个 s8
 *   32线程 × 16 = 512 = 16×32 ✓
 *
 * B[32×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}, 每寄存器 4个 s8
 *   32线程 × 8 = 256 = 32×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 s32 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 s32 = 16×8 ✓
 *
 * Compile: nvcc -gencode arch=compute_90a,code=sm_90a -o mma_int8 mma_int8.cu
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

__global__ void mma_int8_kernel(int32_t *D_out) {
    // A[16×32]: 全部填 s8(1) = 0x01
    uint32_t a0 = 0x01010101;
    uint32_t a1 = 0x01010101;
    uint32_t a2 = 0x01010101;
    uint32_t a3 = 0x01010101;

    // B[32×8]: 全部填 s8(1)
    uint32_t b0 = 0x01010101;
    uint32_t b1 = 0x01010101;

    // C[16×8] = 0 (s32)
    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int32_t d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
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

    printf("====== INT8 s8 (mma.sync, m16n8k32) ======\n");
    printf("A[16x32]=1, B[32x8]=1, C=0, s32累加\n");
    printf("expect D = 32\n\n");

    mma_int8_kernel<<<1, 32>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(int32_t), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 32) pass++;
        printf("Result: %d/128 correct (=32)\n", pass);
        printf("D[0:8]: %d %d %d %d %d %d %d %d\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return 0;
}
