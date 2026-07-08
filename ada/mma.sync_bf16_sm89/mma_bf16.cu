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

__global__ void mma_bf16_kernel(float *D_out) {
    // A[16×16]: 全部填 bf16(1.0), 每寄存器2个
    uint32_t a0 = 0x3F803F80;
    uint32_t a1 = 0x3F803F80;
    uint32_t a2 = 0x3F803F80;
    uint32_t a3 = 0x3F803F80;

    // B[16×8]: 全部填 bf16(1.0)
    uint32_t b0 = 0x3F803F80;
    uint32_t b1 = 0x3F803F80;

    // C[16×8] = 0
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));

    int lane = threadIdx.x % 32;
    D_out[lane * 4 + 0] = d0;
    D_out[lane * 4 + 1] = d1;
    D_out[lane * 4 + 2] = d2;
    D_out[lane * 4 + 3] = d3;
}

int main() {
    float *d_out, h[128];
    CHECK_CUDA(cudaMalloc(&d_out, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(float)));

    printf("====== BF16 (mma.sync, m16n8k16) ======\n");
    printf("A[16x16]=1.0, B[16x8]=1.0, C=0\n");
    printf("expect D = 16.0\n\n");

    mma_bf16_kernel<<<1, 32>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 16.0f) pass++;
        printf("Result: %d/128 correct (=16.0)\n", pass);
        printf("D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return 0;
}
