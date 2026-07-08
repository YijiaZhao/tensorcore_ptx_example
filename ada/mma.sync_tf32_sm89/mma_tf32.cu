/**
 * 最小 TF32 Tensor Core 单元 (sm_80+, Ada L40S/RTX40=sm_89)
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
 * Compile: nvcc -gencode arch=compute_89,code=sm_89 -o mma_tf32 mma_tf32.cu
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

__global__ void mma_tf32_kernel(float *D_out) {
    // A[16×8]: 全部填 tf32(1.0), 每寄存器1个元素
    uint32_t a0 = 0x3F800000;
    uint32_t a1 = 0x3F800000;
    uint32_t a2 = 0x3F800000;
    uint32_t a3 = 0x3F800000;

    // B[8×8]: 全部填 tf32(1.0)
    uint32_t b0 = 0x3F800000;
    uint32_t b1 = 0x3F800000;

    // C[16×8] = 0
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
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

    printf("====== TF32 (mma.sync, m16n8k8) ======\n");
    printf("A[16x8]=1.0, B[8x8]=1.0, C=0\n");
    printf("expect D = 8.0\n\n");

    mma_tf32_kernel<<<1, 32>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 8.0f) pass++;
        printf("Result: %d/128 correct (=8.0)\n", pass);
        printf("D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return 0;
}
