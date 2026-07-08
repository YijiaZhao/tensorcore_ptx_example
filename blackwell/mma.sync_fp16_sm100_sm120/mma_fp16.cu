/**
 * 最小 FP16 Tensor Core 单元 (sm_80+, 6000D=sm_120a)
 *
 * 指令: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *
 * 计算: D[16x8] = A[16x16] × B[16x8] + C[16x8]
 *       每条指令 16×8×16 = 2048 FMA
 *       fp16 是最早的 tensor core 精度 (Volta sm_70 起), 所有代原生
 *
 * ============ 数据格式 ============
 *
 * FP16 (16-bit: [sign|exp(5)|mant(10)], bias=15):
 *   1.0 = 0x3C00    2.0 = 0x4000    0.5 = 0x3800
 *   uint32_t 0x3C003C00 = 2个 fp16(1.0)
 *   (对比 bf16(1.0)=0x3F80: 指数位宽不同, 别混用)
 *
 * ============ 寄存器布局 (与 mma_bf16.cu 完全相同) ============
 *
 * A[16×16] (row-major): 每线程 4× uint32_t, 每寄存器 2个 fp16, 32线程×8 = 256 ✓
 * B[16×8]  (col-major): 每线程 2× uint32_t, 32线程×4 = 128 ✓
 * C/D[16×8]: 每线程 4× float, 32线程×4 = 128 ✓
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_fp16 mma_fp16.cu
 *          (sm_70 及以上都支持, 改对应 -gencode 即可)
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

__global__ void mma_fp16_kernel(float *D_out) {
    // A[16×16]: 全部填 fp16(1.0) = 0x3C00, 每寄存器2个
    uint32_t a0 = 0x3C003C00;
    uint32_t a1 = 0x3C003C00;
    uint32_t a2 = 0x3C003C00;
    uint32_t a3 = 0x3C003C00;

    // B[16×8]: 全部填 fp16(1.0)
    uint32_t b0 = 0x3C003C00;
    uint32_t b1 = 0x3C003C00;

    // C[16×8] = 0
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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

    printf("====== FP16 (mma.sync, m16n8k16) ======\n");
    printf("A[16x16]=1.0, B[16x8]=1.0, C=0\n");
    printf("expect D = 16.0\n\n");

    mma_fp16_kernel<<<1, 32>>>(d_out);
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
