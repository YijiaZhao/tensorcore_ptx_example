/**
 * 最小 FP8 Tensor Core 单元 (sm_89+, H100/H20=sm_90a)
 *
 * ⚠️ sm_90 上此指令是 fp16 模拟 (F2FP.F16.E4M3 + 2×HMMA.16816), 吞吐=fp16;
 *    Hopper 原生 fp8 tensor core 只在 wgmma (见 ../wgmma_fp8_sm90/)
 *
 * 指令: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 *
 * 计算: D[16x8] = A[16x32] × B[32x8] + C[16x8]
 *       每条指令 16×8×32 = 4096 FMA
 *       无 block scale (对比 mma_mxfp4.cu 的 kind::mxf4nvf4.block_scale)
 *
 * ============ 数据格式 ============
 *
 * FP8 e4m3 (8-bit: [sign|exp(4)|mant(3)], bias=7):
 *   value = 2^(exp-7) × (1 + mant/8)
 *   0x38 = 1.0   0x40 = 2.0   0x30 = 0.5   0x3C = 1.5
 *   uint32_t 0x38383838 = 4个 e4m3(1.0)
 *
 * ============ 寄存器布局 ============
 *
 * A[16×32] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}
 *   每寄存器 4 bytes = 4 e4m3
 *   每线程 4×4 = 16 e4m3
 *   32线程 × 16 = 512 = 16×32 ✓
 *
 * B[32×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}
 *   每寄存器 4 e4m3
 *   32线程 × 8 = 256 = 32×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 float 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 float = 16×8 ✓
 *
 * Compile: nvcc -gencode arch=compute_90a,code=sm_90a -o mma_fp8_fp16emu mma_fp8_fp16emu.cu
 *          (sm_89/sm_90/sm_100 也支持, 改对应 -gencode 即可)
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

__global__ void mma_fp8_kernel(float *D_out) {
    // A[16×32]: 全部填 e4m3(1.0) = 0x38
    uint32_t a0 = 0x38383838;
    uint32_t a1 = 0x38383838;
    uint32_t a2 = 0x38383838;
    uint32_t a3 = 0x38383838;

    // B[32×8]: 全部填 e4m3(1.0)
    uint32_t b0 = 0x38383838;
    uint32_t b1 = 0x38383838;

    // C[16×8] = 0
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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

    printf("====== FP8 e4m3 (mma.sync, m16n8k32) ======\n");
    printf("A[16x32]=1.0, B[32x8]=1.0, C=0\n");
    printf("expect D = 32.0\n\n");

    mma_fp8_kernel<<<1, 32>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 32.0f) pass++;
        printf("Result: %d/128 correct (=32.0)\n", pass);
        printf("D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return 0;
}
