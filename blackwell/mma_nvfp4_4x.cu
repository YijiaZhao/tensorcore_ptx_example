/**
 * 最小 NVFP4 Tensor Core 单元 — scale_vec::4X + ue4m3 (sm_120)
 *
 * 指令: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
 *       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
 *
 * 计算: D[16x8] = (sf_A * A[16x64]) × (sf_B * B[64x8]) + C[16x8]
 *
 * ============ 与 2X 版本的区别 ============
 *
 *                    2X (mma_mxfp4.cu)         4X (本文件)
 *   scale个数         2                        4
 *   每个scale覆盖     32个FP4                   16个FP4
 *   scale类型         ue8m0 (纯指数,8bit)       ue4m3 (4bit指数+3bit尾数,8bit)
 *   scale寄存器       16-bit (2×ue8m0)          32-bit (4×ue4m3)
 *   scale精度         只能表示2的幂次            能表示 1.0~480.0 范围的非2幂值
 *
 * ============ ue4m3 编码 ============
 *
 * ue4m3: unsigned, 4-bit exponent, 3-bit mantissa, bias=7
 *   value = 2^(exp - 7) × (1 + mant/8)    (normalized, exp>0)
 *   value = 2^(-6) × (mant/8)             (subnormal, exp=0)
 *
 * 常用值:
 *   0x38 = exp=0111(7), mant=000 → 2^(7-7) × 1.0 = 1.0
 *   0x40 = exp=1000(8), mant=000 → 2^(8-7) × 1.0 = 2.0
 *   0x30 = exp=0110(6), mant=000 → 2^(6-7) × 1.0 = 0.5
 *   0x3C = exp=0111(7), mant=100 → 2^0 × 1.5 = 1.5
 *
 * ============ 寄存器布局 ============
 *
 * A[16×64], B[64×8]: 同2X版本，native FP4 packing (2 nibble/byte)
 *
 * Scale factor (scale_vec::4X, K=64分4段):
 *   sf_A: 32-bit 寄存器, 装4个 ue4m3 (每个8-bit)
 *     byte[0] = K[0:15]   的 scale
 *     byte[1] = K[16:31]  的 scale
 *     byte[2] = K[32:47]  的 scale
 *     byte[3] = K[48:63]  的 scale
 *   sf_B: 同上
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_nvfp4_4x mma_nvfp4_4x.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                  \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                       \
            exit(1);                                               \
        }                                                          \
    } while (0)

__global__ void mma_nvfp4_4x_kernel(uint32_t a_fill, uint32_t b_fill,
                                     uint32_t sf_A, uint32_t sf_B,
                                     float *D_out) {
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"    // D输出: 4×f32
        "{%4,  %5,  %6,  %7},"    // A数据: 4×uint32 (每个装8个FP4)
        "{%8,  %9},"              // B数据: 2×uint32 (每个装8个FP4)
        "{%10, %11, %12, %13},"   // C累加器: 4×f32
        "{%14},"                   // sf_A: 32-bit (4个ue4m3)
        "{%15, %16},"             // {byte-id-a=0, thread-id-a=0}
        "{%17},"                   // sf_B: 32-bit (4个ue4m3)
        "{%18, %19};\n"           // {byte-id-b=0, thread-id-b=0}
        :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
        :   "r"(a_fill), "r"(a_fill), "r"(a_fill), "r"(a_fill),
            "r"(b_fill), "r"(b_fill),
            "f"(c0), "f"(c1), "f"(c2), "f"(c3),
            "r"(sf_A), "h"((uint16_t)0), "h"((uint16_t)0),
            "r"(sf_B), "h"((uint16_t)0), "h"((uint16_t)0)
    );

    int tid = threadIdx.x;
    D_out[tid * 4 + 0] = d0;
    D_out[tid * 4 + 1] = d1;
    D_out[tid * 4 + 2] = d2;
    D_out[tid * 4 + 3] = d3;
}

// ue4m3 decode: unsigned, exp_bits=4, mant_bits=3, bias=7
float ue4m3_to_float(uint8_t val) {
    int exp = (val >> 3) & 0xF;
    int mant = val & 0x7;
    if (exp == 0) {
        return ldexpf((float)mant / 8.0f, -6);  // subnormal
    }
    return ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
}

float fp4_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

int main() {
    float *d_out, h_out[128];
    CHECK_CUDA(cudaMalloc(&d_out, 32 * 4 * sizeof(float)));

    printf("====== NVFP4 Tensor Core scale_vec::4X + ue4m3 (sm_120) ======\n");
    printf("D[16x8] = (sf_A * A[16x64]) x (sf_B * B[64x8]) + C[16x8]\n");
    printf("4个ue4m3 scale, 每个覆盖K维度的16个FP4元素\n\n");

    //=================================================================
    // 验证1: 基本正确性
    //   A=B=全1.0, sf_A=sf_B=全1.0
    //   ue4m3(1.0) = 0x38: exp=0111(7), mant=000 → 2^(7-7)×1.0 = 1.0
    //   sf寄存器 = 4个0x38 = 0x38383838
    //   D = 64 × 1.0 × 1.0 × 1.0 × 1.0 = 64.0
    //=================================================================
    printf("--- 验证1: A=B=fp4(1.0), sf_A=sf_B=ue4m3(1.0) ---\n");
    printf("  a0~a3 = 0x22222222 (8个e2m1=1.0)\n");
    printf("  b0~b1 = 0x22222222\n");
    printf("  sf_A  = 0x38383838 (4个ue4m3=1.0, 覆盖K[0:15],[16:31],[32:47],[48:63])\n");
    printf("  sf_B  = 0x38383838\n");

    mma_nvfp4_4x_kernel<<<1, 32>>>(0x22222222, 0x22222222,
                                    0x38383838, 0x38383838, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  结果 T00: %.1f %.1f %.1f %.1f  (期望 64.0)\n\n",
           h_out[0], h_out[1], h_out[2], h_out[3]);

    //=================================================================
    // 验证2: FP4 e2m1 全部8个正值
    //=================================================================
    printf("--- 验证2: 遍历FP4 e2m1正值, sf=1.0 ---\n");
    printf("  %-6s %-6s %-10s %-10s %-6s\n", "nibble", "e2m1", "实测", "期望", "");

    for (int nib = 0; nib <= 7; nib++) {
        uint8_t byte_val = (nib << 4) | nib;
        uint32_t a_fill = (uint32_t)byte_val | ((uint32_t)byte_val << 8) |
                          ((uint32_t)byte_val << 16) | ((uint32_t)byte_val << 24);

        mma_nvfp4_4x_kernel<<<1, 32>>>(a_fill, 0x22222222,
                                        0x38383838, 0x38383838, d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        float expected = 64.0f * fp4_table[nib];
        printf("  0x%X     %4.1f   %8.1f   %8.1f   %s\n",
               nib, fp4_table[nib], h_out[0], expected,
               (fabsf(h_out[0] - expected) < 0.5f) ? "PASS" : "FAIL");
    }

    //=================================================================
    // 验证3: ue4m3 scale 缩放效果
    //   ue4m3 能表示非2幂值 (相比ue8m0只能2的幂)
    //=================================================================
    printf("\n--- 验证3: ue4m3 scale 缩放 ---\n");
    printf("  A=B=fp4(1.0), sf_B=1.0, sf_A变化\n");
    printf("  %-8s %-8s %-10s %-10s %-6s\n", "ue4m3", "scale", "实测", "期望", "");

    uint8_t scale_tests[] = {
        0x30,  // 2^(6-7)×1.0 = 0.5
        0x38,  // 2^(7-7)×1.0 = 1.0
        0x3C,  // 2^(7-7)×1.5 = 1.5  ← ue8m0做不到!
        0x40,  // 2^(8-7)×1.0 = 2.0
        0x44,  // 2^(8-7)×1.5 = 3.0
        0x48,  // 2^(9-7)×1.0 = 4.0
    };

    for (auto s : scale_tests) {
        float sv = ue4m3_to_float(s);
        uint32_t sfa = (uint32_t)s | ((uint32_t)s << 8) |
                       ((uint32_t)s << 16) | ((uint32_t)s << 24);

        mma_nvfp4_4x_kernel<<<1, 32>>>(0x22222222, 0x22222222,
                                        sfa, 0x38383838, d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        float expected = 64.0f * sv;
        printf("  0x%02X     %5.2f   %8.1f   %8.1f   %s\n",
               s, sv, h_out[0], expected,
               (fabsf(h_out[0] - expected) < 0.5f) ? "PASS" : "FAIL");
    }

    //=================================================================
    // 验证4: 4个scale分别不同 (4X的优势: 每16个K元素独立缩放)
    //   sf_A: K[0:15]=1.0, K[16:31]=2.0, K[32:47]=0.5, K[48:63]=4.0
    //   sf_B: 全1.0
    //   A=B=全1.0
    //   D = 16×1.0×1.0 + 16×2.0×1.0 + 16×0.5×1.0 + 16×4.0×1.0
    //     = 16 + 32 + 8 + 64 = 120.0
    //=================================================================
    printf("\n--- 验证4: 4个scale分别不同 ---\n");
    uint8_t s0 = 0x38;  // 1.0
    uint8_t s1 = 0x40;  // 2.0
    uint8_t s2 = 0x30;  // 0.5
    uint8_t s3 = 0x48;  // 4.0
    uint32_t sf_mixed = (uint32_t)s0 | ((uint32_t)s1 << 8) |
                        ((uint32_t)s2 << 16) | ((uint32_t)s3 << 24);

    printf("  sf_A = {K[0:15]=%.1f, K[16:31]=%.1f, K[32:47]=%.1f, K[48:63]=%.1f}\n",
           ue4m3_to_float(s0), ue4m3_to_float(s1),
           ue4m3_to_float(s2), ue4m3_to_float(s3));
    printf("  sf_B = 全1.0, A=B=全fp4(1.0)\n");

    float expected_mixed = 16.0f * ue4m3_to_float(s0) +
                           16.0f * ue4m3_to_float(s1) +
                           16.0f * ue4m3_to_float(s2) +
                           16.0f * ue4m3_to_float(s3);

    mma_nvfp4_4x_kernel<<<1, 32>>>(0x22222222, 0x22222222,
                                    sf_mixed, 0x38383838, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("  结果: %.1f  期望: %.1f  %s\n",
           h_out[0], expected_mixed,
           (fabsf(h_out[0] - expected_mixed) < 0.5f) ? "PASS" : "FAIL");

    cudaFree(d_out);
    return 0;
}
