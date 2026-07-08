/**
 * 最小 NVFP4 Tensor Core 单元 (sm_120, RTX 6000D)
 *
 * 指令: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X
 *       .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
 *
 * 计算: D[16x8] = (sf_A * A[16x64]) × (sf_B * B[64x8]) + C[16x8]
 *       每条指令 16×8×64 = 8192 FMA
 *
 * ============ 数据格式 ============
 *
 * FP4 e2m1 (4-bit: [sign|exp(2)|mant(1)], bias=1):
 *   0x0=0    0x1=0.5   0x2=1.0   0x3=1.5
 *   0x4=2.0  0x5=3.0   0x6=4.0   0x7=6.0
 *   0x8~0xF = 对应负数
 *
 * UE8M0 scale (8-bit unsigned exponent, bias=127):
 *   value = 2^(byte - 127)
 *   0x7F=1.0  0x80=2.0  0x7E=0.5
 *
 * ============ 寄存器布局 ============
 *
 * Native FP4 packing: 每byte装2个FP4 (lower nibble=elem0, upper nibble=elem1)
 *   byte 0x22 = 两个 FP4(1.0)
 *   uint32_t 0x22222222 = 8个 FP4(1.0)
 *
 * A[16×64] 矩阵 (row-major):
 *   每线程 4个 uint32_t 寄存器 {a0, a1, a2, a3}
 *   每寄存器 4 bytes × 2 FP4/byte = 8 FP4
 *   每线程 4×8 = 32 FP4
 *   32线程 × 32 = 1024 FP4 = 16×64 ✓
 *
 * B[64×8] 矩阵 (col-major):
 *   每线程 2个 uint32_t 寄存器 {b0, b1}
 *   每寄存器 8 FP4
 *   每线程 2×8 = 16 FP4
 *   32线程 × 16 = 512 FP4 = 64×8 ✓
 *
 * C/D[16×8] 累加器/输出:
 *   每线程 4个 float 寄存器 {d0, d1, d2, d3}
 *   32线程 × 4 = 128 float = 16×8 ✓
 *
 * Scale factor (scale_vec::2X, K=64分2段):
 *   sf_A: 16-bit 寄存器, 装2个 ue8m0
 *     bits[7:0]  = K[0:31]  的 scale
 *     bits[15:8] = K[32:63] 的 scale
 *   sf_B: 同上
 *   {byte-id, thread-id}: 指定从warp中哪个线程的哪个byte读取scale
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_nvfp4 mma_mxfp4_block32_ue8m0.cu
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

__global__ void mma_nvfp4_kernel(float *D_out) {
    //=================================================================
    // A[16×64]: 全部填 FP4(1.0)
    //   e2m1(1.0) = 0x2, 每byte两个: 0x22, 每寄存器: 0x22222222
    //=================================================================
    uint32_t a0 = 0x22222222;  // a0: A矩阵 fragment, 8个 FP4(1.0)
    uint32_t a1 = 0x22222222;  // a1: A矩阵 fragment, 8个 FP4(1.0)
    uint32_t a2 = 0x22222222;  // a2: A矩阵 fragment, 8个 FP4(1.0)
    uint32_t a3 = 0x22222222;  // a3: A矩阵 fragment, 8个 FP4(1.0)

    //=================================================================
    // B[64×8]: 全部填 FP4(1.0)
    //=================================================================
    uint32_t b0 = 0x22222222;  // b0: B矩阵 fragment, 8个 FP4(1.0)
    uint32_t b1 = 0x22222222;  // b1: B矩阵 fragment, 8个 FP4(1.0)

    //=================================================================
    // C[16×8]: 累加器初始化为 0
    //=================================================================
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    //=================================================================
    // Scale factors (ue8m0):
    //   scale_vec::2X → 每个scale寄存器装2个ue8m0 (16-bit)
    //   0x7F = 2^(127-127) = 1.0
    //   sf_A = {scale_K0_31=1.0, scale_K32_63=1.0} = 0x7F7F
    //   sf_B = 同上
    //=================================================================
    uint16_t sf_A = 0x7F7F;  // A的scale: K[0:31]=1.0, K[32:63]=1.0
    uint16_t sf_B = 0x7F7F;  // B的scale: K[0:31]=1.0, K[32:63]=1.0

    //=================================================================
    // D[16×8]: 输出
    //=================================================================
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"    // D输出: 4×f32
        "{%4,  %5,  %6,  %7},"    // A数据: 4×uint32 (每个装8个FP4)
        "{%8,  %9},"              // B数据: 2×uint32 (每个装8个FP4)
        "{%10, %11, %12, %13},"   // C累加器: 4×f32
        "{%14},"                   // sf_A: 16-bit (2个ue8m0)
        "{%15, %16},"             // {byte-id-a=0, thread-id-a=0}
        "{%17},"                   // sf_B: 16-bit (2个ue8m0)
        "{%18, %19};\n"           // {byte-id-b=0, thread-id-b=0}
        :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
        :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
            "r"(b0),   "r"(b1),
            "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
            "r"((uint32_t)sf_A), "h"((uint16_t)0), "h"((uint16_t)0),
            "r"((uint32_t)sf_B), "h"((uint16_t)0), "h"((uint16_t)0)
    );

    int tid = threadIdx.x;
    D_out[tid * 4 + 0] = d0;
    D_out[tid * 4 + 1] = d1;
    D_out[tid * 4 + 2] = d2;
    D_out[tid * 4 + 3] = d3;
}

//=================================================================
// 验证用: 可变参数版本
//=================================================================
__global__ void mma_nvfp4_flex(uint32_t a_fill, uint32_t b_fill,
                                uint16_t sf_A, uint16_t sf_B,
                                float *D_out) {
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
        :   "r"(a_fill), "r"(a_fill), "r"(a_fill), "r"(a_fill),
            "r"(b_fill), "r"(b_fill),
            "f"(c0), "f"(c1), "f"(c2), "f"(c3),
            "r"((uint32_t)sf_A), "h"((uint16_t)0), "h"((uint16_t)0),
            "r"((uint32_t)sf_B), "h"((uint16_t)0), "h"((uint16_t)0)
    );

    int tid = threadIdx.x;
    D_out[tid * 4 + 0] = d0;
    D_out[tid * 4 + 1] = d1;
    D_out[tid * 4 + 2] = d2;
    D_out[tid * 4 + 3] = d3;
}

//=================================================================
// FP4 e2m1 查表
//=================================================================
float fp4_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

int main() {
    float *d_out, h_out[128];
    CHECK_CUDA(cudaMalloc(&d_out, 32 * 4 * sizeof(float)));

    printf("====== NVFP4 Tensor Core (sm_120, mxf4nvf4, m16n8k64) ======\n");
    printf("D[16x8] = (sf_A * A[16x64]) x (sf_B * B[64x8]) + C[16x8]\n\n");

    //=================================================================
    // 验证1: 基本正确性
    //   A = 全1.0 (0x22), B = 全1.0 (0x22)
    //   sf_A = sf_B = 1.0 (0x7F)
    //   D[i][j] = sum(k=0..63) sf_A*A[i][k] * sf_B*B[k][j]
    //           = 64 × 1.0 × 1.0 × 1.0 × 1.0 = 64.0
    //=================================================================
    printf("--- 验证1: A=B=fp4(1.0), sf_A=sf_B=1.0 ---\n");
    printf("  a0~a3 = 0x22222222 (每寄存器8个e2m1=1.0)\n");
    printf("  b0~b1 = 0x22222222\n");
    printf("  sf_A  = 0x7F7F (2个ue8m0=1.0, 覆盖K[0:31]和K[32:63])\n");
    printf("  sf_B  = 0x7F7F\n");

    mma_nvfp4_kernel<<<1, 32>>>(d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  结果 T00: %.1f %.1f %.1f %.1f  (期望 64.0)\n\n",
           h_out[0], h_out[1], h_out[2], h_out[3]);

    //=================================================================
    // 验证2: FP4 e2m1 全部8个正值
    //   A = 各FP4值, B = 全1.0, sf_A = sf_B = 1.0
    //   D[i][j] = 64 × fp4(A_nib) × 1.0 × 1.0 × 1.0
    //=================================================================
    printf("--- 验证2: 遍历FP4 e2m1正值 ---\n");
    printf("  B=0x22(1.0), sf_A=sf_B=0x7F7F(1.0)\n");
    printf("  %-6s %-6s %-10s %-10s %-6s\n", "nibble", "e2m1", "实测", "期望", "");

    for (int nib = 0; nib <= 7; nib++) {
        // 两个nibble填相同值: byte = (nib<<4)|nib
        uint8_t byte_val = (nib << 4) | nib;
        uint32_t a_fill = (uint32_t)byte_val | ((uint32_t)byte_val << 8) |
                          ((uint32_t)byte_val << 16) | ((uint32_t)byte_val << 24);

        mma_nvfp4_flex<<<1, 32>>>(a_fill, 0x22222222, 0x7F7F, 0x7F7F, d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        float expected = 64.0f * fp4_table[nib] * 1.0f;
        printf("  0x%X     %4.1f   %8.1f   %8.1f   %s\n",
               nib, fp4_table[nib], h_out[0], expected,
               (fabsf(h_out[0] - expected) < 0.5f) ? "PASS" : "FAIL");
    }

    //=================================================================
    // 验证3: Scale factor (ue8m0) 缩放效果
    //   A = B = 全1.0, sf_B = 1.0
    //   sf_A 变化: 0.25, 0.5, 1.0, 2.0, 4.0
    //   D = 64 × sf_A × 1.0 = 64 × sf_A
    //=================================================================
    printf("\n--- 验证3: Scale factor 缩放 ---\n");
    printf("  A=B=0x22(1.0), sf_B=0x7F7F(1.0), sf_A变化\n");
    printf("  %-8s %-8s %-10s %-10s %-6s\n", "ue8m0", "scale", "实测", "期望", "");

    uint8_t scale_tests[] = {0x7D, 0x7E, 0x7F, 0x80, 0x81};
    for (auto s : scale_tests) {
        float sv = powf(2.0f, (float)s - 127.0f);
        uint16_t sfa = (uint16_t)s | ((uint16_t)s << 8);

        mma_nvfp4_flex<<<1, 32>>>(0x22222222, 0x22222222, sfa, 0x7F7F, d_out);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        float expected = 64.0f * sv;
        printf("  0x%02X     %5.2f   %8.1f   %8.1f   %s\n",
               s, sv, h_out[0], expected,
               (fabsf(h_out[0] - expected) < 0.5f) ? "PASS" : "FAIL");
    }

    cudaFree(d_out);
    return 0;
}
