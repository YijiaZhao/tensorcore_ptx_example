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

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 阶段2: 随机数据 + 随机scale vs CPU 参考 (逐bit精确)
//   数据每元素独立随机, scale 每K段独立随机(对所有行广播, 与硬件语义一致),
//   CPU 参考: D = Σ_seg sfa[seg]*sfb[seg]*Σ_{k∈seg} a*b — scale 数学被完整验证。
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

__global__ void phase2_random_kernel(const uint32_t* A, const uint32_t* B, float* D,
                                     uint32_t sfa, uint32_t sfb) {
    int lane = threadIdx.x % 32;
    uint32_t a0=A[mma_a_idx(lane,0)], a1=A[mma_a_idx(lane,1)],
             a2=A[mma_a_idx(lane,2)], a3=A[mma_a_idx(lane,3)];
    uint32_t b0=B[mma_b_idx(lane,0)], b1=B[mma_b_idx(lane,1)];
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
    D[mma_d_idx(lane,0)]=d0; D[mma_d_idx(lane,1)]=d1;
    D[mma_d_idx(lane,2)]=d2; D[mma_d_idx(lane,3)]=d3;
}

static int phase2_random_vs_cpu(uint32_t seed) {
    const int M = 16, N = 8, K = 64, EBITS = 4, NSEG = 2, ROUNDS = 3;
    static uint8_t hA[16*32], hB[8*32];
    static float refA[16*64], refB[8*64], refD[128], hD[128];
    uint8_t *dA, *dB; float* dD;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(hA)));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(hB)));
    CHECK_CUDA(cudaMalloc(&dD, sizeof(hD)));
    printf("\n====== 阶段2: 随机数据+随机scale vs CPU参考 (seed=%u) ======\n", seed);
    int bad = 0;
    for (int r = 0; r < ROUNDS; r++) {
        float sfaf[8], sfbf[8]; uint32_t sfa = 0, sfb = 0;
        for (int g = 0; g < NSEG; g++) {
            const EncVal& ea = UE8M0_SET[xorshift(&seed) % SETN(UE8M0_SET)];
            const EncVal& eb = UE8M0_SET[xorshift(&seed) % SETN(UE8M0_SET)];
            sfa |= (ea.enc & 0xFF) << (8*g); sfaf[g] = ea.val;
            sfb |= (eb.enc & 0xFF) << (8*g); sfbf[g] = eb.val;
        }
        memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
        fill_random(hA, refA, M, K, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
        fill_random(hB, refB, N, K, E2M1_SET, SETN(E2M1_SET), EBITS, &seed);
        cpu_gemm_ref_bs(refA, refB, refD, M, N, K, sfaf, sfbf, NSEG);
        CHECK_CUDA(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
        phase2_random_kernel<<<1, 32>>>((uint32_t*)dA, (uint32_t*)dB, dD, sfa, sfb);
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
