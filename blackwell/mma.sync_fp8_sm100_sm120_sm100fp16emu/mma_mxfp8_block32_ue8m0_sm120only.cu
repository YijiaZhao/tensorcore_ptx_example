/**
 * 最小 MXFP8 Tensor Core 单元 (sm_120a, 6000D) — fp8 的硬件block scale
 *
 * 指令: mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
 *       .m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
 *
 * 计算: D[16x8] = (sf_A * A[16x32]) × (sf_B * B[32x8]) + C[16x8]
 *
 * MXFP8: e4m3 data + ue8m0 scale, 每32个元素1个scale (K=32恰好1个block)
 *   scale在MMA内部硬件相乘 —— fp8唯一的硬件scale形态
 *   (Ada/Hopper无此指令; DeepSeek式fp32细粒度scale在任何卡上都只能软件promotion)
 *
 * ue8m0: 8-bit纯指数, value=2^(byte-127); 0x7F=1.0, 0x80=2.0, 0x7E=0.5
 *
 * 本例 sf_A=2.0, sf_B=1.0 → D = 2×1×32 = 64.0 (证明scale硬件生效)
 *
 * ============ 寄存器布局 ============
 *
 * A/B/C/D 与裸fp8版 (mma_fp8.cu) 完全相同:
 *   A 4×u32 (16个e4m3/线程), B 2×u32, C/D 4×f32
 * Scale (scale_vec::1X, K=32只有1段):
 *   sf_A: 32-bit寄存器只用低8位 (1个ue8m0)
 *   {byte-id, thread-id}: 指定从warp哪个线程哪个byte读scale (最小例全0)
 *
 * Compile: nvcc -gencode arch=compute_120a,code=sm_120a -o mma_mxfp8 mma_mxfp8.cu
 *          (sm_120a/121a 专属; sm_100 无此指令, B200 走 tcgen05 版)
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

__global__ void mma_mxfp8_kernel(float *D_out) {
    // A[16×32]: 全部填 e4m3(1.0) = 0x38
    uint32_t a0 = 0x38383838, a1 = 0x38383838, a2 = 0x38383838, a3 = 0x38383838;
    // B[32×8]: 全部填 e4m3(1.0)
    uint32_t b0 = 0x38383838, b1 = 0x38383838;
    // C = 0
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    // Scale: sf_A = ue8m0(2.0) = 0x80, sf_B = ue8m0(1.0) = 0x7F
    uint32_t sf_A = 0x80;
    uint32_t sf_B = 0x7F;

    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"    // D: 4×f32
        "{%4,  %5,  %6,  %7},"    // A: 4×u32 (16个e4m3)
        "{%8,  %9},"              // B: 2×u32
        "{%10, %11, %12, %13},"   // C: 4×f32
        "{%14},"                   // sf_A: 1个ue8m0 (低8位)
        "{%15, %16},"             // {byte-id-a=0, thread-id-a=0}
        "{%17},"                   // sf_B
        "{%18, %19};\n"           // {byte-id-b=0, thread-id-b=0}
        :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
        :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
            "r"(b0),   "r"(b1),
            "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
            "r"(sf_A), "h"((uint16_t)0), "h"((uint16_t)0),
            "r"(sf_B), "h"((uint16_t)0), "h"((uint16_t)0)
    );

    int tid = threadIdx.x;
    D_out[tid * 4 + 0] = d0;
    D_out[tid * 4 + 1] = d1;
    D_out[tid * 4 + 2] = d2;
    D_out[tid * 4 + 3] = d3;
}


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
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
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
    const int M = 16, N = 8, K = 32, EBITS = 8, NSEG = 1, ROUNDS = 3;
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
        fill_random(hA, refA, M, K, E4M3_SET, SETN(E4M3_SET), EBITS, &seed);
        fill_random(hB, refB, N, K, E4M3_SET, SETN(E4M3_SET), EBITS, &seed);
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
    float *d_out, h[128];
    CHECK_CUDA(cudaMalloc(&d_out, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_out, 0, 128 * sizeof(float)));

    printf("====== MXFP8 block32 (mma.sync, sm_120a) ======\n");
    printf("kind::mxf8f6f4.block_scale.scale_vec::1X + ue8m0\n");
    printf("A=1.0, B=1.0, sf_A=2.0, sf_B=1.0\n");
    printf("expect D = 2*1*32 = 64.0 (scale硬件生效的证明)\n\n");

    mma_mxfp8_kernel<<<1, 32>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 64.0f) pass++;
        printf("Result: %d/128 correct (=64.0)\n", pass);
        printf("D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return phase2_random_vs_cpu(1);
}
