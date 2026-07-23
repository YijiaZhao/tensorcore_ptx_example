/**
 * Hopper 快速解码: MXFP4 (e2m1) → FP8 (e4m3), 一次解 8 个  [sm_53+, sm_90a]
 *
 * 这是 MegaMoE 权重解码的真实热路径 —— 不用 cvt(逐个转太慢), 而是
 *   prmt.b32(__byte_perm) 从寄存器常量 LUT 一次 permute 出 4 个字节。
 *   8 个 e2m1 nibble → 8 个 e4m3 byte 只需 2 条 byte_perm + 几条位运算。
 *
 * 指令: prmt.b32 (经 __byte_perm 发射) + 位运算(shift/and/or)
 *   - LUT 是两个 32-bit 寄存器常量, 不占 smem, 不访存:
 *       kMagLo=0x3C383000 → mag 0..3 = {0.0, 0.5, 1.0, 1.5} 的 e4m3 字节
 *       kMagHi=0x4C484440 → mag 4..7 = {2.0, 3.0, 4.0, 6.0}
 *   - 为什么不用 cvt: e2m1 打包在 nibble 里, 没有"从 nibble 直接 cvt"的指令;
 *     且解码在 per-MAC 热路径(K维), 逐个 cvt 会成瓶颈, prmt 查表 ~快一个量级。
 *
 * 数据布局 (Marlin 式打包, 与内核一致):
 *   uq 的 nibble i = 第 i 个元素 (低 nibble 在前);
 *   输出 out_lo 4 字节 = 偶元素 {0,2,4,6}; out_hi 4 字节 = 奇元素 {1,3,5,7}。
 *
 * 验证: 随机 e2m1 (取值集 {0,±0.5,±1,±1.5,±2}, 均 e4m3 精确) → GPU prmt 解码 →
 *       CPU 按上面映射取对应字节解 e4m3 回 float, 逐 bit ==。
 * Compile: nvcc -gencode arch=compute_90a,code=sm_90a -o prmt_mxfp4_e4m3 prmt_mxfp4_e4m3.cu
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "../../random_cpu_ref/verify_random.h"
#define CHECK_CUDA(c) do{cudaError_t e=(c);if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// e2m1 magnitudes 0..7 → e4m3 bytes {0,0.5,1,1.5,2,3,4,6}, 打包给 __byte_perm
__device__ __constant__ uint32_t kMagLo = 0x3C383000u;  // mag 0..3
__device__ __constant__ uint32_t kMagHi = 0x4C484440u;  // mag 4..7

__global__ void k_decode_mxfp4(const uint32_t* in, uint32_t* out_lo, uint32_t* out_hi, int nu32) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nu32) return;
    uint32_t uq = in[i];
    // sel: 把每个元素的 3-bit 幅值搬到 byte_perm 选择子的对应 nibble
    uint32_t sel_hi = ((uq >> 4) & 0x7u) | ((uq >> 8) & 0x70u) | ((uq >> 12) & 0x700u) | ((uq >> 16) & 0x7000u);
    uint32_t sel_lo = ( uq       & 0x7u) | ((uq >> 4) & 0x70u) | ((uq >> 8)  & 0x700u) | ((uq >> 12) & 0x7000u);
    uint32_t olo = __byte_perm(kMagLo, kMagHi, sel_lo);   // prmt.b32: 偶元素幅值
    uint32_t ohi = __byte_perm(kMagLo, kMagHi, sel_hi);   //           奇元素幅值
    olo |= (uq << 4) & 0x80808080u;   // 偶元素符号 (nibble bit3 → e4m3 bit7)
    ohi |=  uq       & 0x80808080u;   // 奇元素符号
    out_lo[i] = olo; out_hi[i] = ohi;
}
static float e4m3_decode(uint8_t enc){for(int i=0;i<SETN(E4M3_SET);i++)if((uint8_t)E4M3_SET[i].enc==enc)return E4M3_SET[i].val;return 1e30f;}

int main() {
    const int NU = 64;                 // 64 words × 8 = 512 元素
    const int N = NU * 8;
    uint32_t hIn[NU]; uint32_t hLo[NU], hHi[NU];
    float ref[N], got[N]; uint32_t seed = 1;
    for (int w = 0; w < NU; w++) {     // 每 word 打包 8 个随机 e2m1
        uint32_t uq = 0;
        for (int e = 0; e < 8; e++) {
            const EncVal& v = E2M1_SET[xorshift(&seed) % SETN(E2M1_SET)];
            uq |= (v.enc & 0xF) << (4 * e);
            ref[w * 8 + e] = v.val;
        }
        hIn[w] = uq;
    }
    uint32_t *dIn, *dLo, *dHi;
    CHECK_CUDA(cudaMalloc(&dIn, NU*4)); CHECK_CUDA(cudaMalloc(&dLo, NU*4)); CHECK_CUDA(cudaMalloc(&dHi, NU*4));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, NU*4, cudaMemcpyHostToDevice));
    k_decode_mxfp4<<<(NU+31)/32, 32>>>(dIn, dLo, dHi, NU);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hLo, dLo, NU*4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hHi, dHi, NU*4, cudaMemcpyDeviceToHost));
    // 按映射还原: 偶元素 2j ← out_lo 字节 j; 奇元素 2j+1 ← out_hi 字节 j
    for (int w = 0; w < NU; w++)
        for (int j = 0; j < 4; j++) {
            got[w*8 + 2*j]     = e4m3_decode((hLo[w] >> (8*j)) & 0xFF);
            got[w*8 + 2*j + 1] = e4m3_decode((hHi[w] >> (8*j)) & 0xFF);
        }
    printf("\n====== MXFP4(e2m1) → FP8(e4m3)  prmt.b32 查表解码, 8/次 (seed=%u) ======\n", seed);
    int bad = check_exact(got, ref, N, "mxfp4->e4m3");
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dLo)); CHECK_CUDA(cudaFree(dHi));
    printf(bad ? "  === FAIL ===\n" : "  === PASS (逐bit精确, prmt 热路径解码) ===\n");
    return bad ? 1 : 0;
}
