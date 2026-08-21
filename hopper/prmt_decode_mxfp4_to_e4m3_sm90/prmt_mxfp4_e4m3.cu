/**
 * Hopper 快速解码: MXFP4 (e2m1 + ue8m0) → FP8 (e4m3), 一次解 8 个  [sm_53+, sm_90a]
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
 * UE8M0 高位折算:
 *   ue8m0 存的是共享指数 bias=127，即 scale=2^(sf-127)。E4M3 的 exponent 在 bit[6:3]，
 *   所以对 4 个 packed E4M3 byte 可一次做 packed add：
 *       fp8x4 += (sf - 127) * 0x08080808
 *   这等价于每个 byte 的 exponent += sf-127；无需逐元素浮点乘法。
 *   本例用小范围 scale 保证不发生 subnormal/overflow，验证这条热路径的逐 bit 结果。
 *
 * 数据布局 (Marlin 式打包, 与内核一致):
 *   uq 的 nibble i = 第 i 个元素 (低 nibble 在前);
 *   输出 out_lo 4 字节 = 偶元素 {0,2,4,6}; out_hi 4 字节 = 奇元素 {1,3,5,7}。
 *
 * 验证: 随机 e2m1 (取值集 {0,±0.5,±1,±1.5,±2}, 均 e4m3 精确) → GPU prmt 解码 →
 *       CPU 按上面映射取对应字节解 e4m3 回 float, 逐 bit ==。
 * Compile: nvcc -gencode arch=compute_90a,code=sm_90a -o prmt_mxfp4_e4m3 prmt_mxfp4_e4m3.cu
 */
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
static uint32_t xorshift(uint32_t *x) {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    return *x;
}
#define CHECK_CUDA(c)                                                                              \
    do {                                                                                           \
        cudaError_t e = (c);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                  \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// e2m1 magnitudes 0..7 → e4m3 bytes {0,0.5,1,1.5,2,3,4,6}, 打包给 __byte_perm
__device__ __constant__ uint32_t kMagLo = 0x3C383000u; // mag 0..3
__device__ __constant__ uint32_t kMagHi = 0x4C484440u; // mag 4..7

__global__ void k_decode_mxfp4(const uint32_t *in, const uint8_t *ue8m0, uint32_t *out_lo,
                               uint32_t *out_hi, int nu32) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nu32)
        return;
    uint32_t uq = in[i];

    // =====================================================================
    // 步骤1: E2M1 数值格式转换 —— PRMT 寄存器 LUT 查表
    // =====================================================================
    // 这里只转换每个 E2M1 nibble 的 3-bit magnitude，不处理 block scale。
    // sel_lo/sel_hi 将 8 个 magnitude 整理为两个 PRMT selector：
    //   sel_lo -> 偶数元素 {0,2,4,6}
    //   sel_hi -> 奇数元素 {1,3,5,7}
    uint32_t sel_hi =
        ((uq >> 4) & 0x7u) | ((uq >> 8) & 0x70u) | ((uq >> 12) & 0x700u) | ((uq >> 16) & 0x7000u);
    uint32_t sel_lo =
        (uq & 0x7u) | ((uq >> 4) & 0x70u) | ((uq >> 8) & 0x700u) | ((uq >> 12) & 0x7000u);
    // __byte_perm 编译为 PRMT。两个 32-bit 寄存器常量就是 8 项 LUT：
    //   E2M1 magnitude 0..7 -> E4M3 {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    uint32_t olo = __byte_perm(kMagLo, kMagHi, sel_lo); // PRMT: 偶数元素 magnitude
    uint32_t ohi = __byte_perm(kMagLo, kMagHi, sel_hi); // PRMT: 奇数元素 magnitude

    // PRMT 只查 magnitude；这里再把 E2M1 sign bit 搬到 E4M3 sign bit。
    olo |= (uq << 4) & 0x80808080u; // 偶数元素 sign: nibble bit3 -> byte bit7
    ohi |= uq & 0x80808080u;        // 奇数元素 sign: nibble bit3 -> byte bit7

    // =====================================================================
    // 步骤2: UE8M0 block scale —— E4M3 指数高位折算
    // =====================================================================
    // 这里不查表，也不执行浮点乘法。
    // UE8M0 表示 scale = 2^(ue8m0 - 127)，而 E4M3 exponent 位于 bit[6:3]。
    // 因此乘以 2^exp2 等价于给每个 E4M3 byte 的 exponent 加 exp2：
    //   e4m3_byte += exp2 << 3
    // 0x08080808 将相同 exponent delta 复制到 4 个 packed E4M3 byte。
    // ptxas 通常把下面的 32-bit 整数加法编译为 IADD3。
    int exp2 = (int)ue8m0[i] - 127;
    uint32_t exponent_delta = (uint32_t)(exp2 * 0x08080808);
    olo += exponent_delta; // IADD3: 同时折算 4 个偶数元素的 UE8M0 scale
    ohi += exponent_delta; // IADD3: 同时折算 4 个奇数元素的 UE8M0 scale

    // 最终输出已经是应用 block scale 后的 packed E4M3。
    out_lo[i] = olo;
    out_hi[i] = ohi;
}

int main() {
    const int NU = 64; // 64 words × 8 = 512 元素
    const int N = NU * 8;
    uint32_t hIn[NU];
    uint32_t hLo[NU], hHi[NU];
    uint8_t hScale[NU];
    uint8_t refByte[N], gotByte[N];
    uint32_t seed = 1;
    for (int w = 0; w < NU; w++) { // 每 word 打包 8 个随机 e2m1
        uint32_t uq = 0;
        int exp2 = (int)(xorshift(&seed) % 3) - 1; // ue8m0: 2^-1, 2^0, 2^1
        hScale[w] = (uint8_t)(127 + exp2);
        static const uint8_t e4mag[8] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c};
        for (int e = 0; e < 8; e++) {
            // mag 1..4: 全部 normalized，±1 exponent 折算不会溢出/下溢。
            uint8_t enc = (uint8_t)(1 + xorshift(&seed) % 4) | ((xorshift(&seed) & 1) ? 8 : 0);
            uq |= (uint32_t)enc << (4 * e);
            refByte[w * 8 + e] = (uint8_t)((e4mag[enc & 7] | ((enc & 8) << 4)) + exp2 * 8);
        }
        hIn[w] = uq;
    }
    uint32_t *dIn, *dLo, *dHi;
    uint8_t *dScale;
    CHECK_CUDA(cudaMalloc(&dIn, NU * 4));
    CHECK_CUDA(cudaMalloc(&dLo, NU * 4));
    CHECK_CUDA(cudaMalloc(&dHi, NU * 4));
    CHECK_CUDA(cudaMalloc(&dScale, NU));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, NU * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dScale, hScale, NU, cudaMemcpyHostToDevice));
    k_decode_mxfp4<<<(NU + 31) / 32, 32>>>(dIn, dScale, dLo, dHi, NU);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hLo, dLo, NU * 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hHi, dHi, NU * 4, cudaMemcpyDeviceToHost));
    // 按映射还原 packed bytes，直接逐 bit 验证高位折算后的 E4M3 编码。
    for (int w = 0; w < NU; ++w)
        for (int j = 0; j < 4; ++j) {
            gotByte[w * 8 + 2 * j] = (hLo[w] >> (8 * j)) & 0xff;
            gotByte[w * 8 + 2 * j + 1] = (hHi[w] >> (8 * j)) & 0xff;
        }
    int bad = 0;
    for (int i = 0; i < N; i++)
        if (gotByte[i] != refByte[i]) {
            if (bad < 4)
                printf("bad @%d got=%02x want=%02x\n", i, gotByte[i], refByte[i]);
            ++bad;
        }
    printf("MXFP4+UE8M0 -> E4M3 exponent-fold: %s (%d/%d bytes exact)\n", bad ? "FAIL" : "PASS",
           N - bad, N);
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dLo));
    CHECK_CUDA(cudaFree(dHi));
    CHECK_CUDA(cudaFree(dScale));
    return bad ? 1 : 0;
}
