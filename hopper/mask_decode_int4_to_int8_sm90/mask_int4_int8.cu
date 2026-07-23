/**
 * Hopper 快速解码: INT4 → INT8, 一次解 8 个  [任意 sm, sm_90a]
 *
 * MegaMoE int4-SX 权重解码热路径 —— 纯 shift+mask, 无查表无访存, 比逐个 cvt 快得多。
 *   8 个 int4 nibble → 8 个 int8 byte 只需 2 条 and + 1 条 shift。
 *   (QoQ 两级量化用 uint4 + zero-point: 解码 = 本示例的 unpack 再 vsub4 减 z,
 *    vsub4 = SIMD 4-way 字节减, 一条指令减 4 个; 见 sm90_mxfp4_mega_moe 的 int 路径。)
 *
 * 指令: and.b32 + shr.b32 (shift/mask), 可选 vsub4.u32.u32.u32.sat 减零点
 *
 * 数据布局: uq 的 nibble i = 第 i 个元素(低 nibble 在前);
 *   out_lo 4 字节 = 偶元素 {0,2,4,6} (每字节低 nibble);
 *   out_hi 4 字节 = 奇元素 {1,3,5,7} (每字节高 nibble)。
 *
 * 验证: 随机 uint4 (取值 {0,1,2,3,4}) → GPU shift+mask → CPU 按映射取字节, 逐 bit ==。
 * Compile: nvcc -gencode arch=compute_90a,code=sm_90a -o mask_int4_int8 mask_int4_int8.cu
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "../../random_cpu_ref/verify_random.h"
#define CHECK_CUDA(c) do{cudaError_t e=(c);if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__global__ void k_unpack_int4(const uint32_t* in, uint32_t* out_lo, uint32_t* out_hi, int nu32) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nu32) return;
    uint32_t uq = in[i];
    out_lo[i] =  uq        & 0x0F0F0F0Fu;   // 每字节低 nibble → 偶元素
    out_hi[i] = (uq >> 4)  & 0x0F0F0F0Fu;   // 每字节高 nibble → 奇元素
}

int main() {
    const int NU = 64; const int N = NU * 8;
    const int VAL[5] = {0,1,2,3,4};        // uint4 取值(=int8 值, 无符号解包直取)
    uint32_t hIn[NU], hLo[NU], hHi[NU];
    int ref[N], got[N]; uint32_t seed = 1;
    for (int w = 0; w < NU; w++) {
        uint32_t uq = 0;
        for (int e = 0; e < 8; e++) { int v = VAL[xorshift(&seed) % 5]; uq |= (uint32_t)(v & 0xF) << (4*e); ref[w*8+e] = v; }
        hIn[w] = uq;
    }
    uint32_t *dIn,*dLo,*dHi;
    CHECK_CUDA(cudaMalloc(&dIn,NU*4)); CHECK_CUDA(cudaMalloc(&dLo,NU*4)); CHECK_CUDA(cudaMalloc(&dHi,NU*4));
    CHECK_CUDA(cudaMemcpy(dIn,hIn,NU*4,cudaMemcpyHostToDevice));
    k_unpack_int4<<<(NU+31)/32,32>>>(dIn,dLo,dHi,NU);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hLo,dLo,NU*4,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hHi,dHi,NU*4,cudaMemcpyDeviceToHost));
    for (int w=0; w<NU; w++) for (int j=0;j<4;j++){
        got[w*8+2*j]   = (hLo[w] >> (8*j)) & 0xFF;
        got[w*8+2*j+1] = (hHi[w] >> (8*j)) & 0xFF;
    }
    printf("\n====== INT4 → INT8  shift+mask 解包, 8/次 (seed=%u) ======\n", seed);
    int bad=0; for(int i=0;i<N;i++) if(got[i]!=ref[i]){ if(bad<4)printf("  MISMATCH @%d got %d want %d\n",i,got[i],ref[i]); bad++; }
    printf("  int4->int8 %s (%d/%d exact)\n", bad?"FAIL":"PASS", N-bad, N);
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dLo)); CHECK_CUDA(cudaFree(dHi));
    printf(bad ? "  === FAIL ===\n" : "  === PASS (逐bit精确, shift+mask 热路径解码) ===\n");
    return bad ? 1 : 0;
}
