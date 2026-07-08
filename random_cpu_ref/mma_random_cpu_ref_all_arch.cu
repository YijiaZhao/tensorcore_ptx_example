/**
 * mma.sync 全精度 随机数据 vs CPU参考 验证 (sm_89 / sm_90a / sm_100a / sm_120a 通用)
 *
 * 覆盖: fp16, bf16, tf32, fp8(e4m3), int8(s8), int4(s4) 六种 mma.sync
 *       —— 模拟路径(fp8@sm90/100, int4@sm90+)也一并验证数值等价
 *
 * 与 1.0 系测试互补: 每个元素独立随机 → A/B fragment 映射、行列搞反、
 * descriptor 错位、格式解码错 全部会现形 (全1.0输入对这些不敏感)。
 *
 * fragment 映射(PTX row.col 布局, 全形状统一, 行恒32B=8×u32):
 *   A reg r: row = lane/4 + (r&1)*8,  u32列 = lane%4 + (r>>1)*4
 *   B reg r: row(n) = lane/4,         u32列 = lane%4 + r*4
 *   D reg r: row = lane/4 + (r>>1)*8, col  = (lane%4)*2 + (r&1)
 *
 * 编译(按卡换 -gencode):
 *   nvcc -gencode arch=compute_89,code=sm_89   -std=c++17 -o t mma_random_cpu_ref_all_arch.cu
 *   nvcc -gencode arch=compute_120a,code=sm_120a -std=c++17 -o t mma_random_cpu_ref_all_arch.cu
 * 运行: ./t [seed]   (默认 seed=1, 每种精度跑3轮)
 */
#include <cuda_runtime.h>
#include "verify_random.h"

#define CHECK_CUDA(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__device__ __forceinline__ int a_idx(int lane, int r) {   // u32 index into A[16][8]
    return (lane / 4 + (r & 1) * 8) * 8 + lane % 4 + (r >> 1) * 4;
}
__device__ __forceinline__ int b_idx(int lane, int r) {   // u32 index into B[8][8]
    return (lane / 4) * 8 + lane % 4 + r * 4;
}
__device__ __forceinline__ int d_idx(int lane, int r) {   // float index into D[16][8]
    return (lane / 4 + (r >> 1) * 8) * 8 + (lane % 4) * 2 + (r & 1);
}

// float 累加器族 (fp16/bf16/tf32/fp8)
#define DEF_MMA_FP(NAME, INSTR)                                                  \
__global__ void NAME(const uint32_t* A, const uint32_t* B, float* D) {           \
    int lane = threadIdx.x % 32;                                                 \
    uint32_t a0=A[a_idx(lane,0)], a1=A[a_idx(lane,1)],                            \
             a2=A[a_idx(lane,2)], a3=A[a_idx(lane,3)];                            \
    uint32_t b0=B[b_idx(lane,0)], b1=B[b_idx(lane,1)];                            \
    float d0,d1,d2,d3;                                                           \
    asm volatile(INSTR " {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)                                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),                        \
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));                                   \
    D[d_idx(lane,0)]=d0; D[d_idx(lane,1)]=d1;                                     \
    D[d_idx(lane,2)]=d2; D[d_idx(lane,3)]=d3;                                     \
}
// s32 累加器族 (int8/int4), 结果转float统一比对(|值|≤256精确)
#define DEF_MMA_INT(NAME, INSTR)                                                 \
__global__ void NAME(const uint32_t* A, const uint32_t* B, float* D) {           \
    int lane = threadIdx.x % 32;                                                 \
    uint32_t a0=A[a_idx(lane,0)], a1=A[a_idx(lane,1)],                            \
             a2=A[a_idx(lane,2)], a3=A[a_idx(lane,3)];                            \
    uint32_t b0=B[b_idx(lane,0)], b1=B[b_idx(lane,1)];                            \
    int32_t d0,d1,d2,d3;                                                         \
    asm volatile(INSTR " {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)                                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),                        \
          "r"(0),"r"(0),"r"(0),"r"(0));                                           \
    D[d_idx(lane,0)]=(float)d0; D[d_idx(lane,1)]=(float)d1;                       \
    D[d_idx(lane,2)]=(float)d2; D[d_idx(lane,3)]=(float)d3;                       \
}

DEF_MMA_FP (k_fp16, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
DEF_MMA_FP (k_bf16, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
DEF_MMA_FP (k_tf32, "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
DEF_MMA_FP (k_fp8 , "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
DEF_MMA_INT(k_int8, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
DEF_MMA_INT(k_int4, "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32")

struct Test {
    const char* name; void (*kern)(const uint32_t*, const uint32_t*, float*);
    int K; int ebits; const EncVal* set; int setn;
};

int main(int argc, char** argv) {
    uint32_t seed = argc > 1 ? (uint32_t)atoi(argv[1]) : 1;
    const int M = 16, N = 8, ROW_BYTES = 32, ROUNDS = 3;
    Test tests[] = {
        {"fp16", k_fp16, 16, 16, FP16_SET, SETN(FP16_SET)},
        {"bf16", k_bf16, 16, 16, BF16_SET, SETN(BF16_SET)},
        {"tf32", k_tf32,  8, 32, TF32_SET, SETN(TF32_SET)},
        {"fp8",  k_fp8,  32,  8, E4M3_SET, SETN(E4M3_SET)},
        {"int8", k_int8, 32,  8, S8_SET,   SETN(S8_SET)},
        {"int4", k_int4, 64,  4, S4_SET,   SETN(S4_SET)},
    };

    uint8_t hA[M * ROW_BYTES], hB[N * ROW_BYTES];
    float refA[M * 64], refB[N * 64], refD[M * N], hD[M * N];
    uint8_t *dA, *dB; float* dD;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(hA)));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(hB)));
    CHECK_CUDA(cudaMalloc(&dD, sizeof(hD)));

    printf("====== mma.sync 随机数据 vs CPU参考 (seed=%u, %d轮/精度) ======\n", seed, ROUNDS);
    int total_bad = 0;
    for (auto& t : tests) {
        for (int round = 0; round < ROUNDS; round++) {
            memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
            fill_random(hA, refA, M, t.K, t.set, t.setn, t.ebits, &seed);
            fill_random(hB, refB, N, t.K, t.set, t.setn, t.ebits, &seed);
            cpu_gemm_ref(refA, refB, refD, M, N, t.K);
            CHECK_CUDA(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemset(dD, 0, sizeof(hD)));
            t.kern<<<1, 32>>>((uint32_t*)dA, (uint32_t*)dB, dD);
            cudaError_t e = cudaDeviceSynchronize();
            if (e) { printf("  %-10s KERNEL ERROR: %s\n", t.name, cudaGetErrorString(e)); total_bad++; break; }
            CHECK_CUDA(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));
            char tag[32]; snprintf(tag, sizeof(tag), "%s r%d", t.name, round);
            total_bad += check_exact(hD, refD, M * N, tag);
        }
    }
    printf("====== %s ======\n", total_bad ? "有FAIL, 见上" : "全部PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return total_bad ? 1 : 0;
}
