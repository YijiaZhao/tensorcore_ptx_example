/**
 * wgmma.mma_async 全精度 随机数据 vs CPU参考 验证 (sm_90a 专属)
 *
 * 覆盖: fp16, bf16, tf32, fp8(e4m3 原生QGMMA), int8
 * A[64×K]/B[8×K] 随机编码经 smem descriptor 进 wgmma, CPU float 参考逐元素精确比对。
 *
 * D fragment 映射 (m64n8, 4 reg/线程, warpgroup=4 warp):
 *   warp w 管 16 行: row = 16w + lane/4 + (r>>1)*8, col = (lane%4)*2 + (r&1)
 *
 * 编译: nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 -o t wgmma_random_cpu_ref_sm90.cu
 * 运行: ./t [seed]
 */
#include <cuda_runtime.h>
#include "verify_random.h"

#define CHECK_CUDA(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

constexpr int M = 64, N = 8, ROW_BYTES = 32, THREADS = 128;
constexpr int A_BYTES = M * ROW_BYTES, B_BYTES = N * ROW_BYTES;

__device__ uint32_t smem_u32(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ uint64_t make_desc_wgmma(void const* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)(lbo & 0x3FFF) << 16;   // LBO (>>4 单位)
    desc |= (uint64_t)(sbo & 0x3FFF) << 32;   // SBO (>>4 单位)
    return desc;
}
__device__ __forceinline__ int d_pos(int tid, int r) {   // -> D[64][8] 线性下标
    int lane = tid % 32, w = tid / 32;
    return (16 * w + lane / 4 + (r >> 1) * 8) * 8 + (lane % 4) * 2 + (r & 1);
}

#define WGMMA_PROLOGUE                                                       \
    extern __shared__ char smem[];                                           \
    uint32_t* sA = (uint32_t*)smem; uint32_t* sB = (uint32_t*)(smem + A_BYTES); \
    int tid = threadIdx.x;                                                   \
    for (int i = tid; i < A_BYTES / 4; i += THREADS) sA[i] = ((uint32_t*)A)[i]; \
    for (int i = tid; i < B_BYTES / 4; i += THREADS) sB[i] = ((uint32_t*)B)[i]; \
    __syncthreads();                                                         \
    asm volatile("fence.proxy.async.shared::cta;");                          \
    uint64_t da = make_desc_wgmma(sA, LBO, SBO), db = make_desc_wgmma(sB, LBO, SBO);

// float 累加器族; TAIL: f16/bf16=", 1, 1, 0, 0" (scale+trans), tf32/fp8=", 1, 1"
#define DEF_WGMMA_FP(NAME, INSTR, TAIL)                                      \
__global__ void NAME(const uint8_t* A, const uint8_t* B, float* D, uint32_t LBO, uint32_t SBO) {         \
    WGMMA_PROLOGUE                                                           \
    float d0, d1, d2, d3;                                                    \
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"           \
        "wgmma.fence.sync.aligned;\n\t"                                      \
        INSTR " {%0,%1,%2,%3}, %4, %5, p" TAIL ";\n\t"                       \
        "wgmma.commit_group.sync.aligned;\n\t"                               \
        "wgmma.wait_group.sync.aligned 0;\n\t}\n"                            \
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)                                 \
        : "l"(da), "l"(db), "r"(0u));                                        \
    D[d_pos(tid,0)]=d0; D[d_pos(tid,1)]=d1; D[d_pos(tid,2)]=d2; D[d_pos(tid,3)]=d3; \
}
#define DEF_WGMMA_INT(NAME, INSTR)                                           \
__global__ void NAME(const uint8_t* A, const uint8_t* B, float* D, uint32_t LBO, uint32_t SBO) {         \
    WGMMA_PROLOGUE                                                           \
    int32_t d0, d1, d2, d3;                                                  \
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"           \
        "wgmma.fence.sync.aligned;\n\t"                                      \
        INSTR " {%0,%1,%2,%3}, %4, %5, p;\n\t"                               \
        "wgmma.commit_group.sync.aligned;\n\t"                               \
        "wgmma.wait_group.sync.aligned 0;\n\t}\n"                            \
        : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)                                 \
        : "l"(da), "l"(db), "r"(0u));                                        \
    D[d_pos(tid,0)]=(float)d0; D[d_pos(tid,1)]=(float)d1;                     \
    D[d_pos(tid,2)]=(float)d2; D[d_pos(tid,3)]=(float)d3;                     \
}

DEF_WGMMA_FP (k_fp16, "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16",   ", 1, 1, 0, 0")
DEF_WGMMA_FP (k_bf16, "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ", 1, 1, 0, 0")
DEF_WGMMA_FP (k_tf32, "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32",  ", 1, 1")
DEF_WGMMA_FP (k_fp8 , "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3", ", 1, 1")
DEF_WGMMA_INT(k_int8, "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8")

struct Test {
    const char* name; void (*kern)(const uint8_t*, const uint8_t*, float*, uint32_t, uint32_t);
    int K; int ebits; const EncVal* set; int setn;
};

static uint8_t hA[A_BYTES], hB[B_BYTES], tmpbuf[A_BYTES];
static float refA[M * 64], refB[N * 64], refD[M * N], hD[M * N];
// GMMA no-swizzle K-major: smem 按 8行×16B core-matrix 分块, 不是线性行主
static void to_core_matrix(uint8_t* buf, int rows) {
    memcpy(tmpbuf, buf, rows * 32);
    for (int m = 0; m < rows; m++)
        for (int kb = 0; kb < 32; kb++)
            buf[(m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16] = tmpbuf[m*32 + kb];
}

int main(int argc, char** argv) {
    uint32_t seed = argc > 1 ? (uint32_t)atoi(argv[1]) : 1;
    uint32_t LBO = argc > 2 ? (uint32_t)atoi(argv[2]) : 8;    // core-matrix K方向步长 >>4
    uint32_t SBO = argc > 3 ? (uint32_t)atoi(argv[3]) : 16;   // core-matrix M方向步长 >>4
    const int ROUNDS = 3;
    Test tests[] = {
        {"fp16", k_fp16, 16, 16, FP16_SET, SETN(FP16_SET)},
        {"bf16", k_bf16, 16, 16, BF16_SET, SETN(BF16_SET)},
        {"tf32", k_tf32,  8, 32, TF32_SET, SETN(TF32_SET)},
        {"fp8",  k_fp8,  32,  8, E4M3_SET, SETN(E4M3_SET)},
        {"int8", k_int8, 32,  8, S8_SET,   SETN(S8_SET)},
    };

    uint8_t *dA, *dB; float* dD;
    CHECK_CUDA(cudaMalloc(&dA, A_BYTES));
    CHECK_CUDA(cudaMalloc(&dB, B_BYTES));
    CHECK_CUDA(cudaMalloc(&dD, sizeof(refD)));

    printf("====== wgmma 随机数据 vs CPU参考 (seed=%u LBO=%u SBO=%u, %d轮/精度) ======\n", seed, LBO, SBO, ROUNDS);
    int total_bad = 0;
    for (auto& t : tests) {
        for (int round = 0; round < ROUNDS; round++) {
            memset(hA, 0, sizeof(hA)); memset(hB, 0, sizeof(hB));
            fill_random(hA, refA, M, t.K, t.set, t.setn, t.ebits, &seed);
            fill_random(hB, refB, N, t.K, t.set, t.setn, t.ebits, &seed);
            cpu_gemm_ref(refA, refB, refD, M, N, t.K);
            to_core_matrix(hA, M); to_core_matrix(hB, N);
            CHECK_CUDA(cudaMemcpy(dA, hA, A_BYTES, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, hB, B_BYTES, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemset(dD, 0, sizeof(refD)));
            t.kern<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD, LBO, SBO);
            cudaError_t e = cudaDeviceSynchronize();
            if (e) { printf("  %-10s KERNEL ERROR: %s\n", t.name, cudaGetErrorString(e)); total_bad++; break; }
            CHECK_CUDA(cudaMemcpy(hD, dD, sizeof(refD), cudaMemcpyDeviceToHost));
            char tag[32]; snprintf(tag, sizeof(tag), "%s r%d", t.name, round);
            total_bad += check_exact(hD, refD, M * N, tag);
        }
    }
    printf("====== %s ======\n", total_bad ? "有FAIL, 见上" : "全部PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return total_bad ? 1 : 0;
}
