/**
 * tcgen05.mma 最小单元 — FP8 e4m3 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::f8f6f4
 *       [tmem_c], desc_a, desc_b, idesc, p
 *
 * FP8 e4m3: [sign|exp(4)|mant(3)], bias=7, 1.0 = 0x38
 * 无 block scale (对比 nvfp4/mxfp4 的 kind::mxf4nvf4.block_scale)
 * Tile:  M=128, N=8, K=32 (K = 32字节/元素1字节), 128线程 (4 warp, cta_group::1)
 * FMA:   128 × 8 × 32 = 32,768
 *
 * 数据流 (比 nvfp4 少了写 scale 到 TMEM 这步):
 *   1. tcgen05.alloc  → 分配TMEM (只有累加器)
 *   2. 填充smem        → A[128×32] e4m3, B[8×32] e4m3
 *   3. make_desc()    → 构造smem descriptor (uint64)
 *   4. tcgen05.mma    → 1个线程发射 (elect_one)
 *   5. tcgen05.ld     → 从TMEM读结果
 *   6. tcgen05.dealloc → 释放TMEM
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_fp8 tcgen05_fp8.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

__device__ uint32_t smem_u32(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// SmemDescriptor: [0:14) addr>>4, [16:30) LBO>>4, [32:46) SBO>>4, [46:48) version=1
// ⚠️ no-swizzle 布局是 8行×16B core-matrix 分块(不是线性行主): K方向间距LBO=128B, M/N方向SBO=256B
//    本例全1.0输入对排布不敏感; 喂真实数据时 smem 必须按 core-matrix 排, 见 random_cpu_ref/
__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    (void)stride_bytes;
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    desc |= (uint64_t)(1) << 46;
    return desc;
}

// InstrDescriptor for kind::f8f6f4 (非block_scale布局, 与mxf4nvf4不同):
//   [4:6) d_format=1(f32), [7:10) a_format=0(E4M3), [10:13) b_format=0(E4M3)
//   [17:23) n_dim=N>>3, [24:29) m_dim=M>>4
__device__ uint32_t make_idesc() {
    uint32_t d = 0;
    d |= (1 << 4);                    // d_format = F32
    // a_format = 0 (E4M3), b_format = 0 (E4M3)
    d |= (1 << 17);                   // n_dim = N/8 = 1
    d |= (8 << 24);                   // m_dim = M/16 = 8
    return d;
}

constexpr int M = 128, N = 8, K = 32;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K;      // 4096
constexpr int B_BYTES = N * K;      // 256

__global__ void tcgen05_fp8_kernel(float* D_out) {
    extern __shared__ char smem[];
    uint8_t*  smem_a    = (uint8_t*)smem;
    uint8_t*  smem_b    = smem_a + A_BYTES;
    uint32_t* smem_tmem = (uint32_t*)(smem_b + B_BYTES + 64);

    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;

    // A=B=e4m3(1.0)=0x38
    for (int i = tid; i < A_BYTES; i += THREADS) smem_a[i] = 0x38;
    for (int i = tid; i < B_BYTES; i += THREADS) smem_b[i] = 0x38;
    __syncthreads();

    // TMEM alloc (整个warp 0)
    if (warp_id == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(smem_u32(smem_tmem)), "r"(32));
    }
    __syncthreads();

    uint32_t tmem_base = *smem_tmem;
    uint32_t tmem_c    = tmem_base;

    uint64_t desc_a = make_desc(smem_a, K);   // 行 = 32 bytes
    uint64_t desc_b = make_desc(smem_b, K);
    uint32_t idesc  = make_idesc();

    // tcgen05.mma FP8 (1个线程发射); p=0 → 不读入C
    __syncthreads();
    if (tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 "
            "[%0], %1, %2, %3, p;\n\t"
            "}\n"
            : : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                "r"(idesc), "r"(0u));
    }
    __syncthreads();

    // 读TMEM
    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n\t"
                     "tcgen05.wait::ld.sync.aligned;\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));
        D_out[lane * 4 + 0] = __uint_as_float(r0);
        D_out[lane * 4 + 1] = __uint_as_float(r1);
        D_out[lane * 4 + 2] = __uint_as_float(r2);
        D_out[lane * 4 + 3] = __uint_as_float(r3);
    }

    // 释放TMEM
    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_base), "r"(32));
    }
}

int main() {
    float *d_out, h[128];
    CHECK_CUDA(cudaMalloc(&d_out, 4096));

    printf("====== FP8 e4m3 (tcgen05.mma, sm_100) ======\n");
    printf("kind::f8f6f4 (无block scale)\n");
    printf("A=smem e4m3(1.0), B=smem e4m3(1.0)\n");
    printf("M=%d N=%d K=%d, expect D=32.0\n\n", M, N, K);

    CHECK_CUDA(cudaMemset(d_out, 0, 4096));
    tcgen05_fp8_kernel<<<1, THREADS, 8192>>>(d_out);
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
