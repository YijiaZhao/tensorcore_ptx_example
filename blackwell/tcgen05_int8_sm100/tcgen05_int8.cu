/**
 * tcgen05.mma 最小单元 — INT8 s8 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::i8
 *       [tmem_c], desc_a, desc_b, idesc, p
 *
 * S8: 有符号8-bit整数, 累加器是 S32 (d_format=2)
 *     InstrDescriptor a/b_format: 0=U8, 1=S8 (本例)
 * Tile:  M=128, N=8, K=32 (K = 32字节/元素1字节), 128线程 (4 warp, cta_group::1)
 * IMAD:  128 × 8 × 32 = 32,768
 *
 * 注意: Blackwell tensor core 没有 int4 —— tcgen05 无 kind::i4
 *       (mma.sync 的 s4 也在 sm_90+ 被移除, fp4 取代了 int4)
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_int8 tcgen05_int8.cu
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

// InstrDescriptor for kind::i8:
//   [4:6) d_format=2(S32), [7:10) a_format=1(S8, 0=U8), [10:13) b_format=1(S8)
//   [17:23) n_dim=N>>3, [24:29) m_dim=M>>4
__device__ uint32_t make_idesc() {
    uint32_t d = 0;
    d |= (2 << 4);                    // d_format = S32
    d |= (1 << 7);                    // a_format = S8
    d |= (1 << 10);                   // b_format = S8
    d |= (1 << 17);                   // n_dim = N/8 = 1
    d |= (8 << 24);                   // m_dim = M/16 = 8
    return d;
}

constexpr int M = 128, N = 8, K = 32;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K;      // 4096
constexpr int B_BYTES = N * K;      // 256

__global__ void tcgen05_int8_kernel(int32_t* D_out) {
    extern __shared__ char smem[];
    uint8_t*  smem_a    = (uint8_t*)smem;
    uint8_t*  smem_b    = smem_a + A_BYTES;
    uint32_t* smem_tmem = (uint32_t*)(smem_b + B_BYTES + 64);

    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;

    // A=B=s8(1)=0x01
    for (int i = tid; i < A_BYTES; i += THREADS) smem_a[i] = 0x01;
    for (int i = tid; i < B_BYTES; i += THREADS) smem_b[i] = 0x01;
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

    // tcgen05.mma INT8 (1个线程发射); p=0 → 不读入C
    __syncthreads();
    if (tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::i8 "
            "[%0], %1, %2, %3, p;\n\t"
            "}\n"
            : : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                "r"(idesc), "r"(0u));
    }
    __syncthreads();

    // 读TMEM (s32结果)
    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n\t"
                     "tcgen05.wait::ld.sync.aligned;\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));
        D_out[lane * 4 + 0] = (int32_t)r0;
        D_out[lane * 4 + 1] = (int32_t)r1;
        D_out[lane * 4 + 2] = (int32_t)r2;
        D_out[lane * 4 + 3] = (int32_t)r3;
    }

    // 释放TMEM
    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_base), "r"(32));
    }
}

int main() {
    int32_t *d_out, h[128];
    CHECK_CUDA(cudaMalloc(&d_out, 4096));

    printf("====== INT8 s8 (tcgen05.mma, sm_100) ======\n");
    printf("kind::i8, d_format=S32\n");
    printf("A=smem s8(1), B=smem s8(1)\n");
    printf("M=%d N=%d K=%d, expect D=32\n\n", M, N, K);

    CHECK_CUDA(cudaMemset(d_out, 0, 4096));
    tcgen05_int8_kernel<<<1, THREADS, 8192>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(int32_t), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 32) pass++;
        printf("Result: %d/128 correct (=32)\n", pass);
        printf("D[0:8]: %d %d %d %d %d %d %d %d\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    cudaFree(d_out);
    return 0;
}
