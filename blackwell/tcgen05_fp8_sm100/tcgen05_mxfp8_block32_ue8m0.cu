/**
 * tcgen05.mma 最小单元 — MXFP8 block32 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
 *       [tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p
 *
 * MXFP8: e4m3 data + ue8m0 scale, 每32个FP8共享1个scale —— fp8 唯一的硬件scale形态
 *   (对比: 裸fp8 kind::f8f6f4 无scale; DeepSeek式fp32细粒度scale只能软件promotion)
 * ue8m0: 8-bit纯指数, value=2^(byte-127), 只能表示2的幂; 0x7F=1.0, 0x80=2.0
 * Tile:  M=128, N=8, K=32 (1X: K=32恰好1个block/行), 128线程 (4 warp, cta_group::1)
 *
 * 本例特意用 sf_A=2.0, sf_B=1.0 → D = 2×1×32 = 64.0
 * (若scale未生效结果会是32.0, 以此证明scale确实在MMA内部硬件相乘)
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_mxfp8 tcgen05_mxfp8.cu
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

// SmemDescriptor: [0:14) addr>>4, [16:30) stride>>4, [46:48) version=1
__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 16;
    desc |= (uint64_t)(1) << 46;
    return desc;
}

// InstrDescriptorBlockScaled for MXFP8: a/b=E4M3(0), scale=UE8M0, M=128, N=8
__device__ uint32_t make_idesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    // a_format = 0 (E4M3), b_format = 0 (E4M3)
    d |= (1 << 17);                   // n_dim = N/8 = 1
    d |= (1 << 23);                   // scale_format = 1 (UE8M0)
    d |= (8 << 24);                   // m_dim = M/16 = 8
    d |= ((tsfa >> 30) & 3) << 29;    // a_sf_id
    d |= ((tsfb >> 30) & 3) << 4;     // b_sf_id
    return d;
}

constexpr int M = 128, N = 8, K = 32;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K;      // 4096
constexpr int B_BYTES = N * K;      // 256

__global__ void tcgen05_mxfp8_kernel(float* D_out) {
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
    uint32_t tmem_sfa  = tmem_base + 8;
    uint32_t tmem_sfb  = tmem_base + 12;

    // 写scale: sf_A = ue8m0(2.0) = 0x80, sf_B = ue8m0(1.0) = 0x7F
    if (warp_id == 0) {
        uint32_t v = 0x80808080u;
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};\n"
                     : : "r"(tmem_sfa), "r"(v), "r"(v), "r"(v), "r"(v));
    }
    __syncthreads();
    if (warp_id == 0) {
        uint32_t v = 0x7F7F7F7Fu;
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};\n"
                     : : "r"(tmem_sfb), "r"(v), "r"(v), "r"(v), "r"(v));
    }
    __syncthreads();

    uint64_t desc_a = make_desc(smem_a, K);   // 行 = 32 bytes
    uint64_t desc_b = make_desc(smem_b, K);
    uint32_t idesc  = make_idesc(tmem_sfa, tmem_sfb);

    // tcgen05.mma MXFP8 (1个线程发射)
    __syncthreads();
    if (tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            : : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                "r"(idesc), "r"(0u), "r"(tmem_sfa), "r"(tmem_sfb));
    }
    __syncthreads();

    // 读TMEM
    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n"
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

    printf("====== MXFP8 block32 (tcgen05.mma, sm_100) ======\n");
    printf("kind::mxf8f6f4.block_scale.scale_vec::1X\n");
    printf("A=e4m3(1.0), B=e4m3(1.0), sf_A=ue8m0(2.0), sf_B=ue8m0(1.0)\n");
    printf("M=%d N=%d K=%d, expect D = 2*1*%d = 64.0 (scale硬件生效的证明)\n\n", M, N, K, K);

    CHECK_CUDA(cudaMemset(d_out, 0, 4096));
    tcgen05_mxfp8_kernel<<<1, THREADS, 8192>>>(d_out);
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
    return 0;
}
