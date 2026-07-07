/**
 * tcgen05.mma 最大tile — MXFP4 block32 (sm_100)
 *
 * cta_group::1 (4 warps = 128 threads), M=128, N=256, K=64
 * 单条指令: 128 × 256 × 64 = 2,097,152 FMA (~2M FMA)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_mxfp4_max tcgen05_mxfp4_max.cu
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

__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)(((stride_bytes >> 4) & 0x3FFF)) << 16;
    desc |= (uint64_t)(1) << 46;
    return desc;
}

// MXFP4: scale_format=1 (UE8M0)
__device__ uint32_t make_idesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    d |= (1 << 7);                    // a_format = E2M1
    d |= (1 << 10);                   // b_format = E2M1
    d |= (32 << 17);                  // n_dim = N/8 = 32 (N=256)
    d |= (1 << 23);                   // scale_format = UE8M0
    d |= (8 << 24);                   // m_dim = M/16 = 8 (M=128)
    d |= ((tsfa >> 30) & 3) << 29;
    d |= ((tsfb >> 30) & 3) << 4;
    return d;
}

constexpr int M = 128, N = 256, K = 64;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K / 2;  // 4096
constexpr int B_BYTES = N * K / 2;  // 8192

__global__ void tcgen05_mxfp4_max_kernel(float* D_out) {
    extern __shared__ char smem[];
    uint8_t*  smem_a    = (uint8_t*)smem;
    uint8_t*  smem_b    = smem_a + A_BYTES;
    uint32_t* smem_tmem = (uint32_t*)(smem_b + B_BYTES + 64);

    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;

    for (int i = tid; i < A_BYTES; i += THREADS) smem_a[i] = 0x22;
    for (int i = tid; i < B_BYTES; i += THREADS) smem_b[i] = 0x22;
    __syncthreads();

    if (warp_id == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(smem_u32(smem_tmem)), "r"(512));
    }
    __syncthreads();

    uint32_t tmem_base = *smem_tmem;
    uint32_t tmem_c    = tmem_base;
    uint32_t tmem_sfa  = tmem_base + 256;
    uint32_t tmem_sfb  = tmem_base + 300;

    // scale: ue8m0(1.0) = 0x7F
    uint32_t sf_val = 0x7F7F7F7Fu;
    if (warp_id == 0) {
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};\n"
                     : : "r"(tmem_sfa), "r"(sf_val), "r"(sf_val), "r"(sf_val), "r"(sf_val));
    }
    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};\n"
                     : : "r"(tmem_sfb), "r"(sf_val), "r"(sf_val), "r"(sf_val), "r"(sf_val));
    }
    __syncthreads();

    uint64_t desc_a = make_desc(smem_a, K / 2);
    uint64_t desc_b = make_desc(smem_b, K / 2);
    uint32_t idesc  = make_idesc(tmem_sfa, tmem_sfb);

    if (tid == 0) {
        printf("MXFP4 MAX: M=%d N=%d K=%d, cta_group::1 (8 warps)\n", M, N, K);
        printf("FMA per instruction: %d\n", M * N * K);
    }
    __syncthreads();

    if (tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            : : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                "r"(idesc), "r"(0u), "r"(tmem_sfa), "r"(tmem_sfb));
    }
    __syncthreads();

    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));
        D_out[lane * 4 + 0] = __uint_as_float(r0);
        D_out[lane * 4 + 1] = __uint_as_float(r1);
        D_out[lane * 4 + 2] = __uint_as_float(r2);
        D_out[lane * 4 + 3] = __uint_as_float(r3);
    }

    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_base), "r"(512));
    }
}

int main() {
    float *d_out, h[128];
    CHECK_CUDA(cudaMalloc(&d_out, 4096));

    printf("====== MXFP4 MAX tile (tcgen05.mma cta_group::1) ======\n");
    printf("M=%d N=%d K=%d = %d FMA per instruction\n\n", M, N, K, M*N*K);

    CHECK_CUDA(cudaMemset(d_out, 0, 4096));
    tcgen05_mxfp4_max_kernel<<<1, THREADS, 20480>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n", cudaGetErrorString(e)); return 1; }

    CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost));
    int pass = 0;
    for (int i = 0; i < 128; i++) if (h[i] == 64.0f) pass++;
    printf("Result: %d/128 correct (=64.0)\n", pass);
    printf("D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
           h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

    cudaFree(d_out);
    return 0;
}
