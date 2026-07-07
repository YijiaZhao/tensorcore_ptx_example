/**
 * tcgen05.mma 最小单元 — NVFP4 block16 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
 *       [tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p
 *
 * NVFP4: e2m1 data + ue4m3 scale, 每16个FP4共享1个scale
 * Tile:  M=128, N=8, K=64, 128线程 (4 warp, cta_group::1)
 * FMA:   128 × 8 × 64 = 65,536
 *
 * 数据流:
 *   1. tcgen05.alloc  → 分配TMEM (累加器 + scale)
 *   2. 填充smem        → A[128×64] FP4, B[8×64] FP4
 *   3. tcgen05.st     → 写ue4m3 scale到TMEM
 *   4. make_desc()    → 构造smem descriptor (uint64)
 *   5. tcgen05.mma    → 1个线程发射 (elect_one)
 *   6. tcgen05.ld     → 从TMEM读结果
 *   7. tcgen05.dealloc → 释放TMEM
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_nvfp4 tcgen05_nvfp4.cu
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

// InstrDescriptorBlockScaled for NVFP4: a/b=E2M1, scale=UE4M3, M=128, N=8
__device__ uint32_t make_idesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    d |= (1 << 7);                    // a_format = E2M1
    d |= (1 << 10);                   // b_format = E2M1
    d |= (1 << 17);                   // n_dim = N/8 = 1
    // scale_format = 0 (UE4M3)
    d |= (8 << 24);                   // m_dim = M/16 = 8
    d |= ((tsfa >> 30) & 3) << 29;    // a_sf_id
    d |= ((tsfb >> 30) & 3) << 4;     // b_sf_id
    return d;
}

constexpr int M = 128, N = 8, K = 64;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K / 2;  // 4096
constexpr int B_BYTES = N * K / 2;  // 256

__global__ void tcgen05_nvfp4_kernel(float* D_out) {
    extern __shared__ char smem[];
    uint8_t*  smem_a    = (uint8_t*)smem;
    uint8_t*  smem_b    = smem_a + A_BYTES;
    uint32_t* smem_tmem = (uint32_t*)(smem_b + B_BYTES + 64);

    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;

    // A=B=FP4(1.0)=0x22
    for (int i = tid; i < A_BYTES; i += THREADS) smem_a[i] = 0x22;
    for (int i = tid; i < B_BYTES; i += THREADS) smem_b[i] = 0x22;
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

    // 写scale: ue4m3(1.0) = 0x38
    uint32_t sf_val = 0x38383838u;
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

    // tcgen05.mma NVFP4 (1个线程发射)
    __syncthreads();
    if (tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
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

// ============================================================
// TS变体: A在TMEM (kind::mxf4nvf4 + block_scale)
//
// CUTLASS没实现这个变体,但硬件实测支持!
// 指令: tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
//       [tmem_c], [tmem_a], desc_b, idesc, [tmem_sfa], [tmem_sfb], p
//                  ↑ A在TMEM (TS)
//
// 与SS的区别: desc_a (smem descriptor) → [tmem_a] (TMEM地址)
// ============================================================
__global__ void tcgen05_nvfp4_ts_kernel(float* D_out) {
    extern __shared__ char smem[];
    uint8_t*  smem_b    = (uint8_t*)smem;        // 只需B在smem
    uint32_t* smem_tmem = (uint32_t*)(smem_b + B_BYTES + 64);

    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;

    // B=FP4(1.0)=0x22
    for (int i = tid; i < B_BYTES; i += THREADS) smem_b[i] = 0x22;
    __syncthreads();

    // TMEM alloc: D + A + scale
    if (warp_id == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(smem_u32(smem_tmem)), "r"(32));
    }
    __syncthreads();

    uint32_t tmem_base = *smem_tmem;
    uint32_t tmem_c    = tmem_base;
    uint32_t tmem_a    = tmem_base + 8;   // A在TMEM
    uint32_t tmem_sfa  = tmem_base + 16;
    uint32_t tmem_sfb  = tmem_base + 20;

    // 写A=FP4(1.0)到TMEM
    if (warp_id == 0) {
        uint32_t v = 0x22222222u;
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};\n"
                     : : "r"(tmem_a), "r"(v), "r"(v), "r"(v), "r"(v));
    }
    __syncthreads();

    // 写scale: ue4m3(1.0) = 0x38
    uint32_t sf_val = 0x38383838u;
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

    uint64_t desc_b = make_desc(smem_b, K / 2);
    uint32_t idesc  = make_idesc(tmem_sfa, tmem_sfb);

    // tcgen05.mma TS: [tmem_c], [tmem_a], desc_b, idesc, [tmem_sfa], [tmem_sfb], p
    __syncthreads();
    if (tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0], [%1], %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            : : "r"(tmem_c),      // [tmem_c]:   D在TMEM
                "r"(tmem_a),      // [tmem_a]:   A在TMEM ← TS!
                "l"(desc_b),      // desc_b:     B在smem
                "r"(idesc),
                "r"(0u),
                "r"(tmem_sfa), "r"(tmem_sfb));
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

    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_base), "r"(32));
    }
}

int main() {
    float *d_out, h[128];
    CHECK_CUDA(cudaMalloc(&d_out, 4096));

    // ---- SS: NVFP4 block16 (A=smem, B=smem, 有scale) ----
    printf("====== SS: NVFP4 block16 (tcgen05.mma, sm_100) ======\n");
    printf("kind::mxf4nvf4.block_scale.scale_vec::4X\n");
    printf("A=smem, B=smem, scale=ue4m3(1.0)\n");
    printf("M=%d N=%d K=%d, expect D=64.0\n\n", M, N, K);

    CHECK_CUDA(cudaMemset(d_out, 0, 4096));
    tcgen05_nvfp4_kernel<<<1, THREADS, 8192>>>(d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n\n", cudaGetErrorString(e)); }
    else {
        CHECK_CUDA(cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost));
        int pass = 0;
        for (int i = 0; i < 128; i++) if (h[i] == 64.0f) pass++;
        printf("Result: %d/128 correct (=64.0)\n", pass);
        printf("D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n\n",
               h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    }

    // ---- TS: NVFP4 block16 (A=TMEM, B=smem, 有scale) ----
    printf("====== TS: NVFP4 block16 A=TMEM (tcgen05.mma, sm_100) ======\n");
    printf("kind::mxf4nvf4.block_scale.scale_vec::4X\n");
    printf("A=TMEM, B=smem, scale=ue4m3(1.0)\n");
    printf("M=%d N=%d K=%d, expect D=64.0\n", M, N, K);
    printf("Note: CUTLASS未实现此变体, 但硬件实测支持\n\n");

    CHECK_CUDA(cudaMemset(d_out, 0, 4096));
    tcgen05_nvfp4_ts_kernel<<<1, THREADS, 8192>>>(d_out);
    e = cudaDeviceSynchronize();
    if (e) { printf("Error: %s\n\n", cudaGetErrorString(e)); }
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
