/*
 * Minimal tcgen05.mma cta_group::2 demo for NVFP4 block16 on Blackwell.
 *
 * Instruction:
 *   tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X
 *
 * Shape:
 *   M = 128, N = 16, K = 64
 *   cta_group::2 is a CTA pair: launch a 2-CTA cluster, 128 threads per CTA.
 *
 * Compile:
 *   nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -O2 \
 *        -o tcgen05_nvfp4_ctagroup2_2cta_m256 tcgen05_nvfp4_ctagroup2_2cta_m256.cu
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::printf("CUDA error at line %d: %s\n", __LINE__,                \
                        cudaGetErrorString(err));                                \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

constexpr int M = 128;
constexpr int N = 16;
constexpr int K = 64;
constexpr int THREADS = 128;
constexpr int CTA_GROUP = 2;
constexpr int A_BYTES = M * K / 2;
constexpr int B_BYTES = N * K / 2;
constexpr int OUT_FLOATS = 128;

__device__ uint32_t smem_u32(void const* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// SmemDescriptor: [0:14) addr>>4, [16:30) stride>>4, [46:48) version=1.
__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    uint64_t desc = 0;
    desc |= static_cast<uint64_t>((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    // ⚠️ no-swizzle 实际是 8行×16B core-matrix 分块(全1.0输入不敏感, 真实数据排布见 random_cpu_ref/)
    (void)stride_bytes;
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    desc |= static_cast<uint64_t>(1) << 46;
    return desc;
}

// InstrDescriptorBlockScaled for NVFP4: A/B=e2m1, scale=ue4m3.
__device__ uint32_t make_idesc(uint32_t tmem_sfa, uint32_t tmem_sfb) {
    uint32_t idesc = 0;
    idesc |= 1u << 7;                         // a_format = e2m1
    idesc |= 1u << 10;                        // b_format = e2m1
    idesc |= static_cast<uint32_t>(N / 8) << 17;  // n_dim = N / 8
    idesc |= static_cast<uint32_t>(M >> 7) << 27; // m_dim = M >> 7
    idesc |= ((tmem_sfa >> 30) & 3u) << 29;   // a_sf_id
    idesc |= ((tmem_sfb >> 30) & 3u) << 4;    // b_sf_id
    return idesc;
}

__global__ void tcgen05_nvfp4_cg2_kernel(float* out) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t cta_rank = cluster.block_rank();

    extern __shared__ char smem[];
    auto* smem_a = reinterpret_cast<uint8_t*>(smem);
    auto* smem_b = smem_a + A_BYTES;
    auto* smem_tmem = reinterpret_cast<uint32_t*>(smem_b + B_BYTES + 64);
    auto* mbar_done = reinterpret_cast<uint64_t*>(smem_b + B_BYTES + 128);

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;

    // e2m1 fp4 1.0 is encoded as 0x2; pack two fp4 values per byte.
    for (int i = tid; i < A_BYTES; i += THREADS) {
        smem_a[i] = 0x22;
    }
    for (int i = tid; i < B_BYTES; i += THREADS) {
        smem_b[i] = 0x22;
    }
    __syncthreads();

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :
                     : "r"(smem_u32(mbar_done)));
    }
    __syncthreads();

    if (warp_id == 0) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :
                     : "r"(smem_u32(smem_tmem)), "r"(32));
    }
    __syncthreads();
    cluster.sync();

    uint32_t tmem_base = *smem_tmem;
    uint32_t tmem_c = tmem_base;
    uint32_t tmem_sfa = tmem_base + 16;
    uint32_t tmem_sfb = tmem_base + 20;

    // ue4m3 scale 1.0 is 0x38.
    uint32_t sf = 0x38383838u;
    if (warp_id == 0) {
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     :
                     : "r"(tmem_sfa), "r"(sf), "r"(sf), "r"(sf), "r"(sf));
    }
    __syncthreads();
    if (warp_id == 0) {
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     :
                     : "r"(tmem_sfb), "r"(sf), "r"(sf), "r"(sf), "r"(sf));
    }
    __syncthreads();
    cluster.sync();

    uint64_t desc_a = make_desc(smem_a, K / 2);
    uint64_t desc_b = make_desc(smem_b, K / 2);
    uint32_t idesc = make_idesc(tmem_sfa, tmem_sfb);

    // For cta_group::2, a single thread from the CTA pair issues the MMA.
    if (cta_rank == 0 && tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(0u),
              "r"(tmem_sfa), "r"(tmem_sfb));
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
                     :
                     : "r"(smem_u32(mbar_done)));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
            "@p bra DONE_%=;\n\t"
            "bra WAIT_%=;\n\t"
            "DONE_%=:\n\t"
            "}\n"
            :
            : "r"(smem_u32(mbar_done)));
    }
    __syncthreads();
    cluster.sync();

    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n\t"
                     "tcgen05.wait::ld.sync.aligned;"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                     : "r"(tmem_c));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        int out_base = cta_rank * OUT_FLOATS + lane * 4;
        out[out_base + 0] = __uint_as_float(r0);
        out[out_base + 1] = __uint_as_float(r1);
        out[out_base + 2] = __uint_as_float(r2);
        out[out_base + 3] = __uint_as_float(r3);
    }

    __syncthreads();
    cluster.sync();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     :
                     : "r"(tmem_base), "r"(32));
    }
}

int main() {
    float* d_out = nullptr;
    float h_out[CTA_GROUP * OUT_FLOATS] = {};
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(h_out)));
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(h_out)));

    std::printf("====== tcgen05.mma cta_group::2 NVFP4 block16 ======\n");
    std::printf("M=%d N=%d K=%d cluster_ctas=%d threads_per_cta=%d, expect 64.0\n",
                M, N, K, CTA_GROUP, THREADS);

    CHECK_CUDA(cudaFuncSetAttribute(
        tcgen05_nvfp4_cg2_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = CTA_GROUP;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(CTA_GROUP, 1, 1);
    config.blockDim = dim3(THREADS, 1, 1);
    config.dynamicSmemBytes = 8192;
    config.attrs = &attr;
    config.numAttrs = 1;

    CHECK_CUDA(cudaLaunchKernelEx(&config, tcgen05_nvfp4_cg2_kernel, d_out));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    int pass[CTA_GROUP] = {};
    for (int i = 0; i < CTA_GROUP * OUT_FLOATS; ++i) {
        pass[i / OUT_FLOATS] += (h_out[i] == 64.0f);
    }

    std::printf("CTA0 result: %d/%d correct (=64.0)\n", pass[0], OUT_FLOATS);
    std::printf("CTA0 D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                h_out[0], h_out[1], h_out[2], h_out[3],
                h_out[4], h_out[5], h_out[6], h_out[7]);
    std::printf("CTA1 result: %d/%d correct (=64.0)\n", pass[1], OUT_FLOATS);
    std::printf("CTA1 D[0:8]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                h_out[OUT_FLOATS + 0], h_out[OUT_FLOATS + 1],
                h_out[OUT_FLOATS + 2], h_out[OUT_FLOATS + 3],
                h_out[OUT_FLOATS + 4], h_out[OUT_FLOATS + 5],
                h_out[OUT_FLOATS + 6], h_out[OUT_FLOATS + 7]);

    CHECK_CUDA(cudaFree(d_out));
    return pass[0] == OUT_FLOATS ? 0 : 2;
}
