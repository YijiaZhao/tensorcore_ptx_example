/*
 * Minimal TMA (cp.async.bulk) unit -- the exact data-movement primitive
 * DeepGEMM MegaMoE uses to pull tokens across GPUs.
 *
 * Two-hop bulk copy (mirrors deep_gemm/ptx/tma.cuh tma_load_1d + tma_store_1d):
 *   (1) remote/local global --cp.async.bulk--> smem   (async, mbarrier-completed)
 *   (2) smem                --cp.async.bulk--> local global
 *
 * The 1D "bulk" TMA variant needs NO tensor-map descriptor: raw pointers +
 * an mbarrier that tracks the transaction byte count. That is the smallest
 * possible TMA unit.
 *
 * If run with >=2 GPUs and P2P available, the SOURCE buffer is placed on GPU1
 * and pulled by a kernel on GPU0 over NVLink -- identical to DeepGEMM dispatch.
 * With 1 GPU it degenerates to an intra-GPU global->smem->global bulk copy.
 *
 * Build (B200 = sm_100):
 *   nvcc -std=c++17 -O3 -arch=sm_100a -o minimal_p2p_tma minimal_p2p_tma.cu
 * (Hopper: -arch=sm_90a)
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t e__ = (call);                                              \
        if (e__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                         cudaGetErrorString(e__));                             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---- TMA / mbarrier PTX (copied from deep_gemm/ptx/tma.cuh) ----------------

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

// Issue an async bulk copy: global -> shared, completion tracked by mbarrier.
__device__ __forceinline__ void tma_load_1d(void* dst_smem, const void* src_global,
                                            uint64_t* mbar, uint32_t nbytes) {
    uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n" ::"r"(d),
        "l"(src_global), "r"(nbytes), "r"(m)
        : "memory");
}

// Tell the mbarrier how many bytes to expect for this transaction.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t nbytes) {
    uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n" ::"r"(nbytes),
                 "r"(m));
}

// Spin until the mbarrier phase flips (transaction complete).
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n\t"
        ".reg .pred P;\n\t"
        "W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1, %2;\n\t"
        "@P bra D;\n\t"
        "bra W;\n\t"
        "D:\n\t"
        "}" ::"r"(m),
        "r"(phase), "r"(0x989680));
}

// Issue an async bulk copy: shared -> global.
__device__ __forceinline__ void tma_store_1d(void* dst_global, const void* src_smem,
                                             uint32_t nbytes) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src_smem));
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(dst_global),
        "r"(s), "r"(nbytes)
        : "memory");
}

__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

// ---- Kernel: pull `nbytes` from src (possibly a peer/GPU1 ptr) through smem to dst
// One CTA, chunked to CHUNK bytes per TMA transaction (like DeepGEMM's kNumBytesPerPull).
template <uint32_t CHUNK>
__global__ void tma_pull_kernel(const uint8_t* __restrict__ src,
                                uint8_t* __restrict__ dst, uint32_t nbytes) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t mbar;

    if (threadIdx.x == 0) mbarrier_init(&mbar, 1);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // barrier init visible to TMA

    uint32_t phase = 0;
    for (uint32_t off = 0; off < nbytes; off += CHUNK) {
        uint32_t bytes = (nbytes - off < CHUNK) ? (nbytes - off) : CHUNK;
        if (threadIdx.x == 0) {
            tma_load_1d(smem, src + off, &mbar, bytes);   // global(remote) -> smem
            mbarrier_arrive_expect_tx(&mbar, bytes);
            mbarrier_wait(&mbar, phase);                   // wait load complete
            tma_store_1d(dst + off, smem, bytes);          // smem -> global(local)
            tma_store_commit();
            tma_store_wait();
        }
        __syncthreads();
        phase ^= 1;
    }
}

static bool try_enable_peer(int src, int dst) {
    int can = 0;
    CHECK(cudaDeviceCanAccessPeer(&can, src, dst));
    if (!can) return false;
    CHECK(cudaSetDevice(src));
    cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK(e);
    cudaGetLastError();
    return true;
}

int main() {
    constexpr uint32_t CHUNK = 16 * 1024;          // 16 KiB per TMA transaction
    constexpr uint32_t N = 1u << 20;               // 1 Mi elements
    constexpr uint32_t BYTES = N * sizeof(float);

    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    printf("Found %d GPU(s)\n", ndev);

    bool p2p = false;
    int src_dev = 0;
    if (ndev >= 2) p2p = try_enable_peer(0, 1);
    printf("Mode: %s\n", p2p ? "cross-GPU pull (src on GPU1, kernel on GPU0)"
                             : "single-GPU global->smem->global");

    // Source buffer: on GPU1 if P2P, else on GPU0.
    src_dev = p2p ? 1 : 0;
    CHECK(cudaSetDevice(src_dev));
    float* d_src = nullptr;
    CHECK(cudaMalloc(&d_src, BYTES));

    // Fill source with a known pattern.
    float* h = (float*)malloc(BYTES);
    for (uint32_t i = 0; i < N; ++i) h[i] = (float)(i % 997) + 0.25f;
    CHECK(cudaMemcpy(d_src, h, BYTES, cudaMemcpyHostToDevice));

    // Destination + kernel run on GPU0.
    CHECK(cudaSetDevice(0));
    float* d_dst = nullptr;
    CHECK(cudaMalloc(&d_dst, BYTES));
    CHECK(cudaMemset(d_dst, 0, BYTES));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    tma_pull_kernel<CHUNK><<<1, 32, CHUNK>>>((const uint8_t*)d_src, (uint8_t*)d_dst, BYTES);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // Verify.
    float* out = (float*)malloc(BYTES);
    CHECK(cudaMemcpy(out, d_dst, BYTES, cudaMemcpyDeviceToHost));
    uint32_t errors = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (out[i] != h[i]) {
            if (errors < 5) printf("  mismatch @%u: got %f want %f\n", i, out[i], h[i]);
            ++errors;
        }

    printf("\n[TMA bulk copy] %u bytes, chunk=%u : %s  (%.3f ms, %.1f GB/s)\n",
           BYTES, CHUNK, errors ? "FAIL" : "OK", ms, BYTES / (ms * 1e6));
    printf("Result: %s\n", errors ? "FAILED" : "ALL PASS");

    free(h);
    free(out);
    CHECK(cudaFree(d_dst));
    CHECK(cudaSetDevice(src_dev));
    CHECK(cudaFree(d_src));
    return errors ? 1 : 0;
}
