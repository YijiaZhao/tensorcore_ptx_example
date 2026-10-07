/*
 * Minimal GPU-to-GPU copy through the Copy Engine (CE), compared with an SM copy.
 *
 * The Copy Engine is a DMA engine inside the GPU. It is not driven by any PTX
 * instruction: the host calls cudaMemcpyPeerAsync(), the driver writes a DMA
 * descriptor into the engine's command queue, and the engine moves the bytes
 * over NVLink or PCIe by itself. No SM executes anything during the transfer.
 *
 * This program moves the same buffer GPU_SRC -> GPU_DST three ways and times each:
 *   ce      : cudaMemcpyPeerAsync                              (Copy Engine DMA)
 *   sm push : kernel on GPU_SRC: ld.global local, st.global into the peer pointer
 *   sm pull : kernel on GPU_DST: ld.global from the peer pointer, st.global local
 * and checks the result on GPU_DST.
 *
 * Remote STORES are posted: the SM fires them and moves on, so push can approach
 * link speed with modest occupancy. Remote LOADS are round trips: every 16 B load
 * waits for the interconnect (several us across a PCIe root complex) and the number
 * of loads in flight per SM is small, so pull is bandwidth = in-flight bytes / latency.
 * The grid argument lets you see both effects; on a switch-less PCIe host too MANY
 * writers also hurt (root-complex contention), so push peaks at a small grid.
 * ../allreduce_sm100_sm120/README.md §4.6 has the measured table for an 8-GPU RTX 6000D box.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -arch=sm_100 -o minimal_p2p_copy_engine minimal_p2p_copy_engine.cu
 *   (any sm_80+ works; nothing here is Blackwell-specific)
 * Run:
 *   ./minimal_p2p_copy_engine [src_gpu=0] [dst_gpu=1] [bytes=67108864] [sm_grid=0 (auto)]
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                                     \
    do {                                                                                \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

// SM copy: every thread moves 16 bytes per iteration with one ld.global.v4 and one
// st.global.v4. Whichever of `src`/`dst` lives on the other GPU is reached through the
// same instruction; the interconnect is invisible to the ISA. Launched on the source GPU
// it is a PUSH (remote store); launched on the destination GPU it is a PULL (remote load).
__global__ void sm_copy_kernel(const int4* __restrict__ src, int4* __restrict__ dst, size_t n_int4) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n_int4; i += stride) {
        dst[i] = src[i];
    }
}

static void enable_peer_access(int from, int to) {
    int can = 0;
    CHECK(cudaDeviceCanAccessPeer(&can, from, to));
    if (!can) {
        std::fprintf(stderr, "GPU %d cannot access GPU %d memory (no P2P). Both CE and SM would fall back to host staging.\n", from, to);
        std::exit(1);
    }
    CHECK(cudaSetDevice(from));
    cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK(e);
    cudaGetLastError();   // clear "already enabled"
}

// time `iters` runs of fn() on `stream`, return microseconds per run
template <typename F>
static double time_us(cudaStream_t stream, int iters, F fn) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 3; i++) fn();                // warm-up
    CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < iters; i++) fn();
    CHECK(cudaEventRecord(b, stream));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return ms * 1e3 / iters;
}

static bool verify(int dst_gpu, const uint8_t* d_dst, const std::vector<uint8_t>& expect) {
    std::vector<uint8_t> got(expect.size());
    CHECK(cudaSetDevice(dst_gpu));
    CHECK(cudaMemcpy(got.data(), d_dst, got.size(), cudaMemcpyDeviceToHost));
    return got == expect;
}

int main(int argc, char** argv) {
    const int    src_gpu = argc > 1 ? std::atoi(argv[1]) : 0;
    const int    dst_gpu = argc > 2 ? std::atoi(argv[2]) : 1;
    const size_t bytes   = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : (size_t(64) << 20);
    const int    sm_grid = argc > 4 ? std::atoi(argv[4]) : 0;   // blocks for the SM copy; 0 = auto
    if (bytes % 16 != 0) { std::fprintf(stderr, "bytes must be a multiple of 16\n"); return 1; }

    // P2P mapping is needed by BOTH paths: without it cudaMemcpyPeerAsync silently stages
    // through host memory, and the SM path cannot dereference the peer pointer at all.
    enable_peer_access(src_gpu, dst_gpu);
    enable_peer_access(dst_gpu, src_gpu);

    // ---- buffers
    std::vector<uint8_t> h_src(bytes);
    for (size_t i = 0; i < bytes; i++) h_src[i] = uint8_t(i * 2654435761u >> 13);

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    CHECK(cudaSetDevice(src_gpu));
    CHECK(cudaMalloc(&d_src, bytes));
    CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaSetDevice(dst_gpu));
    CHECK(cudaMalloc(&d_dst, bytes));

    // Work is issued from the SOURCE GPU's stream for both paths (PUSH direction).
    CHECK(cudaSetDevice(src_gpu));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // ---- path 1: Copy Engine. One API call = one DMA descriptor; no kernel is launched.
    CHECK(cudaMemsetAsync(d_dst, 0, bytes, stream));   // (memset also runs on a CE)
    auto ce_copy = [&] { CHECK(cudaMemcpyPeerAsync(d_dst, dst_gpu, d_src, src_gpu, bytes, stream)); };
    ce_copy();
    CHECK(cudaStreamSynchronize(stream));
    const bool ce_ok = verify(dst_gpu, d_dst, h_src);
    CHECK(cudaSetDevice(src_gpu));
    const double ce_us = time_us(stream, 20, ce_copy);

    // ---- path 2: SM. A kernel on the source GPU stores straight into the peer's memory.
    CHECK(cudaMemsetAsync(d_dst, 0, bytes, stream));
    const size_t n_int4 = bytes / 16;
    const int    tpb    = 256;
    const int    grid   = sm_grid > 0 ? sm_grid : int(std::min<size_t>((n_int4 + tpb - 1) / tpb, 1024));
    auto sm_copy = [&] {
        sm_copy_kernel<<<grid, tpb, 0, stream>>>(reinterpret_cast<const int4*>(d_src),
                                                 reinterpret_cast<int4*>(d_dst), n_int4);
    };
    sm_copy();
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream));
    const bool sm_ok = verify(dst_gpu, d_dst, h_src);
    CHECK(cudaSetDevice(src_gpu));
    const double sm_us = time_us(stream, 20, sm_copy);

    // ---- path 3: SM pull. The same kernel launched on the DESTINATION GPU: loads are remote.
    CHECK(cudaSetDevice(dst_gpu));
    cudaStream_t dst_stream;
    CHECK(cudaStreamCreate(&dst_stream));
    CHECK(cudaMemsetAsync(d_dst, 0, bytes, dst_stream));
    auto sm_pull = [&] {
        sm_copy_kernel<<<grid, tpb, 0, dst_stream>>>(reinterpret_cast<const int4*>(d_src),
                                                     reinterpret_cast<int4*>(d_dst), n_int4);
    };
    sm_pull();
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(dst_stream));
    const bool pull_ok = verify(dst_gpu, d_dst, h_src);
    CHECK(cudaSetDevice(dst_gpu));
    const double pull_us = time_us(dst_stream, 20, sm_pull);
    CHECK(cudaStreamDestroy(dst_stream));

    // ---- report
    const double gib = double(bytes) / (1u << 30);
    std::printf("GPU %d -> GPU %d, %zu bytes\n", src_gpu, dst_gpu, bytes);
    std::printf("  copy engine (cudaMemcpyPeerAsync): %9.1f us  %6.1f GB/s  %s\n",
                ce_us, double(bytes) / ce_us * 1e-3, ce_ok ? "OK" : "MISMATCH");
    std::printf("  SM push (remote st.global)       : %9.1f us  %6.1f GB/s  %s  (grid %d x %d)\n",
                sm_us, double(bytes) / sm_us * 1e-3, sm_ok ? "OK" : "MISMATCH", grid, tpb);
    std::printf("  SM pull (remote ld.global)       : %9.1f us  %6.1f GB/s  %s  (grid %d x %d)\n",
                pull_us, double(bytes) / pull_us * 1e-3, pull_ok ? "OK" : "MISMATCH", grid, tpb);
    (void)gib;

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_src));
    CHECK(cudaSetDevice(dst_gpu));
    CHECK(cudaFree(d_dst));
    return (ce_ok && sm_ok && pull_ok) ? 0 : 2;
}
