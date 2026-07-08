/**
 * Minimal warp-specialization skeleton (no async copy, no MMA).
 *
 * 1 producer warp + 1 consumer warp synchronizing through two named barriers.
 * This is the bare sync pattern that CUTLASS Hopper/Blackwell GEMMs scale up:
 *   producer: [wait EMPTY] → load → arrive FULL → ...
 *   consumer: wait FULL    → consume → [arrive EMPTY] → ...
 *
 * Strip-down vs ws_tcgen05.cu:
 *   - no cp.async / TMA       (plain gmem loads)
 *   - no double-buffered smem (single stage)
 *   - no tcgen05.mma          (consumer just reduces floats)
 *   - no setmaxnreg           (1+1 warp, no register pressure)
 *
 * Compile:
 *   nvcc -arch=sm_80 -std=c++17 -O3 -o ws_min ws_minimal_named_barrier_skeleton.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

constexpr int TILE_SZ       = 256;
constexpr int N_TILES       = 4;
constexpr int PROD_THREADS  = 32;
constexpr int CONS_THREADS  = 32;
constexpr int TOTAL_THREADS = PROD_THREADS + CONS_THREADS;   // 64

// named-barrier resource IDs (0..15 per CTA)
#define BAR_FULL   0   // producer → consumer  (data ready)
#define BAR_EMPTY  1   // consumer → producer  (smem free)

__global__ void ws_min_kernel(const float* __restrict__ g_in,
                              float* __restrict__ g_out) {
    __shared__ float smem[TILE_SZ];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    if (warp_id == 0) {
        // -------- Producer warp --------
        for (int t = 0; t < N_TILES; ++t) {
            // wait: smem free (skip first tile — initially free)
            if (t > 0) {
                asm volatile("bar.sync %0, %1;"
                             :: "r"(BAR_EMPTY), "r"(TOTAL_THREADS));
            }

            // produce: gmem → smem
            #pragma unroll
            for (int i = lane; i < TILE_SZ; i += 32)
                smem[i] = g_in[t * TILE_SZ + i];

            // signal: data ready (non-blocking arrive)
            asm volatile("bar.arrive %0, %1;"
                         :: "r"(BAR_FULL), "r"(TOTAL_THREADS));
        }
    } else if (warp_id == 1) {
        // -------- Consumer warp --------
        float acc = 0.f;
        for (int t = 0; t < N_TILES; ++t) {
            // wait: data ready
            asm volatile("bar.sync %0, %1;"
                         :: "r"(BAR_FULL), "r"(TOTAL_THREADS));

            // consume: reduce tile
            #pragma unroll
            for (int i = lane; i < TILE_SZ; i += 32)
                acc += smem[i];

            // signal: smem free (skip after last tile)
            if (t < N_TILES - 1) {
                asm volatile("bar.arrive %0, %1;"
                             :: "r"(BAR_EMPTY), "r"(TOTAL_THREADS));
            }
        }

        // warp-reduce and write out
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) g_out[blockIdx.x] = acc;
    }
}

int main() {
    constexpr int N = TILE_SZ * N_TILES;
    float* h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h, N * sizeof(float), cudaMemcpyHostToDevice));

    ws_min_kernel<<<1, TOTAL_THREADS>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    float out = 0.f;
    CHECK_CUDA(cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("sum = %.1f  (expected %d.0)\n", out, N);

    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
