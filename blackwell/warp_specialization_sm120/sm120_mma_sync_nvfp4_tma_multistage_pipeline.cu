/**
 * SM120 NVFP4 TMA multistage pipeline.
 *
 * CTA layout:
 *   warp 0   : producer, elected lane submits A/B TMA transfers
 *   warps 1-4: consumers, each computes one M16 x N64 output strip
 *
 * Tile layout:
 *   CTA output : M64 x N64
 *   one stage : A[64,64] + B[64,64], packed E2M1
 *   one MMA   : m16n8k64, E2M1 + UE4M3 block scale -> FP32
 *
 * Build example:
 *   nvcc -gencode arch=compute_120a,code=sm_120a -std=c++17 -O3 \
 *     -DPIPELINE_STAGES=3 -DPIPELINE_K_ITERS=8 -o t this_file.cu
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t error_ = (call);                                                               \
        if (error_ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
                         cudaGetErrorString(error_));                                              \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

#ifndef PIPELINE_STAGES
#define PIPELINE_STAGES 3
#endif
#ifndef PIPELINE_K_ITERS
#define PIPELINE_K_ITERS 8
#endif

constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kKTile = 64;
constexpr int kStages = PIPELINE_STAGES;
constexpr int kKIters = PIPELINE_K_ITERS;
constexpr int kProducerWarps = 1;
constexpr int kConsumerWarps = 4;
constexpr int kThreads = (kProducerWarps + kConsumerWarps) * 32;
constexpr int kABytesPerStage = kM * kKTile / 2;
constexpr int kBBytesPerStage = kN * kKTile / 2;
constexpr int kOutputElements = kM * kN;

static_assert(kStages >= 2 && kStages <= 4, "PIPELINE_STAGES must be 2..4");
static_assert(kStages <= kKIters, "PIPELINE_STAGES must not exceed K iterations");

#define EMPTY_BARRIER(stage) ((stage) * 2 + 1)

// Convert a generic pointer into the 32-bit shared-memory address expected by PTX.
__device__ __forceinline__ uint32_t shared_address(const void *pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void tma_load_1d(void *destination_shared, const void *source_global,
                                            uint64_t *completion_barrier, uint32_t bytes) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];" ::"r"(shared_address(destination_shared)),
                 "l"(source_global), "r"(bytes), "r"(shared_address(completion_barrier))
                 : "memory");
}

__device__ __forceinline__ void barrier_expect_bytes(uint64_t *barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;" ::"r"(bytes),
                 "r"(shared_address(barrier))
                 : "memory");
}

__device__ __forceinline__ void barrier_wait_phase(uint64_t *barrier, uint32_t phase) {
    asm volatile("{\n\t"
                 ".reg .pred ready;\n\t"
                 "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1;\n\t"
                 "@!ready bra WAIT_%=;\n\t"
                 "}" ::"r"(shared_address(barrier)),
                 "r"(phase)
                 : "memory");
}

// One native SM120 NVFP4 atom: D[16x8] += scaled A[16x64] * scaled B[64x8].
// The four FP32 accumulator registers belong to one lane.
__device__ __forceinline__ void issue_nvfp4_mma(float (&accumulator)[4], uint32_t a0, uint32_t a1,
                                                uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
                                                uint32_t scale_a, uint32_t scale_b) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{%11,%12},{%13},{%14,%15};"
        : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "h"(uint16_t{0}),
          "h"(uint16_t{0}), "r"(scale_b), "h"(uint16_t{0}), "h"(uint16_t{0}));
}

__global__ void nvfp4_tma_pipeline_kernel(const uint8_t *global_a, const uint8_t *global_b,
                                          float *global_d, uint32_t scale_a, uint32_t scale_b) {
    // Shared-memory ring layout: [A stage 0..S-1][B stage 0..S-1][barriers].
    extern __shared__ __align__(128) uint8_t shared_storage[];

    uint8_t *shared_a[kStages];
    uint8_t *shared_b[kStages];
    uint8_t *a_base = shared_storage;
    uint8_t *b_base = a_base + kStages * kABytesPerStage;
    for (int stage = 0; stage < kStages; ++stage) {
        shared_a[stage] = a_base + stage * kABytesPerStage;
        shared_b[stage] = b_base + stage * kBBytesPerStage;
    }
    uint64_t *load_barriers = reinterpret_cast<uint64_t *>(b_base + kStages * kBBytesPerStage);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // One transaction barrier per stage. Only the first S threads initialize them.
    if (threadIdx.x < kStages) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
            shared_address(&load_barriers[threadIdx.x])));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // Warp specialization: warp 0 produces data; warps 1-4 consume it.
    if (warp_id == 0) {
        // Producer loop combines prologue and steady state:
        //   k_iter < S  : fill initially empty stages (prologue)
        //   k_iter >= S : wait for consumer release, then reuse the ring slot
        for (int k_iter = 0; k_iter < kKIters; ++k_iter) {
            const int stage = k_iter % kStages;
            if (k_iter >= kStages) {
                asm volatile("bar.sync %0, %1;" ::"r"(EMPTY_BARRIER(stage)), "n"(kThreads));
            }
            if (lane_id == 0) {
                tma_load_1d(shared_a[stage], global_a + k_iter * kABytesPerStage,
                            &load_barriers[stage], kABytesPerStage);
                tma_load_1d(shared_b[stage], global_b + k_iter * kBBytesPerStage,
                            &load_barriers[stage], kBBytesPerStage);
                barrier_expect_bytes(&load_barriers[stage], kABytesPerStage + kBBytesPerStage);
            }
        }
        return;
    }

    // Consumer warp i owns rows [16*i, 16*i+16) and all 64 columns.
    // Eight independent accumulator fragments cover N64 as 8 x N8 atoms.
    const int consumer_id = warp_id - 1;
    const int row_base = consumer_id * 16;
    uint32_t load_phase[kStages] = {};
    float accumulators[8][4] = {};

    for (int k_iter = 0; k_iter < kKIters; ++k_iter) {
        const int stage = k_iter % kStages;
        barrier_wait_phase(&load_barriers[stage], load_phase[stage]);
        load_phase[stage] ^= 1u;

        // Load the lane-specific packed E2M1 fragments from shared memory.
        const uint32_t *a_words = reinterpret_cast<const uint32_t *>(shared_a[stage]);
        const uint32_t *b_words = reinterpret_cast<const uint32_t *>(shared_b[stage]);

        const int a_row = lane_id / 4;
        const int a_pack = lane_id % 4;
        const uint32_t a0 = a_words[(row_base + a_row) * 8 + a_pack];
        const uint32_t a1 = a_words[(row_base + a_row + 8) * 8 + a_pack];
        const uint32_t a2 = a_words[(row_base + a_row) * 8 + a_pack + 4];
        const uint32_t a3 = a_words[(row_base + a_row + 8) * 8 + a_pack + 4];

// Reuse the same A fragment across eight N8 output atoms.
#pragma unroll
        for (int n_tile = 0; n_tile < 8; ++n_tile) {
            const int b_column = n_tile * 8 + lane_id / 4;
            const int b_pack = lane_id % 4;
            const uint32_t b0 = b_words[b_column * 8 + b_pack];
            const uint32_t b1 = b_words[b_column * 8 + b_pack + 4];
            issue_nvfp4_mma(accumulators[n_tile], a0, a1, a2, a3, b0, b1, scale_a, scale_b);
        }

        __syncwarp();
        asm volatile("bar.arrive %0, %1;" ::"r"(EMPTY_BARRIER(stage)), "n"(kThreads));
    }

    // Minimal epilogue: consumer warps directly store FP32 accumulators.
    for (int n_tile = 0; n_tile < 8; ++n_tile) {
        const int row0 = row_base + lane_id / 4;
        const int row1 = row0 + 8;
        const int column0 = n_tile * 8 + (lane_id % 4) * 2;
        global_d[row0 * kN + column0] = accumulators[n_tile][0];
        global_d[row0 * kN + column0 + 1] = accumulators[n_tile][1];
        global_d[row1 * kN + column0] = accumulators[n_tile][2];
        global_d[row1 * kN + column0 + 1] = accumulators[n_tile][3];
    }
}

// Standalone exact test: packed 0x2 is E2M1 1.0 and UE4M3 0x38 is scale 1.0.
int main() {
    const size_t a_bytes = size_t{kKIters} * kABytesPerStage;
    const size_t b_bytes = size_t{kKIters} * kBBytesPerStage;
    uint8_t *device_a, *device_b;
    float *device_d;
    CHECK_CUDA(cudaMalloc(&device_a, a_bytes));
    CHECK_CUDA(cudaMalloc(&device_b, b_bytes));
    CHECK_CUDA(cudaMalloc(&device_d, kOutputElements * sizeof(float)));
    CHECK_CUDA(cudaMemset(device_a, 0x22, a_bytes)); // packed E2M1 1.0
    CHECK_CUDA(cudaMemset(device_b, 0x22, b_bytes));

    const int shared_bytes =
        kStages * (kABytesPerStage + kBBytesPerStage) + kStages * sizeof(uint64_t);
    CHECK_CUDA(cudaFuncSetAttribute(nvfp4_tma_pipeline_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));

    constexpr uint32_t kScaleOne = 0x38383838u;
    nvfp4_tma_pipeline_kernel<<<1, kThreads, shared_bytes>>>(device_a, device_b, device_d,
                                                             kScaleOne, kScaleOne);
    CHECK_CUDA(cudaDeviceSynchronize());

    float host_d[kOutputElements];
    CHECK_CUDA(cudaMemcpy(host_d, device_d, sizeof(host_d), cudaMemcpyDeviceToHost));
    int errors = 0;
    const float expected = float(kKIters * kKTile);
    for (float value : host_d)
        errors += value != expected;
    std::printf("SM120 NVFP4 TMA stages=%d: %s (%d/%d exact)\n", kStages, errors ? "FAIL" : "PASS",
                kOutputElements - errors, kOutputElements);
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
