/*
 * Minimal cross-GPU P2P acquire/release + volatile-load demo.
 *
 * What it demonstrates:
 *   GPU0 writes a float4 payload into GPU1 memory, then release-stores a flag.
 *   GPU1 acquire-loads the flag until ready, then volatile-loads the float4.
 *
 * Compile:
 *   nvcc -std=c++17 -O2 -arch=sm_90 -o minimal_p2p_acqrel_volatile minimal_p2p_acqrel_volatile.cu
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess)                                                \
        {                                                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err__));                                      \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

struct alignas(16) Mailbox
{
    float4 data;
    uint32_t flag;
    uint32_t padding[3];
};

constexpr uint32_t READY_VALUE = 2;

__device__ __forceinline__ void store_release_sys_u32(uint32_t* addr, uint32_t value)
{
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(value), "l"(addr) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_sys_u32(uint32_t const* addr)
{
    uint32_t value;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(value) : "l"(addr) : "memory");
    return value;
}

__device__ __forceinline__ float4 load_volatile_float4(float4 const* addr)
{
    float4 value;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                 : "l"(addr)
                 : "memory");
    return value;
}

__global__ void producer_kernel(Mailbox* remote_box)
{
    if (threadIdx.x == 0)
    {
        remote_box->data = make_float4(1.0f, 2.0f, 3.0f, 4.0f);

        // Publish the payload. The release/sys store orders the data write before the ready flag
        // at system scope, which is the scope needed for cross-GPU communication.
        store_release_sys_u32(&remote_box->flag, READY_VALUE);
    }
}

__global__ void consumer_kernel(Mailbox* local_box, float4* out)
{
    if (threadIdx.x == 0)
    {
        while (load_acquire_sys_u32(&local_box->flag) != READY_VALUE)
        {
        }

        // Force an actual memory load in the polling/communication style used by custom AR kernels.
        out[0] = load_volatile_float4(&local_box->data);
    }
}

static void enable_peer_access(int src, int dst)
{
    int can_access = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, src, dst));
    if (!can_access)
    {
        std::fprintf(stderr, "Device %d cannot access peer device %d\n", src, dst);
        std::exit(1);
    }

    CHECK_CUDA(cudaSetDevice(src));
    cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
    {
        CHECK_CUDA(err);
    }
    cudaGetLastError(); // Clear cudaErrorPeerAccessAlreadyEnabled if it happened.
}

int main()
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2)
    {
        std::fprintf(stderr, "Need at least 2 CUDA devices, found %d\n", device_count);
        return 1;
    }

    constexpr int producer_dev = 0;
    constexpr int consumer_dev = 1;

    enable_peer_access(producer_dev, consumer_dev);

    Mailbox* box_on_consumer = nullptr;
    float4* out_on_consumer = nullptr;

    CHECK_CUDA(cudaSetDevice(consumer_dev));
    CHECK_CUDA(cudaMalloc(&box_on_consumer, sizeof(Mailbox)));
    CHECK_CUDA(cudaMalloc(&out_on_consumer, sizeof(float4)));
    CHECK_CUDA(cudaMemset(box_on_consumer, 0, sizeof(Mailbox)));
    CHECK_CUDA(cudaMemset(out_on_consumer, 0, sizeof(float4)));

    cudaStream_t producer_stream;
    cudaStream_t consumer_stream;

    CHECK_CUDA(cudaSetDevice(producer_dev));
    CHECK_CUDA(cudaStreamCreate(&producer_stream));

    CHECK_CUDA(cudaSetDevice(consumer_dev));
    CHECK_CUDA(cudaStreamCreate(&consumer_stream));
    consumer_kernel<<<1, 1, 0, consumer_stream>>>(box_on_consumer, out_on_consumer);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaSetDevice(producer_dev));
    producer_kernel<<<1, 1, 0, producer_stream>>>(box_on_consumer);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(producer_stream));

    CHECK_CUDA(cudaSetDevice(consumer_dev));
    CHECK_CUDA(cudaStreamSynchronize(consumer_stream));

    float4 out;
    CHECK_CUDA(cudaMemcpy(&out, out_on_consumer, sizeof(out), cudaMemcpyDeviceToHost));

    std::printf("consumer read: %.1f %.1f %.1f %.1f\n", out.x, out.y, out.z, out.w);
    bool ok = out.x == 1.0f && out.y == 2.0f && out.z == 3.0f && out.w == 4.0f;
    std::printf("%s\n", ok ? "PASS" : "FAIL");

    CHECK_CUDA(cudaSetDevice(producer_dev));
    CHECK_CUDA(cudaStreamDestroy(producer_stream));

    CHECK_CUDA(cudaSetDevice(consumer_dev));
    CHECK_CUDA(cudaStreamDestroy(consumer_stream));
    CHECK_CUDA(cudaFree(out_on_consumer));
    CHECK_CUDA(cudaFree(box_on_consumer));

    return ok ? 0 : 1;
}
