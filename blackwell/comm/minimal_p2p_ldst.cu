/*
 * Minimal GPU0 -> GPU1 P2P global load/store demo.
 *
 * GPU0:
 *   st.global.v4.f32        writes payload into GPU1 memory
 *   st.global.release.sys   publishes ready flag
 *
 * GPU1:
 *   ld.global.acquire.sys   waits for ready flag
 *   ld.global.v4.f32        reads payload from GPU1 memory
 *
 * Compile:
 *   nvcc -std=c++17 -O2 -arch=sm_90 -o minimal_p2p_ldst minimal_p2p_ldst.cu
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                             \
    do                                                                         \
    {                                                                          \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess)                                               \
        {                                                                      \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                cudaGetErrorString(err__));                                     \
            std::exit(1);                                                       \
        }                                                                      \
    } while (0)

struct alignas(16) Box
{
    float4 payload;
    uint32_t ready;
    uint32_t pad[3];
};

constexpr uint32_t READY_VALUE = 2;

__device__ __forceinline__ void st_global_v4_f32(float4* addr, float4 v)
{
    asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

__device__ __forceinline__ float4 ld_global_v4_f32(float4 const* addr)
{
    float4 v;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(addr)
                 : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys_u32(uint32_t* addr, uint32_t v)
{
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(v), "l"(addr) : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire_sys_u32(uint32_t const* addr)
{
    uint32_t v;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(v) : "l"(addr) : "memory");
    return v;
}

__global__ void gpu0_send_to_gpu1(Box* box_on_gpu1)
{
    if (threadIdx.x == 0)
    {
        st_global_v4_f32(&box_on_gpu1->payload, make_float4(1.f, 2.f, 3.f, 4.f));
        st_release_sys_u32(&box_on_gpu1->ready, READY_VALUE);
    }
}

__global__ void gpu1_recv(Box* box_on_gpu1, float4* out_on_gpu1)
{
    if (threadIdx.x == 0)
    {
        while (ld_acquire_sys_u32(&box_on_gpu1->ready) != READY_VALUE)
        {
        }
        out_on_gpu1[0] = ld_global_v4_f32(&box_on_gpu1->payload);
    }
}

static void enable_peer_access(int src, int dst)
{
    int can_access = 0;
    CHECK(cudaDeviceCanAccessPeer(&can_access, src, dst));
    if (!can_access)
    {
        std::fprintf(stderr, "GPU%d cannot access GPU%d memory by P2P\n", src, dst);
        std::exit(1);
    }

    CHECK(cudaSetDevice(src));
    cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
    {
        CHECK(e);
    }
    cudaGetLastError();
}

int main()
{
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 2)
    {
        std::fprintf(stderr, "need >= 2 GPUs, found %d\n", ndev);
        return 1;
    }

    enable_peer_access(0, 1);

    CHECK(cudaSetDevice(1));
    Box* box = nullptr;
    float4* out = nullptr;
    CHECK(cudaMalloc(&box, sizeof(Box)));
    CHECK(cudaMalloc(&out, sizeof(float4)));
    CHECK(cudaMemset(box, 0, sizeof(Box)));
    CHECK(cudaMemset(out, 0, sizeof(float4)));

    cudaStream_t s0, s1;
    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamCreate(&s0));

    CHECK(cudaSetDevice(1));
    CHECK(cudaStreamCreate(&s1));
    gpu1_recv<<<1, 1, 0, s1>>>(box, out);
    CHECK(cudaGetLastError());

    CHECK(cudaSetDevice(0));
    gpu0_send_to_gpu1<<<1, 1, 0, s0>>>(box);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(s0));

    CHECK(cudaSetDevice(1));
    CHECK(cudaStreamSynchronize(s1));

    float4 h;
    CHECK(cudaMemcpy(&h, out, sizeof(h), cudaMemcpyDeviceToHost));
    std::printf("GPU1 read: %.1f %.1f %.1f %.1f\n", h.x, h.y, h.z, h.w);

    CHECK(cudaFree(out));
    CHECK(cudaFree(box));
    CHECK(cudaStreamDestroy(s1));

    CHECK(cudaSetDevice(0));
    CHECK(cudaStreamDestroy(s0));
    return 0;
}
