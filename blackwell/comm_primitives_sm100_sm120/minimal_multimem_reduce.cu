/*
 * Minimal 2-GPU CUDA multicast / multimem reduce demo.
 *
 * It creates one multicast object backed by one physical allocation on GPU0 and
 * one physical allocation on GPU1.  Each GPU writes its local replica through a
 * normal unicast mapping, then both GPUs read the multicast mapping with:
 *
 *   multimem.ld_reduce.relaxed.sys.global.add.f32
 *
 * Expected output:
 *   GPU0 reduce result: 30.0
 *   GPU1 reduce result: 30.0
 *
 * Compile:
 *   nvcc -std=c++17 -O2 -arch=sm_90 -lcuda -o minimal_multimem_reduce minimal_multimem_reduce.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess)                                               \
        {                                                                      \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err__));                                     \
            std::exit(1);                                                       \
        }                                                                      \
    } while (0)

#define CHECK_CU(call)                                                          \
    do                                                                         \
    {                                                                          \
        CUresult err__ = (call);                                                \
        if (err__ != CUDA_SUCCESS)                                              \
        {                                                                      \
            char const* name__ = nullptr;                                       \
            char const* msg__ = nullptr;                                        \
            cuGetErrorName(err__, &name__);                                     \
            cuGetErrorString(err__, &msg__);                                    \
            std::fprintf(stderr, "CUDA driver error %s:%d: %s (%s)\n",          \
                __FILE__, __LINE__, name__ ? name__ : "unknown",               \
                msg__ ? msg__ : "no message");                                 \
            std::exit(1);                                                       \
        }                                                                      \
    } while (0)

static size_t round_up(size_t x, size_t alignment)
{
    return ((x + alignment - 1) / alignment) * alignment;
}

__global__ void init_one_float(float* ptr, float value)
{
    if (threadIdx.x == 0)
    {
        ptr[0] = value;
    }
}

__global__ void multimem_reduce_one_float(float const* mc_ptr, float* out)
{
    if (threadIdx.x == 0)
    {
        float sum = 0.0f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                     : "=f"(sum)
                     : "l"(mc_ptr)
                     : "memory");
#else
        asm volatile("trap;");
#endif
        out[0] = sum;
    }
}

int main()
{
    constexpr int kNumDevices = 2;
    constexpr int kDeviceIds[kNumDevices] = {0, 1};

    int runtime_device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&runtime_device_count));
    if (runtime_device_count < kNumDevices)
    {
        std::fprintf(stderr, "Need at least 2 CUDA devices, found %d\n", runtime_device_count);
        return 1;
    }

    CHECK_CU(cuInit(0));

    CUdevice cu_devs[kNumDevices];
    for (int i = 0; i < kNumDevices; ++i)
    {
        CHECK_CU(cuDeviceGet(&cu_devs[i], kDeviceIds[i]));

        int multicast_supported = 0;
        CHECK_CU(cuDeviceGetAttribute(
            &multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cu_devs[i]));
        if (!multicast_supported)
        {
            std::fprintf(stderr, "GPU%d does not support CUDA multicast objects\n", kDeviceIds[i]);
            return 1;
        }
    }

    CUmulticastObjectProp mc_prop = {};
    mc_prop.numDevices = kNumDevices;
    mc_prop.handleTypes = 0;

    size_t mc_granularity = 0;
    CHECK_CU(cuMulticastGetGranularity(
        &mc_granularity, &mc_prop, CU_MULTICAST_GRANULARITY_MINIMUM));

    CUmemAllocationProp alloc_props[kNumDevices] = {};
    size_t alloc_granularity = 0;
    for (int i = 0; i < kNumDevices; ++i)
    {
        alloc_props[i].type = CU_MEM_ALLOCATION_TYPE_PINNED;
        alloc_props[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        alloc_props[i].location.id = kDeviceIds[i];
        alloc_props[i].requestedHandleTypes = static_cast<CUmemAllocationHandleType>(0);

        size_t g = 0;
        CHECK_CU(cuMemGetAllocationGranularity(&g, &alloc_props[i], CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        alloc_granularity = std::max(alloc_granularity, g);
    }

    size_t alloc_size = round_up(sizeof(float), std::max(mc_granularity, alloc_granularity));
    mc_prop.size = alloc_size;

    CUmemGenericAllocationHandle mc_handle = {};
    CHECK_CU(cuMulticastCreate(&mc_handle, &mc_prop));

    for (int i = 0; i < kNumDevices; ++i)
    {
        CHECK_CU(cuMulticastAddDevice(mc_handle, cu_devs[i]));
    }

    CUmemGenericAllocationHandle mem_handles[kNumDevices] = {};
    CUdeviceptr uc_ptrs[kNumDevices] = {};
    for (int i = 0; i < kNumDevices; ++i)
    {
        CHECK_CU(cuMemCreate(&mem_handles[i], alloc_size, &alloc_props[i], 0));
        CHECK_CU(cuMulticastBindMem(mc_handle, 0, mem_handles[i], 0, alloc_size, 0));

        CHECK_CU(cuMemAddressReserve(&uc_ptrs[i], alloc_size, 0, 0, 0));
        CHECK_CU(cuMemMap(uc_ptrs[i], alloc_size, 0, mem_handles[i], 0));

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = kDeviceIds[i];
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CHECK_CU(cuMemSetAccess(uc_ptrs[i], alloc_size, &access, 1));
    }

    CUdeviceptr mc_ptr = {};
    CHECK_CU(cuMemAddressReserve(&mc_ptr, alloc_size, 0, 0, 0));
    CHECK_CU(cuMemMap(mc_ptr, alloc_size, 0, mc_handle, 0));

    CUmemAccessDesc mc_access[kNumDevices] = {};
    for (int i = 0; i < kNumDevices; ++i)
    {
        mc_access[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        mc_access[i].location.id = kDeviceIds[i];
        mc_access[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CHECK_CU(cuMemSetAccess(mc_ptr, alloc_size, mc_access, kNumDevices));

    float* outputs[kNumDevices] = {};
    for (int i = 0; i < kNumDevices; ++i)
    {
        CHECK_CUDA(cudaSetDevice(kDeviceIds[i]));
        CHECK_CUDA(cudaMalloc(&outputs[i], sizeof(float)));
        CHECK_CUDA(cudaMemset(outputs[i], 0, sizeof(float)));
    }

    CHECK_CUDA(cudaSetDevice(kDeviceIds[0]));
    init_one_float<<<1, 1>>>(reinterpret_cast<float*>(uc_ptrs[0]), 10.0f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(kDeviceIds[1]));
    init_one_float<<<1, 1>>>(reinterpret_cast<float*>(uc_ptrs[1]), 20.0f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < kNumDevices; ++i)
    {
        CHECK_CUDA(cudaSetDevice(kDeviceIds[i]));
        multimem_reduce_one_float<<<1, 1>>>(reinterpret_cast<float const*>(mc_ptr), outputs[i]);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    for (int i = 0; i < kNumDevices; ++i)
    {
        float h = 0.0f;
        CHECK_CUDA(cudaSetDevice(kDeviceIds[i]));
        CHECK_CUDA(cudaMemcpy(&h, outputs[i], sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("GPU%d reduce result: %.1f\n", kDeviceIds[i], h);
    }

    for (int i = 0; i < kNumDevices; ++i)
    {
        CHECK_CUDA(cudaSetDevice(kDeviceIds[i]));
        CHECK_CUDA(cudaFree(outputs[i]));
    }

    CHECK_CU(cuMemUnmap(mc_ptr, alloc_size));
    CHECK_CU(cuMemAddressFree(mc_ptr, alloc_size));

    for (int i = 0; i < kNumDevices; ++i)
    {
        CHECK_CU(cuMemUnmap(uc_ptrs[i], alloc_size));
        CHECK_CU(cuMemAddressFree(uc_ptrs[i], alloc_size));
        CHECK_CU(cuMemRelease(mem_handles[i]));
    }
    CHECK_CU(cuMemRelease(mc_handle));

    return 0;
}
