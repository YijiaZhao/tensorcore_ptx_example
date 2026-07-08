// Build:
//   nvcc -std=c++17 check_multimem_nvlink_multicast.cu -lcuda -o check_multimem_nvlink_multicast
//
// Run:
//   ./check_multimem_nvlink_multicast           # check all visible CUDA devices as one team
//   ./check_multimem_nvlink_multicast 0 1 2 3   # check an explicit multicast team

#include <cuda.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CU(call)                                                     \
    do {                                                                   \
        CUresult _status = (call);                                         \
        if (_status != CUDA_SUCCESS) {                                     \
            const char* _name = nullptr;                                   \
            const char* _desc = nullptr;                                   \
            cuGetErrorName(_status, &_name);                               \
            cuGetErrorString(_status, &_desc);                             \
            std::cerr << "CUDA driver error at " << #call << ": "         \
                      << (_name ? _name : "unknown") << " - "             \
                      << (_desc ? _desc : "unknown") << std::endl;         \
            return 2;                                                      \
        }                                                                  \
    } while (0)

static std::string cu_result_name(CUresult status) {
    const char* name = nullptr;
    cuGetErrorName(status, &name);
    return name ? name : "UNKNOWN_CUDA_ERROR";
}

static std::string cu_result_desc(CUresult status) {
    const char* desc = nullptr;
    cuGetErrorString(status, &desc);
    return desc ? desc : "";
}

static int get_attr(CUdevice dev, CUdevice_attribute attr) {
    int value = 0;
    CUresult status = cuDeviceGetAttribute(&value, attr, dev);
    return status == CUDA_SUCCESS ? value : 0;
}

int main(int argc, char** argv) {
    CHECK_CU(cuInit(0));

    int driver_version = 0;
    CHECK_CU(cuDriverGetVersion(&driver_version));
    std::cout << "CUDA driver version: " << driver_version << std::endl;

    int device_count = 0;
    CHECK_CU(cuDeviceGetCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    std::vector<int> ordinals;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            int ordinal = std::atoi(argv[i]);
            if (ordinal < 0 || ordinal >= device_count) {
                std::cerr << "Invalid CUDA device ordinal " << ordinal
                          << ", visible device count is " << device_count << std::endl;
                return 1;
            }
            ordinals.push_back(ordinal);
        }
    } else {
        for (int i = 0; i < device_count; ++i) {
            ordinals.push_back(i);
        }
    }

    if (ordinals.size() < 2) {
        std::cerr << "Need at least 2 devices for a useful multimem/multicast team." << std::endl;
        return 1;
    }

    std::vector<CUdevice> devices;
    bool attrs_ok = true;

    std::cout << "\nPer-device capability:" << std::endl;
    for (int ordinal : ordinals) {
        CUdevice dev;
        CHECK_CU(cuDeviceGet(&dev, ordinal));
        devices.push_back(dev);

        char name[256] = {};
        CHECK_CU(cuDeviceGetName(name, sizeof(name), dev));

        int cc_major = get_attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        int cc_minor = get_attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        int vmm = get_attr(dev, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED);

#if CUDA_VERSION >= 12010
        int multicast = get_attr(dev, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED);
#else
        int multicast = 0;
#endif

        std::cout << "  GPU " << ordinal << ": " << name
                  << ", sm_" << cc_major << cc_minor
                  << ", VMM=" << vmm
                  << ", multicast=" << multicast << std::endl;

        if (!vmm || !multicast) {
            attrs_ok = false;
        }
    }

#if CUDA_VERSION < 12010
    std::cerr << "\nCUDA toolkit headers are older than 12.1, so multicast APIs/attributes "
              << "are unavailable at compile time." << std::endl;
    return 1;
#else
    if (!attrs_ok) {
        std::cerr << "\nResult: NOT usable for multimem/NVLink-SHARP multicast. "
                  << "At least one selected GPU lacks VMM or multicast support." << std::endl;
        return 1;
    }

    CUmulticastObjectProp prop = {};
    prop.numDevices = static_cast<unsigned int>(devices.size());
    prop.size = 1ull << 20;
    prop.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.flags = 0;

    size_t min_granularity = 0;
    CUresult status = cuMulticastGetGranularity(
        &min_granularity, &prop, CU_MULTICAST_GRANULARITY_MINIMUM);
    if (status != CUDA_SUCCESS) {
        std::cerr << "\ncuMulticastGetGranularity failed: " << cu_result_name(status)
                  << " - " << cu_result_desc(status) << std::endl;
        std::cerr << "Result: NOT usable for multimem on this selected team." << std::endl;
        return 1;
    }
    prop.size = min_granularity;

    CUmemGenericAllocationHandle mc_handle = 0;
    status = cuMulticastCreate(&mc_handle, &prop);
    if (status != CUDA_SUCCESS) {
        std::cerr << "\ncuMulticastCreate failed: " << cu_result_name(status)
                  << " - " << cu_result_desc(status) << std::endl;
        std::cerr << "Result: NOT usable for multimem on this selected team. "
                  << "Check NVSwitch/fabric manager/driver support." << std::endl;
        return 1;
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        status = cuMulticastAddDevice(mc_handle, devices[i]);
        if (status != CUDA_SUCCESS) {
            cuMemRelease(mc_handle);
            std::cerr << "\ncuMulticastAddDevice failed for selected GPU index " << i
                      << ": " << cu_result_name(status)
                      << " - " << cu_result_desc(status) << std::endl;
            std::cerr << "Result: NOT usable for multimem on this selected team." << std::endl;
            return 1;
        }
    }

    cuMemRelease(mc_handle);

    std::cout << "\nMulticast minimum granularity: " << min_granularity << " bytes" << std::endl;
    std::cout << "Result: usable for CUDA multicast/multimem object creation on this team."
              << std::endl;
    return 0;
#endif
}
