// nvfp4_allreduce.cuh
// =============================================================================
// NVFP4-compressed all-reduce for Blackwell, following TRT-LLM's custom all-reduce.
//
// WHAT IT DOES
//   Every rank holds a BF16 tensor of `numel` elements. After the call every rank holds
//   the element-wise sum over all ranks, in BF16. The bytes that cross NVLink / PCIe are
//   NVFP4 (E2M1 + one UE4M3 block scale per 16 values = 0.5625 B/elt) instead of BF16
//   (2 B/elt) — 3.56x fewer bytes on the wire.
//
// ALGORITHM (identical to TRT-LLM customAllReduceKernels.cu, only the payload differs)
//   two-shot (launch_fused_rank, recommended):
//     1. quantize my copy of every rank's shard, PUSH it into that rank's comm buffer
//     2. block_barrier                       (all pushes landed everywhere)
//     3. reduce my own shard locally over the RANKS columns, requantize once
//     4. block_barrier                       (every reduced shard is ready)
//     5. PULL every rank's reduced shard, dequantize to BF16 output
//   one-shot (launch_oneshot_fused_rank):
//     1. quantize my whole tensor once, PUSH it into every rank's comm buffer
//     2. block_barrier
//     3. reduce all RANKS columns locally, dequantize to BF16 output
//   split (launch_twoshot_rank): the two-shot algorithm as five separate launches
//     (quantize | barrier | reduce-scatter | barrier | all-gather) — for per-phase timing.
//
// ACCESS PATTERN (this is what makes it fast; copied from TRT-LLM, see README)
//   * one thread owns one CHUNK of 32 elements = 16 B packed E2M1 + 2 B block scales
//   * every peer read/write is one 16-byte load/store (+ one 2-byte scale load/store)
//   * peer order is rotated by rank, (local_rank + i) % RANKS, so at any instant the
//     RANKS ranks talk to RANKS different peers instead of all hitting rank 0
//   * PUSH (write remote, read local): remote stores are posted and do not stall the SM;
//     remote loads would each pay a full interconnect round trip
//   * barriers are in-kernel flag exchanges (st.release / ld.acquire), not extra launches
//
// COMM BUFFER LAYOUT (per rank, peer-visible)
//   [parity 0: column 0 .. column RANKS-1][parity 1: column 0 .. column RANKS-1]
//   column     = one NVFP4 shard (two-shot) or one whole tensor (one-shot):
//                [packed E2M1: elts/2 bytes][block scales: elts/16 bytes], padded to 16 B
//   column c   = data PUSHED by source rank c. After the local reduce, column 0 holds the
//                reduced result (two-shot only).
//   parity     = flag & 1. Consecutive calls alternate parity, so call N+1 can start
//                pushing while a slow rank is still reading call N.
//
// ACCURACY   rel_rmse ~0.14 vs FP32 (two quantization passes: inputs, then the reduced sum).
//
// CODEC      E2M1 encode: hardware cvt.rn.satfinite.e2m1x2.f32 on every Blackwell.
//            E2M1 decode: hardware cvt.rn.f16x2.e2m1x2 on sm_100/sm_103, PRMT register
//            table on sm_120 (measured faster there). These are arch-specific ("a")
//            instructions: build with an EXPLICIT
//                -gencode arch=compute_120a,code=sm_120a      (or 100a / 103a)
//            The -arch=sm_120a shorthand does NOT enable them in nvcc 13.x.
//
// CALLER CONTRACT
//   * cudaSetDevice(rank) before each launch; launch ALL ranks before synchronizing any
//     (the in-kernel barriers need every rank in flight)
//   * numel % (RANKS * 32) == 0
//   * comm buffers and barrier words must be peer-visible (cudaDeviceEnablePeerAccess in
//     one process, or IPC / fabric handles across processes) and zero-initialised once
//   * `peer_*` arguments are DEVICE arrays of RANKS pointers, one per rank, on the
//     calling rank's device
//   * `flag` is a monotonically increasing counter, +1 per call, shared by all ranks
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace nvfp4_ar {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
static constexpr int   SF_VEC_SIZE     = 16;    // elements per UE4M3 block scale
static constexpr int   ELTS_PER_THREAD = 8;     // quantize_kernel: 8 E2M1 = one uint32 per thread
static constexpr int   EPT32           = 32;    // chunk: 32 elements per thread everywhere else
static constexpr int   CHUNK_ELTS      = EPT32;
static constexpr int   MAX_BLOCKS      = 2048;  // fused-kernel grid cap; barrier words scale with it
static constexpr int   MAX_RANKS       = 8;
static constexpr float E2M1_MAX        = 6.0f;  // largest E2M1 magnitude

// -----------------------------------------------------------------------------
// Standalone NVFP4 tensor layout (split path): [packed E2M1 | block scales | fp32 global scale]
// -----------------------------------------------------------------------------
struct Layout {
    size_t numel;
    size_t packed_bytes;   // numel / 2
    size_t scale_bytes;    // numel / 16
    size_t total_bytes;    // padded to 16 B

    __host__ __device__ static Layout make(size_t numel) {
        Layout L;
        L.numel        = numel;
        L.packed_bytes = numel / 2;
        L.scale_bytes  = numel / SF_VEC_SIZE;
        const size_t raw = L.packed_bytes + L.scale_bytes + sizeof(float);
        L.total_bytes  = (raw + 15) & ~size_t(15);
        return L;
    }
    __host__ __device__ uint8_t* packed(uint8_t* base) const { return base; }
    __host__ __device__ uint8_t* scales(uint8_t* base) const { return base + packed_bytes; }
    __host__ __device__ float*   gscale(uint8_t* base) const {
        return reinterpret_cast<float*>(base + packed_bytes + scale_bytes);
    }
};

// -----------------------------------------------------------------------------
// Scalar helpers
// -----------------------------------------------------------------------------
__device__ __forceinline__ float recip_ftz(float a) {
    float r;
    asm volatile("rcp.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
    return r;
}

// Reference (slow) scalar E2M1 codec. Not used on the hot path; kept for tests.
__device__ __forceinline__ float e2m1_to_float(uint8_t nibble) {
    const float magnitude[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float m = magnitude[nibble & 7];
    return (nibble & 8) ? -m : m;
}
__device__ __forceinline__ uint8_t float_to_e2m1(float v) {
    const uint8_t sign = (v < 0.f) ? 8 : 0;
    const float a = fabsf(v);
    uint8_t code;
    if      (a < 0.25f) code = 0;
    else if (a < 0.75f) code = 1;
    else if (a < 1.25f) code = 2;
    else if (a < 1.75f) code = 3;
    else if (a < 2.5f)  code = 4;
    else if (a < 3.5f)  code = 5;
    else if (a < 5.f)   code = 6;
    else                code = 7;
    return sign | code;
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t code) {
    __nv_fp8_e4m3 f;
    f.__x = code;
    return float(f);
}

// -----------------------------------------------------------------------------
// E2M1 codec: 8 values <-> one uint32 (nibble i = element i)
// -----------------------------------------------------------------------------

// decode8_e2m1: 8 E2M1 nibbles -> 8 floats.
// Both hardware cvt.rn.f16x2.e2m1x2 and the PRMT register-LUT below work on
// sm_100 and sm_120; this #if selects by measured end-to-end all-reduce performance,
// not instruction availability. On B200 (sm_100) hardware cvt vs PRMT was nearly
// tied (3862 -> 3846 us one-shot; 2671 -> 2672 us two-shot). On RTX 6000D (sm_120),
// hardware cvt made one-shot 37% and two-shot 13% slower than PRMT. See README §4.7.
// To compare the alternative on either architecture, change this compile-time branch
// and rebuild for that device; there is no runtime dispatch.
__device__ __forceinline__ void decode8_e2m1(uint32_t nibbles, float out[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1200)
    // One byte (2 nibbles) -> one f16x2; low nibble lands in .x.
    uint32_t h0, h1, h2, h3;
    asm("{\n"
        ".reg .b8 b0,b1,b2,b3;\n"
        "mov.b32 {b0,b1,b2,b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}"
        : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3)
        : "r"(nibbles));
    float2 f;
    f = __half22float2(*reinterpret_cast<__half2*>(&h0)); out[0] = f.x; out[1] = f.y;
    f = __half22float2(*reinterpret_cast<__half2*>(&h1)); out[2] = f.x; out[3] = f.y;
    f = __half22float2(*reinterpret_cast<__half2*>(&h2)); out[4] = f.x; out[5] = f.y;
    f = __half22float2(*reinterpret_cast<__half2*>(&h3)); out[6] = f.x; out[7] = f.y;
#else
    // PRMT register table. The 8 E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} are stored as their
    // E4M3 byte encodings in two 32-bit constants; __byte_perm picks 4 of them per call,
    // the sign bit is OR-ed back in, and the E4M3 bytes are converted with the FP8 unit.
    const uint32_t kMagLo = 0x3C383000u;   // E4M3 bytes for {0, .5, 1, 1.5}
    const uint32_t kMagHi = 0x4C484440u;   // E4M3 bytes for {2, 3, 4, 6}
    // Selector for even elements (nibbles 0,2,4,6) and odd elements (1,3,5,7).
    const uint32_t sel_even = (nibbles & 0x7u) | ((nibbles >> 4) & 0x70u) | ((nibbles >> 8) & 0x700u) | ((nibbles >> 12) & 0x7000u);
    const uint32_t sel_odd  = ((nibbles >> 4) & 0x7u) | ((nibbles >> 8) & 0x70u) | ((nibbles >> 12) & 0x700u) | ((nibbles >> 16) & 0x7000u);
    const uint32_t e4m3_even = __byte_perm(kMagLo, kMagHi, sel_even) | ((nibbles << 4) & 0x80808080u);
    const uint32_t e4m3_odd  = __byte_perm(kMagLo, kMagHi, sel_odd)  | ( nibbles       & 0x80808080u);
    float2 f;
    f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(e4m3_even & 0xFFFFu), __NV_E4M3))); out[0] = f.x; out[2] = f.y;
    f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(e4m3_even >> 16),     __NV_E4M3))); out[4] = f.x; out[6] = f.y;
    f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(e4m3_odd  & 0xFFFFu), __NV_E4M3))); out[1] = f.x; out[3] = f.y;
    f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(e4m3_odd  >> 16),     __NV_E4M3))); out[5] = f.x; out[7] = f.y;
#endif
}

// encode8_e2m1: 8 floats already scaled into E2M1 range -> uint32 of 8 nibbles.
__device__ __forceinline__ uint32_t encode8_e2m1(const float v[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Hardware cvt with saturation. Operand order (hi, lo) matches TRT-LLM's fp32_vec_to_e2m1.
    uint32_t nibbles;
    asm("{\n"
        ".reg .b8 b0,b1,b2,b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0,%2,%1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1,%4,%3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2,%6,%5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3,%8,%7;\n"
        "mov.b32 %0,{b0,b1,b2,b3};\n"
        "}"
        : "=r"(nibbles)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return nibbles;
#else
    // Compare-and-sum fallback (round-to-nearest thresholds between adjacent E2M1 values).
    uint32_t nibbles = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float a = fabsf(v[i]);
        uint32_t code = (uint32_t)(a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f)
                      + (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f);
        if (v[i] < 0.f) code |= 8u;
        nibbles |= code << (i * 4);
    }
    return nibbles;
#endif
}

// -----------------------------------------------------------------------------
// 32-element chunk helpers. One thread = one chunk = 16 B packed + 2 B block scales.
// -----------------------------------------------------------------------------
union Packed16 {                 // 16 bytes = 8 BF16
    int4           packed;
    __nv_bfloat162 unpacked[4];
};

struct Chunk32 {                 // NVFP4 wire form of 32 elements
    uint4    packed;             // 32 E2M1 nibbles
    uint16_t scales;             // two UE4M3 block scales (low byte = elements 0-15)
};

__device__ __forceinline__ void zero32(float acc[32]) {
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
}

// 32 BF16 (four 16-byte loads) -> 32 floats
__device__ __forceinline__ void load_bf16x32(const __nv_bfloat16* src, float v[32]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        Packed16 p;
        p.packed = *reinterpret_cast<const int4*>(src + k * 8);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 f = __bfloat1622float2(p.unpacked[j]);
            v[k * 8 + 2 * j]     = f.x;
            v[k * 8 + 2 * j + 1] = f.y;
        }
    }
}

// 32 floats -> 32 BF16 (four 16-byte stores)
__device__ __forceinline__ void store_bf16x32(__nv_bfloat16* dst, const float v[32]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        Packed16 p;
#pragma unroll
        for (int j = 0; j < 4; j++)
            p.unpacked[j] = __floats2bfloat162_rn(v[k * 8 + 2 * j], v[k * 8 + 2 * j + 1]);
        *reinterpret_cast<int4*>(dst + k * 8) = p.packed;
    }
}

// q16: quantize 16 floats -> 2 x uint32 E2M1 nibbles + one UE4M3 block scale.
//   block scale = SF * amax / 6, stored as UE4M3; values are divided by the STORED (rounded)
//   scale so encode and decode agree bit-for-bit.
__device__ __forceinline__ void q16(const float* v, float SF, uint32_t* out_nibbles, uint8_t& out_scale) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));

    const float          scale_f  = SF * (amax * recip_ftz(E2M1_MAX));
    const __nv_fp8_e4m3  scale_q  = __nv_fp8_e4m3(scale_f);
    const float          scale_qf = float(scale_q);
    const float          inv      = (amax != 0.f) ? recip_ftz(scale_qf * recip_ftz(SF)) : 0.f;
    out_scale = scale_q.__x;

#pragma unroll
    for (int half = 0; half < 2; half++) {
        float scaled[8];
#pragma unroll
        for (int i = 0; i < 8; i++) scaled[i] = v[half * 8 + i] * inv;
        out_nibbles[half] = encode8_e2m1(scaled);
    }
}

// quantize 32 floats -> one Chunk32
__device__ __forceinline__ Chunk32 quantize_chunk32(const float v[32], float SF) {
    uint32_t nibbles[4];
    uint8_t  scale[2];
    q16(v,      SF, nibbles,     scale[0]);
    q16(v + 16, SF, nibbles + 2, scale[1]);
    Chunk32 c;
    c.packed = make_uint4(nibbles[0], nibbles[1], nibbles[2], nibbles[3]);
    c.scales = uint16_t(scale[0]) | (uint16_t(scale[1]) << 8);
    return c;
}

// dq32_acc: dequantize 32 E2M1 with their 2 block scales and ADD into acc[32].
__device__ __forceinline__ void dq32_acc(uint4 packed, uint16_t scales, float invSF, float* acc) {
    const float bs0 = ue4m3_to_float(uint8_t(scales & 0xFF)) * invSF;   // elements  0-15
    const float bs1 = ue4m3_to_float(uint8_t(scales >> 8))   * invSF;   // elements 16-31
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int q = 0; q < 4; q++) {
        float dq[8];
        decode8_e2m1(words[q], dq);
        const float bs = (q < 2) ? bs0 : bs1;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[q * 8 + i] += dq[i] * bs;
    }
}
__device__ __forceinline__ void dequantize_accumulate(const Chunk32& c, float invSF, float acc[32]) {
    dq32_acc(c.packed, c.scales, invSF, acc);
}

// A Column is one NVFP4 tensor inside a comm buffer: [packed | scales].
// `elt` is an element offset (multiple of 32) inside the column.
struct Column {
    uint8_t* base;
    size_t   packed_bytes;

    __device__ __forceinline__ Chunk32 load(size_t elt) const {
        Chunk32 c;
        c.packed = *reinterpret_cast<const uint4*>(base + elt / 2);
        c.scales = *reinterpret_cast<const uint16_t*>(base + packed_bytes + elt / SF_VEC_SIZE);
        return c;
    }
    __device__ __forceinline__ void store(size_t elt, const Chunk32& c) const {
        *reinterpret_cast<uint4*>(base + elt / 2) = c.packed;
        *reinterpret_cast<uint16_t*>(base + packed_bytes + elt / SF_VEC_SIZE) = c.scales;
    }
};

// -----------------------------------------------------------------------------
// Comm-buffer geometry for the fused kernels
// -----------------------------------------------------------------------------
// bytes of one column holding `elts` elements, padded to 16 B
__host__ __device__ inline size_t column_bytes(size_t elts) {
    return ((elts / 2 + elts / SF_VEC_SIZE) + 15) & ~size_t(15);
}
// two-shot: column = one shard
__host__ __device__ inline size_t fused_slot_bytes(size_t shard)            { return column_bytes(shard); }
__host__ __device__ inline size_t fused_comm_bytes(size_t shard, int world) { return size_t(2) * world * fused_slot_bytes(shard); }
// one-shot: column = whole tensor
__host__ __device__ inline size_t oneshot_slot_bytes(size_t numel)            { return column_bytes(numel); }
__host__ __device__ inline size_t oneshot_comm_bytes(size_t numel, int world) { return size_t(2) * world * oneshot_slot_bytes(numel); }

// column `col` of the parity set selected by `flag`, inside comm buffer `buf`
__device__ __forceinline__ Column column_of(uint8_t* buf, uint32_t flag, int ranks, int col, size_t slot_bytes, size_t packed_bytes) {
    const int parity_base = (flag & 1) ? ranks : 0;
    Column c;
    c.base         = buf + size_t(parity_base + col) * slot_bytes;
    c.packed_bytes = packed_bytes;
    return c;
}

// -----------------------------------------------------------------------------
// Barriers
// -----------------------------------------------------------------------------

// Split path: all-to-all counter barrier as its own launch. `flag` must increase per call.
__global__ void barrier_kernel(uint64_t** sig, int rank, int world, uint64_t flag) {
    const int t = threadIdx.x;
    if (blockIdx.x == 0 && t < world) {
        // tell rank t that I (rank) arrived, then wait until rank t told me
        asm volatile("st.global.release.sys.b64 [%1], %0;" :: "l"(flag), "l"(&sig[t][rank]));
        uint64_t seen;
        do {
            asm volatile("ld.global.acquire.sys.b64 %0,[%1];" : "=l"(seen) : "l"(&sig[rank][t]));
        } while (seen != flag);
    }
    __syncthreads();
}

// Fused path: TRT-LLM block_barrier. Block b of every rank synchronises with block b of
// every other rank. Barrier words per rank: [parity][block 0..MAX_BLOCKS][rank], see
// barrier_word_offset(). Two arrays (in / out) so the two barriers of one call never alias.
__device__ __forceinline__ void st_flag_release(uint32_t flag, uint32_t* addr) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" :: "r"(flag), "l"(addr));
}
__device__ __forceinline__ uint32_t ld_flag_acquire(uint32_t* addr) {
    uint32_t f;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(f) : "l"(addr));
    return f;
}
// index of the first of `world` words used by block `block` under parity of `flag`
__device__ __forceinline__ uint32_t barrier_word_offset(int block, uint32_t flag, int world) {
    uint32_t off = uint32_t(block + 1) * world;
    if (flag & 1) off += uint32_t(MAX_BLOCKS + 1) * world;
    return off;
}
__host__ __device__ constexpr size_t fused_barrier_words(int world) {
    return size_t(2) * (MAX_BLOCKS + 1) * world + 8;
}

__device__ __forceinline__ void block_barrier(uint32_t** signals, uint32_t flag, int local_rank, int world, int tidx, int bidx) {
    // Every thread of this block must have ISSUED its data stores before one thread publishes
    // the flag: bar.sync makes them observed by the publishing thread, and st.release.sys is
    // cumulative, so the peer's ld.acquire sees the data once it sees the flag.
    // (TRT-LLM's block_barrier omits this __syncthreads(); on PCIe with many blocks that shows
    // up as run-to-run non-determinism — see README.)
    __syncthreads();
    if (tidx < world) {
        const uint32_t off = barrier_word_offset(bidx, flag, world);
        // thread t: tell rank t "my block bidx is done" ...
        st_flag_release(flag, signals[tidx] + off + local_rank);
        // ... then wait until rank t's block bidx said the same to me
        uint32_t* from_peer = signals[local_rank] + off + tidx;
        while (ld_flag_acquire(from_peer) != flag) {}
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Split-path kernels (five launches; see launch_twoshot_rank)
// -----------------------------------------------------------------------------

// BF16 input -> standalone NVFP4 payload (Layout). One thread = 8 elements; a pair of
// adjacent threads shares one 16-element block scale via shfl.
// grid = ceil(numel / 8 / TPB), TPB a multiple of 32.
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ in, uint8_t* __restrict__ out, Layout L, float SF) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= L.numel / ELTS_PER_THREAD) return;
    const size_t base = tid * ELTS_PER_THREAD;

    float2 v[4];
    float  amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&in[base + i * 2]));
        amax = fmaxf(amax, fmaxf(fabsf(v[i].x), fabsf(v[i].y)));
    }
    amax = fmaxf(__shfl_xor_sync(0xffffffffu, amax, 1), amax);   // block scale spans 2 threads

    const float         scale_f  = SF * (amax * recip_ftz(E2M1_MAX));
    const __nv_fp8_e4m3 scale_q  = __nv_fp8_e4m3(scale_f);
    const float         inv      = (amax != 0.f) ? recip_ftz(float(scale_q) * recip_ftz(SF)) : 0.f;

    float scaled[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        scaled[2 * i]     = v[i].x * inv;
        scaled[2 * i + 1] = v[i].y * inv;
    }
    *reinterpret_cast<uint32_t*>(&L.packed(out)[base / 2]) = encode8_e2m1(scaled);
    if ((tid & 1) == 0) L.scales(out)[base / SF_VEC_SIZE] = scale_q.__x;
    if (tid == 0)       *L.gscale(out) = SF;
}

// Reduce-scatter: sum MY shard over every rank's payload (PULL, rotated peer order),
// requantize once with SFr, write into `my_reduced`.
__global__ void reducescatter_kernel(uint8_t** peer_payload, uint8_t* my_reduced, Layout L, int world, int rank, size_t shard, float SFr) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= shard / CHUNK_ELTS) return;
    const size_t elt = size_t(rank) * shard + tid * CHUNK_ELTS;   // global element offset

    float acc[32];
    zero32(acc);
    for (int i = 0; i < world; i++) {
        const int src = (rank + i) % world;
        uint8_t* p = peer_payload[src];
        const Column col{L.packed(p), L.packed_bytes};
        dequantize_accumulate(col.load(elt), recip_ftz(*L.gscale(p)), acc);
    }

    const Column out{L.packed(my_reduced), L.packed_bytes};
    out.store(elt, quantize_chunk32(acc, SFr));
    if (tid == 0) *L.gscale(my_reduced) = SFr;
}

// All-gather: pull each shard's reduced NVFP4 from its owner (rotated order) -> BF16 output.
__global__ void allgather_kernel(uint8_t** peer_reduced, __nv_bfloat16* __restrict__ out, Layout L, int world, size_t shard, int rank) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= L.numel / CHUNK_ELTS) return;

    const size_t threads_per_shard = shard / CHUNK_ELTS;
    const int    step   = int(tid / threads_per_shard);            // which shard this thread copies
    const int    owner  = (step + rank) % world;                    // rotated: rank r starts at shard r
    const size_t elt    = size_t(owner) * shard + (tid - size_t(step) * threads_per_shard) * CHUNK_ELTS;

    uint8_t* p = peer_reduced[owner];
    const Column col{L.packed(p), L.packed_bytes};
    float v[32];
    zero32(v);
    dequantize_accumulate(col.load(elt), recip_ftz(*L.gscale(p)), v);
    store_bf16x32(out + elt, v);
}

// -----------------------------------------------------------------------------
// Fused two-shot kernel (TRT-LLM twoShotAllReduceKernel structure, NVFP4 payload)
// -----------------------------------------------------------------------------
template <int RANKS>
__global__ void twoshot_fused_kernel(
    const __nv_bfloat16* __restrict__ local_input, __nv_bfloat16* __restrict__ local_output,
    uint8_t** comm_bufs, uint32_t** barrier_in, uint32_t** barrier_out,
    int local_rank, size_t elts_per_rank, size_t slot_bytes, size_t elts_per_block,
    float SF, float SFr, uint32_t flag)
{
    const int    bidx         = blockIdx.x;
    const int    tidx         = threadIdx.x;
    const size_t packed_bytes = elts_per_rank / 2;
    const float  invSF        = recip_ftz(SF);
    const float  invSFr       = recip_ftz(SFr);

    // Rotated peer order: step i talks to rank (local_rank + i) % RANKS.
    int      rank_at_step[RANKS];
    uint8_t* comm_at_step[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++) {
        rank_at_step[i] = (local_rank + i) % RANKS;
        comm_at_step[i] = comm_bufs[rank_at_step[i]];
    }
    uint8_t* my_comm = comm_bufs[local_rank];

    // This block owns [chunk_start, chunk_end) of every shard; a thread strides by blockDim*32.
    const size_t chunk_start = size_t(bidx) * elts_per_block + size_t(tidx) * CHUNK_ELTS;
    const size_t chunk_end   = min(chunk_start + elts_per_block, elts_per_rank);
    const size_t stride      = size_t(blockDim.x) * CHUNK_ELTS;

    // ---- phase 1: quantize my copy of rank r's shard, PUSH into r's buffer at column local_rank
    for (size_t lo = chunk_start; lo < chunk_end; lo += stride) {
#pragma unroll
        for (int i = 0; i < RANKS; i++) {
            float v[32];
            load_bf16x32(local_input + size_t(rank_at_step[i]) * elts_per_rank + lo, v);
            const Column dst = column_of(comm_at_step[i], flag, RANKS, local_rank, slot_bytes, packed_bytes);
            dst.store(lo, quantize_chunk32(v, SF));
        }
    }

    block_barrier(barrier_in, flag, local_rank, RANKS, tidx, bidx);

    // ---- phase 2: my shard = sum of the RANKS columns of MY buffer (local reads),
    //               requantized once with SFr into column 0
    for (size_t lo = chunk_start; lo < chunk_end; lo += stride) {
        float acc[32];
        zero32(acc);
#pragma unroll
        for (int r = 0; r < RANKS; r++) {
            const int col = (r + RANKS - local_rank) % RANKS;   // rotated summation order
            const Column src = column_of(my_comm, flag, RANKS, col, slot_bytes, packed_bytes);
            dequantize_accumulate(src.load(lo), invSF, acc);
        }
        const Column reduced = column_of(my_comm, flag, RANKS, 0, slot_bytes, packed_bytes);
        reduced.store(lo, quantize_chunk32(acc, SFr));
    }

    block_barrier(barrier_out, flag, local_rank, RANKS, tidx, bidx);

    // ---- phase 3: PULL every rank's column 0 (its reduced shard), dequantize -> BF16 output
    for (size_t lo = chunk_start; lo < chunk_end; lo += stride) {
#pragma unroll
        for (int i = 0; i < RANKS; i++) {
            const Column src = column_of(comm_at_step[i], flag, RANKS, 0, slot_bytes, packed_bytes);
            float v[32];
            zero32(v);
            dequantize_accumulate(src.load(lo), invSFr, v);
            store_bf16x32(local_output + size_t(rank_at_step[i]) * elts_per_rank + lo, v);
        }
    }
}

// -----------------------------------------------------------------------------
// Fused one-shot kernel (TRT-LLM oneShotAllReduceKernel structure, NVFP4 payload)
// -----------------------------------------------------------------------------
template <int RANKS>
__global__ void oneshot_fused_kernel(
    const __nv_bfloat16* __restrict__ local_input, __nv_bfloat16* __restrict__ local_output,
    uint8_t** comm_bufs, uint32_t** barrier,
    int local_rank, size_t numel, size_t slot_bytes, size_t elts_per_block,
    float SF, uint32_t flag)
{
    const int    bidx         = blockIdx.x;
    const int    tidx         = threadIdx.x;
    const size_t packed_bytes = numel / 2;
    const float  invSF        = recip_ftz(SF);

    uint8_t* comm_at_step[RANKS];
#pragma unroll
    for (int i = 0; i < RANKS; i++) comm_at_step[i] = comm_bufs[(local_rank + i) % RANKS];
    uint8_t* my_comm = comm_bufs[local_rank];

    const size_t chunk_start = size_t(bidx) * elts_per_block + size_t(tidx) * CHUNK_ELTS;
    const size_t chunk_end   = min(chunk_start + elts_per_block, numel);
    const size_t stride      = size_t(blockDim.x) * CHUNK_ELTS;

    // ---- phase 1: quantize my chunk once, PUSH it into every rank's buffer at column local_rank
    for (size_t lo = chunk_start; lo < chunk_end; lo += stride) {
        float v[32];
        load_bf16x32(local_input + lo, v);
        const Chunk32 q = quantize_chunk32(v, SF);
#pragma unroll
        for (int i = 0; i < RANKS; i++) {
            const Column dst = column_of(comm_at_step[i], flag, RANKS, local_rank, slot_bytes, packed_bytes);
            dst.store(lo, q);
        }
    }

    block_barrier(barrier, flag, local_rank, RANKS, tidx, bidx);

    // ---- phase 2: sum the RANKS columns of MY buffer (local reads) -> BF16 output
    for (size_t lo = chunk_start; lo < chunk_end; lo += stride) {
        float acc[32];
        zero32(acc);
#pragma unroll
        for (int r = 0; r < RANKS; r++) {
            const int col = (r + RANKS - local_rank) % RANKS;
            const Column src = column_of(my_comm, flag, RANKS, col, slot_bytes, packed_bytes);
            dequantize_accumulate(src.load(lo), invSF, acc);
        }
        store_bf16x32(local_output + lo, acc);
    }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

// Default grid for NVLink (B200 sweep): one chunk per block up to 256 blocks, then 4 chunks
// per thread up to MAX_BLOCKS; never fewer than 16 blocks. On PCIe pass grid = 4..8 explicitly
// (measured 1.3-1.8x faster there: fewer concurrent writers, larger PCIe transactions).
inline int default_fused_grid(size_t elts, int TPB) {
    const size_t elts_per_pass = size_t(TPB) * CHUNK_ELTS;
    const size_t chunks        = (elts + elts_per_pass - 1) / elts_per_pass;
    size_t g = chunks < 256 ? chunks : 256;
    if (chunks / 4 > g) g = chunks / 4;
    if (g < 16) g = 16;
    if (g > size_t(MAX_BLOCKS)) g = MAX_BLOCKS;
    return int(g);
}
// elements each block covers, rounded up to whole passes of TPB*32
inline size_t elts_per_block_for(size_t elts, int grid, int TPB) {
    const size_t elts_per_pass = size_t(TPB) * CHUNK_ELTS;
    const size_t per_block     = size_t(grid) * elts_per_pass;
    return ((elts + per_block - 1) / per_block) * elts_per_pass;
}

// Two-shot, one kernel (recommended).
//   peer_comm      : device array[RANKS] of every rank's comm buffer, fused_comm_bytes(numel/RANKS, RANKS) bytes each
//   barrier_in/out : device arrays[RANKS] of every rank's barrier words, fused_barrier_words(RANKS) uint32 each
//   SFScaleVal     : per-tensor scale, e.g. (E2M1_MAX * 448) / amax
//   flag           : +1 per call, shared by all ranks
//   grid           : 0 = NVLink default; PCIe: 4..8
template <int RANKS>
inline void launch_fused_rank(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint8_t** peer_comm, uint32_t** barrier_in, uint32_t** barrier_out,
    int rank, size_t numel, float SFScaleVal, uint32_t flag,
    cudaStream_t stream = 0, int TPB = 256, int grid = 0)
{
    const size_t shard = numel / RANKS;
    const size_t slot  = fused_slot_bytes(shard);
    const float  SFr   = SFScaleVal / float(RANKS);      // the reduced sum is RANKS x larger
    if (grid <= 0) grid = default_fused_grid(shard, TPB);
    const size_t epb = elts_per_block_for(shard, grid, TPB);
    twoshot_fused_kernel<RANKS><<<grid, TPB, 0, stream>>>(
        input, output, peer_comm, barrier_in, barrier_out, rank, shard, slot, epb, SFScaleVal, SFr, flag);
}

// One-shot, one kernel.
//   peer_comm : oneshot_comm_bytes(numel, RANKS) bytes per rank; barrier: fused_barrier_words(RANKS) uint32 per rank
//   On NVLink one-shot and two-shot both sit on the latency floor below ~2M elements; on PCIe
//   use grid = 1 and prefer two-shot unless the message is tiny.
template <int RANKS>
inline void launch_oneshot_fused_rank(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint8_t** peer_comm, uint32_t** barrier,
    int rank, size_t numel, float SFScaleVal, uint32_t flag,
    cudaStream_t stream = 0, int TPB = 256, int grid = 0)
{
    const size_t slot = oneshot_slot_bytes(numel);
    if (grid <= 0) grid = default_fused_grid(numel, TPB);
    const size_t epb = elts_per_block_for(numel, grid, TPB);
    oneshot_fused_kernel<RANKS><<<grid, TPB, 0, stream>>>(
        input, output, peer_comm, barrier, rank, numel, slot, epb, SFScaleVal, flag);
}

// Split path, five launches (`flag` and `flag + 1` are consumed).
//   payload / reduced          : this rank's NVFP4 buffers, Layout::make(numel).total_bytes each, peer-visible
//   peer_payloads / peer_reduced / barrier_sig : device arrays[world] of every rank's pointers
inline void launch_twoshot_rank(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint8_t* payload, uint8_t* reduced,
    uint8_t** peer_payloads, uint8_t** peer_reduced, uint64_t** barrier_sig,
    int world, int rank, size_t numel, float SFScaleVal, uint64_t flag,
    cudaStream_t stream = 0, int TPB = 256)
{
    const Layout L      = Layout::make(numel);
    const size_t shard  = numel / world;
    const float  SFr    = SFScaleVal / float(world);
    const int grid_q    = int((numel / ELTS_PER_THREAD + TPB - 1) / TPB);
    const int grid_rs   = int((shard / CHUNK_ELTS   + TPB - 1) / TPB);
    const int grid_ag   = int((numel / CHUNK_ELTS   + TPB - 1) / TPB);

    quantize_kernel     <<<grid_q,  TPB,       0, stream>>>(input, payload, L, SFScaleVal);
    barrier_kernel      <<<1,       MAX_RANKS, 0, stream>>>(barrier_sig, rank, world, flag);
    reducescatter_kernel<<<grid_rs, TPB,       0, stream>>>(peer_payloads, reduced, L, world, rank, shard, SFr);
    barrier_kernel      <<<1,       MAX_RANKS, 0, stream>>>(barrier_sig, rank, world, flag + 1);
    allgather_kernel    <<<grid_ag, TPB,       0, stream>>>(peer_reduced, output, L, world, shard, rank);
}

} // namespace nvfp4_ar
