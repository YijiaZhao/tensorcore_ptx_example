// nvfp4_allreduce.cuh
// -----------------------------------------------------------------------------
// Self-contained NVFP4-compressed all-reduce (two-shot: reduce-scatter + all-gather).
// Extracted from bench_ar.cu — the config that beats TRT-LLM's BF16 custom AR
// (same two-shot algorithm) by 1.1-4.1x on an 8-GPU PCIe box, by moving 3.56x
// fewer bytes on the wire (NVFP4 vs BF16).
//
// Algorithm (TRT-LLM twoShotAllReduceKernel shape, NVFP4 payload):
//   0. each rank quantizes its BF16 input -> NVFP4 into its peer-visible payload
//   1. barrier
//   2. reduce-scatter: each rank owns shard s=[rank*shard, (rank+1)*shard); it reads
//      EVERY peer's NVFP4 payload over ITS shard only, dequant->FP32 accumulate,
//      then RE-QUANTIZES the reduced shard to NVFP4 ONCE into a separate reduced buffer
//   3. barrier
//   4. all-gather: each rank pulls every shard's reduced NVFP4 result from its owner,
//      dequant -> BF16 output.
//   Traffic is O(N) (each rank moves ~2(N-1)/N), not O(N^2).
//
// Wire format: NVFP4 = E2M1 packed (numel/2 B) + per-16 UE4M3 block scale (numel/16 B)
//              + per-tensor FP32 global scale. ~0.5625 B/elt = 3.56x smaller than BF16.
//
// Accuracy: rel_rmse ~0.14 (two quantization passes: input + reduced sum).
// Portability: E2M1 encoded manually (no cvt.e2m1x2 PTX) -> runs on sm_100 AND sm_120.
//
// IPC / peer-pointer contract:
//   The caller is responsible for making each rank's `payload` and `reduced` buffers
//   peer-accessible (single-process cudaDeviceEnablePeerAccess + raw pointers, OR
//   multi-process cudaIpcGetMemHandle/OpenMemHandle exchange). `peer_payloads[i]` /
//   `peer_reduced[i]` must be device arrays (on the calling rank's device) of `world`
//   pointers to every rank's payload / reduced buffer. `barrier_sig[i]` likewise points
//   to every rank's uint64 barrier buffer (MAX_RANKS words, zero-initialized once).
//
// Build target: nvcc -arch=sm_120a (or sm_100a) -O3 -std=c++17
// -----------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace nvfp4_ar {

static constexpr int   SF_VEC_SIZE     = 16;   // values per UE4M3 block scale
static constexpr int   ELTS_PER_THREAD = 8;    // 8 e2m1 = one uint32 per thread
static constexpr int   MAX_RANKS       = 8;
static constexpr float E2M1_MAX        = 6.0f;

// NVFP4 payload layout: [ packed e2m1 | block scales | float global scale ], 16B-aligned.
struct Layout {
    size_t numel, packed_bytes, scale_bytes, total_bytes;
    __host__ __device__ static Layout make(size_t numel) {
        Layout L; L.numel = numel; L.packed_bytes = numel / 2; L.scale_bytes = numel / SF_VEC_SIZE;
        size_t raw = L.packed_bytes + L.scale_bytes + sizeof(float);
        L.total_bytes = (raw + 15) & ~size_t(15); return L;
    }
    __host__ __device__ uint8_t* packed(uint8_t* b) const { return b; }
    __host__ __device__ uint8_t* scales(uint8_t* b) const { return b + packed_bytes; }
    __host__ __device__ float*   gscale(uint8_t* b) const { return reinterpret_cast<float*>(b + packed_bytes + scale_bytes); }
};

// ---- device helpers ----
__device__ __forceinline__ float recip_ftz(float a){ float b; asm volatile("rcp.approx.ftz.f32 %0,%1;":"=f"(b):"f"(a)); return b; }
__device__ __forceinline__ float e2m1_to_float(uint8_t n){ const float lut[8]={0,.5f,1,1.5f,2,3,4,6}; float m=lut[n&7]; return (n&8)?-m:m; }
__device__ __forceinline__ uint8_t float_to_e2m1(float v){ uint8_t s=(v<0)?8:0; float a=fabsf(v); uint8_t c;
    if(a<.25f)c=0; else if(a<.75f)c=1; else if(a<1.25f)c=2; else if(a<1.75f)c=3; else if(a<2.5f)c=4; else if(a<3.5f)c=5; else if(a<5.f)c=6; else c=7; return s|c; }
__device__ __forceinline__ float ue4m3_to_float(uint8_t code){ __nv_fp8_e4m3 f; f.__x=code; return (float)f; }

// ---- kernels ----

// Quantize BF16 input -> NVFP4 payload. `SFScaleVal` = per-tensor global scale
// (e.g. (E2M1_MAX*448)/amax). Launch grid = ceil(numel/ELTS_PER_THREAD / TPB), TPB multiple of 32.
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ in, uint8_t* __restrict__ ob, Layout L, float SF) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= L.numel / ELTS_PER_THREAD) return;
    const size_t base = tid * ELTS_PER_THREAD; float2 fp2[4]; float mx = 0;
#pragma unroll
    for (int i=0;i<4;i++){ __nv_bfloat162 v=*reinterpret_cast<const __nv_bfloat162*>(&in[base+i*2]); fp2[i]=__bfloat1622float2(v); mx=fmaxf(mx,fmaxf(fabsf(fp2[i].x),fabsf(fp2[i].y))); }
    mx = fmaxf(__shfl_xor_sync(0xffffffffu, mx, 1), mx);
    float SFv = SF*(mx*recip_ftz(E2M1_MAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os = (mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0; uint32_t e=0;
#pragma unroll
    for (int i=0;i<4;i++){ e|=uint32_t(float_to_e2m1(fp2[i].x*os))<<((2*i)*4); e|=uint32_t(float_to_e2m1(fp2[i].y*os))<<((2*i+1)*4); }
    *reinterpret_cast<uint32_t*>(&L.packed(ob)[base/2]) = e;
    if ((tid&1)==0) L.scales(ob)[base/SF_VEC_SIZE] = sf.__x;
    if (tid==0) *L.gscale(ob) = SF;
}

// Reduce-scatter: reduce this rank's shard across all peers, requantize once into `my`
// (a SEPARATE reduced buffer — never overwrite the input payload in place).
// `SFScaleReduced` should be SFScaleVal/world (the sum spans ~world x the input amax).
__global__ void reducescatter_kernel(uint8_t** peer, uint8_t* my, Layout L, int world, int rank, size_t shard, float SFr) {
    const size_t lt = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (lt >= shard/ELTS_PER_THREAD) return;
    const size_t g = size_t(rank)*shard + lt*ELTS_PER_THREAD, po = g/2, si = g/SF_VEC_SIZE; float acc[8];
#pragma unroll
    for (int i=0;i<8;i++) acc[i]=0;
    for (int r=0;r<world;r++){ uint8_t* b=peer[r]; uint32_t e=*reinterpret_cast<const uint32_t*>(&L.packed(b)[po]);
        float bs=ue4m3_to_float(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
        for (int i=0;i<8;i++) acc[i]+=e2m1_to_float((e>>(i*4))&0xF)*bs; }
    float mx=0;
#pragma unroll
    for (int i=0;i<8;i++) mx=fmaxf(mx,fabsf(acc[i]));
    mx = fmaxf(__shfl_xor_sync(0xffffffffu, mx, 1), mx);
    float SFv=SFr*(mx*recip_ftz(E2M1_MAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SFr)):0; uint32_t e=0;
#pragma unroll
    for (int i=0;i<8;i++) e|=uint32_t(float_to_e2m1(acc[i]*os))<<(i*4);
    *reinterpret_cast<uint32_t*>(&L.packed(my)[po]) = e;
    if ((lt&1)==0) L.scales(my)[si] = sf.__x;
    if (lt==0) *L.gscale(my) = SFr;
}

// All-gather: pull each shard's reduced NVFP4 result from its owner -> BF16 output.
__global__ void allgather_kernel(uint8_t** peer_reduced, __nv_bfloat16* __restrict__ out, Layout L, int world, size_t shard) {
    const size_t tid = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (tid >= L.numel/ELTS_PER_THREAD) return;
    const size_t g = tid*ELTS_PER_THREAD; const int owner = int(g/shard); uint8_t* b = peer_reduced[owner];
    const size_t po=g/2, si=g/SF_VEC_SIZE; uint32_t e=*reinterpret_cast<const uint32_t*>(&L.packed(b)[po]);
    float bs=ue4m3_to_float(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
    for (int i=0;i<8;i++) out[g+i]=__float2bfloat16(e2m1_to_float((e>>(i*4))&0xF)*bs);
}

// All-to-all counter barrier (TRT-LLM multi_gpu_barrier pattern). Use a MONOTONIC
// increasing `flag` per call so stale values never false-pass.
__global__ void barrier_kernel(uint64_t** sig, int rank, int world, uint64_t flag) {
    int t = threadIdx.x;
    if (blockIdx.x==0 && t<world) {
        asm volatile("st.global.release.sys.b64 [%1], %0;" ::"l"(flag), "l"(&sig[t][rank]));
        uint64_t f; do { asm volatile("ld.global.acquire.sys.b64 %0,[%1];" : "=l"(f) : "l"(&sig[rank][t])); } while (f != flag);
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Host launcher for ONE rank's two-shot NVFP4 all-reduce (call once per rank/device;
// the caller sets the device and drives the cross-rank barrier ordering via `flag`).
// Precondition: caller already cudaSetDevice(rank).
//   input           : BF16 [numel] on this device
//   output          : BF16 [numel] on this device
//   payload         : this rank's NVFP4 payload buffer (Layout::total_bytes), peer-visible
//   reduced         : this rank's reduced-shard NVFP4 buffer (Layout::total_bytes), peer-visible
//   peer_payloads   : device array[world] of every rank's payload pointer
//   peer_reduced    : device array[world] of every rank's reduced pointer
//   barrier_sig     : device array[world] of every rank's uint64 barrier buffer
//   SFScaleVal      : per-tensor global scale ((E2M1_MAX*448)/amax)
//   flag            : monotonic barrier flag base; this call consumes flag and flag+1
//   stream          : CUDA stream
// numel must satisfy numel % (world*SF_VEC_SIZE) == 0.
// NOTE: correct cross-rank execution requires all `world` ranks to be launched before
// any barrier completes (launch all, then sync) — same as TRT-LLM's model.
// -----------------------------------------------------------------------------
inline void launch_twoshot_rank(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint8_t* payload, uint8_t* reduced,
    uint8_t** peer_payloads, uint8_t** peer_reduced, uint64_t** barrier_sig,
    int world, int rank, size_t numel, float SFScaleVal, uint64_t flag,
    cudaStream_t stream = 0, int TPB = 256)
{
    Layout L = Layout::make(numel);
    const size_t shard = numel / world;
    const float  SFr   = SFScaleVal / float(world);
    const int grid   = int((numel/ELTS_PER_THREAD + TPB - 1) / TPB);
    const int grid_s = int((shard/ELTS_PER_THREAD + TPB - 1) / TPB);

    quantize_kernel     <<<grid,   TPB, 0, stream>>>(input, payload, L, SFScaleVal);
    barrier_kernel      <<<1, MAX_RANKS, 0, stream>>>(barrier_sig, rank, world, flag);
    reducescatter_kernel<<<grid_s, TPB, 0, stream>>>(peer_payloads, reduced, L, world, rank, shard, SFr);
    barrier_kernel      <<<1, MAX_RANKS, 0, stream>>>(barrier_sig, rank, world, flag + 1);
    allgather_kernel    <<<grid,   TPB, 0, stream>>>(peer_reduced, output, L, world, shard);
}

} // namespace nvfp4_ar
