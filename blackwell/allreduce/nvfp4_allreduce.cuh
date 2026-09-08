// nvfp4_allreduce.cuh
// -----------------------------------------------------------------------------
// Self-contained NVFP4-compressed all-reduce (two-shot: reduce-scatter + all-gather).
// Extracted from bench_ar.cu. Same two-shot algorithm as TRT-LLM's custom AR; beats TRT-LLM's
// BF16 custom AR 1.7-5x and TRT-LLM's FP8 low-precision AR 1.2-2.1x (PCIe) by moving fewer
// bytes on the wire (NVFP4 0.5625 B/elt vs FP8 ~1.06 vs BF16 2.0). See README.md for tables.
//
// Two entry points, both following TRT-LLM's twoShotAllReduceKernel access pattern
// (rank-rotated peer order, 16-byte loads/stores, PUSH reduce-scatter):
//
//   launch_fused_rank()   ONE kernel (recommended): quantize+push -> block_barrier -> local
//                         reduce + requantize -> block_barrier -> pull + dequantize. This is
//                         TRT-LLM's fused two-shot with the payload swapped for NVFP4.
//   launch_twoshot_rank() five launches (quantize | barrier | reduce-scatter | barrier |
//                         all-gather) — same kernels split up, useful for per-phase timing.
//
// Both: each rank owns shard s=[rank*shard,(rank+1)*shard); the reduced shard is requantized
// to NVFP4 ONCE (scale SFScaleVal/world) before the all-gather. Traffic is O(N) per rank.
// Every thread moves 32 elements = 16 B packed E2M1 + 2 B block scales per peer; without that
// (4 B/thread) the all-gather was request-bound at ~200 GB/s on NVLink, and without the rank
// rotation all ranks read owner 0 first (7 readers on one GPU's NVLink egress) — see README.
//
// Wire format: NVFP4 = E2M1 packed (numel/2 B) + per-16 UE4M3 block scale (numel/16 B)
//              + per-tensor FP32 global scale. ~0.5625 B/elt = 3.56x smaller than BF16.
//
// Accuracy: rel_rmse ~0.14 (two quantization passes: input + reduced sum).
// Codec: hardware E2M1 cvt (cvt.rn.satfinite.e2m1x2.f32 / cvt.rn.f16x2.e2m1x2) on ALL Blackwell
//   (sm_100/101/103/120); PRMT-LUT / compare-sum fallback elsewhere. The E2M1 cvts are
//   arch-specific ("a") features: build with EXPLICIT -gencode arch=compute_XXXa,code=sm_XXXa.
//   The -arch=sm_XXXa shorthand does NOT enable them in nvcc 13.x (ptxas "not supported").
//
// IPC / peer-pointer contract:
//   The caller is responsible for making each rank's `payload`/`reduced` (5-launch) or
//   `comm` (fused) buffers and barrier words peer-accessible (single-process cudaDeviceEnablePeerAccess + raw pointers, OR
//   multi-process cudaIpcGetMemHandle/OpenMemHandle exchange). `peer_payloads[i]` /
//   `peer_reduced[i]` must be device arrays (on the calling rank's device) of `world`
//   pointers to every rank's payload / reduced buffer. `barrier_sig[i]` likewise points
//   to every rank's uint64 barrier buffer (MAX_RANKS words, zero-initialized once).
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17   (or 100a / 103a)
// -----------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace nvfp4_ar {

static constexpr int   SF_VEC_SIZE     = 16;   // values per UE4M3 block scale
static constexpr int   ELTS_PER_THREAD = 8;    // quantize kernel: 8 e2m1 = one uint32 per thread
static constexpr int   EPT32           = 32;   // RS/AG/fused: 32 e2m1 = one 16-byte load + 2 scale bytes
static constexpr int   MAX_BLOCKS      = 2048; // fused kernel grid cap (barrier words scale with it)
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

// ---- FAST CODEC (all Blackwell: sm_100/101/103/120). MUST build with explicit arch-specific
// gencode, e.g. -gencode arch=compute_120a,code=sm_120a — the -arch=sm_XXXa shorthand does NOT
// enable the E2M1 cvt instructions in nvcc 13.x. Non-Blackwell falls back to PRMT / compare-sum.
// decode8: 8 e2m1 nibbles (nibble i = elt i) -> 8 floats.
__device__ __forceinline__ void decode8_e2m1(uint32_t uq, float d[8]){
// Decode path is chosen per arch from measurement (see README): the hardware cvt.f16x2.e2m1x2 is
// ~1-2% faster on sm_100/sm_103 (full-rate conversion pipe) but 13-37% SLOWER than the PRMT
// register-LUT on sm_120 (consumer die, low-throughput e2m1 cvt pipe). So: hardware on sm_100/103,
// PRMT on sm_120 and everything else.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=1000) && (__CUDA_ARCH__<1200)
    uint32_t h0,h1,h2,h3;   // hardware cvt.rn.f16x2.e2m1x2: one byte (2 nibbles) -> f16x2, low nibble -> .x
    asm("{\n.reg .b8 b0,b1,b2,b3;\nmov.b32 {b0,b1,b2,b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        :"=r"(h0),"=r"(h1),"=r"(h2),"=r"(h3):"r"(uq));
    float2 f;
    f=__half22float2(*reinterpret_cast<__half2*>(&h0)); d[0]=f.x; d[1]=f.y;
    f=__half22float2(*reinterpret_cast<__half2*>(&h1)); d[2]=f.x; d[3]=f.y;
    f=__half22float2(*reinterpret_cast<__half2*>(&h2)); d[4]=f.x; d[5]=f.y;
    f=__half22float2(*reinterpret_cast<__half2*>(&h3)); d[6]=f.x; d[7]=f.y;
#else
    // PRMT register-LUT fallback: e2m1 mag 0..7 -> e4m3 bytes {0,.5,1,1.5 | 2,3,4,6}, then fp8 cvt
    const uint32_t kMagLo=0x3C383000u, kMagHi=0x4C484440u;
    uint32_t sel_lo=(uq&0x7u)|((uq>>4)&0x70u)|((uq>>8)&0x700u)|((uq>>12)&0x7000u);
    uint32_t sel_hi=((uq>>4)&0x7u)|((uq>>8)&0x70u)|((uq>>12)&0x700u)|((uq>>16)&0x7000u);
    uint32_t olo=__byte_perm(kMagLo,kMagHi,sel_lo)|((uq<<4)&0x80808080u);
    uint32_t ohi=__byte_perm(kMagLo,kMagHi,sel_hi)|( uq     &0x80808080u);
    float2 f;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(olo&0xFFFFu),__NV_E4M3))); d[0]=f.x; d[2]=f.y;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(olo>>16),    __NV_E4M3))); d[4]=f.x; d[6]=f.y;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(ohi&0xFFFFu),__NV_E4M3))); d[1]=f.x; d[3]=f.y;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(ohi>>16),    __NV_E4M3))); d[5]=f.x; d[7]=f.y;
#endif
}
// encode8: 8 floats (scaled into e2m1 range) -> uint32 of 8 e2m1 nibbles (nibble i = elt i).
__device__ __forceinline__ uint32_t encode8_e2m1(const float v[8]){
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=1000)
    uint32_t val;   // hardware cvt.rn.satfinite.e2m1x2.f32, operand order per TRT-LLM fp32_vec_to_e2m1
    asm("{\n.reg .b8 b0,b1,b2,b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0,%2,%1;\ncvt.rn.satfinite.e2m1x2.f32 b1,%4,%3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2,%6,%5;\ncvt.rn.satfinite.e2m1x2.f32 b3,%8,%7;\n"
        "mov.b32 %0,{b0,b1,b2,b3};\n}"
        :"=r"(val):"f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),"f"(v[4]),"f"(v[5]),"f"(v[6]),"f"(v[7]));
    return val;
#else
    uint32_t e=0;
#pragma unroll
    for(int i=0;i<8;i++){ float a=fabsf(v[i]);
        uint32_t c=(uint32_t)(a>=0.25f)+(a>=0.75f)+(a>=1.25f)+(a>=1.75f)+(a>=2.5f)+(a>=3.5f)+(a>=5.0f);
        c|=(v[i]<0.f)?8u:0u; e|=c<<(i*4); }
    return e;
#endif
}

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
    float os = (mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0;
    float sv[8];
#pragma unroll
    for (int i=0;i<4;i++){ sv[2*i]=fp2[i].x*os; sv[2*i+1]=fp2[i].y*os; }
    uint32_t e=encode8_e2m1(sv);
    *reinterpret_cast<uint32_t*>(&L.packed(ob)[base/2]) = e;
    if ((tid&1)==0) L.scales(ob)[base/SF_VEC_SIZE] = sf.__x;
    if (tid==0) *L.gscale(ob) = SF;
}

// ---- 16-byte helpers shared by the split and fused paths ----
union Packed16 { int4 packed; __nv_bfloat162 unpacked[4]; };
// quantize 16 floats -> 2 x uint32 e2m1 nibbles + one UE4M3 block scale (scale = SF*amax/6)
__device__ __forceinline__ void q16(const float* v, float SF, uint32_t* ow, uint8_t& sfb){
    float mx=0;
#pragma unroll
    for(int i=0;i<16;i++) mx=fmaxf(mx,fabsf(v[i]));
    float SFv=SF*(mx*recip_ftz(E2M1_MAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0; sfb=sf.__x;
#pragma unroll
    for(int q=0;q<2;q++){ float sv[8];
#pragma unroll
        for(int i=0;i<8;i++) sv[i]=v[q*8+i]*os;
        ow[q]=encode8_e2m1(sv); }
}
// dequantize 32 e2m1 (16 B) with 2 block scales, accumulate into acc[32]
__device__ __forceinline__ void dq32_acc(uint4 e, uint16_t s2, float invSF, float* acc){
    float bs0=ue4m3_to_float(uint8_t(s2&0xFF))*invSF, bs1=ue4m3_to_float(uint8_t(s2>>8))*invSF;
    uint32_t w[4]={e.x,e.y,e.z,e.w};
#pragma unroll
    for(int q=0;q<4;q++){ float dq[8]; decode8_e2m1(w[q],dq); float bs=(q<2)?bs0:bs1;
#pragma unroll
        for(int i=0;i<8;i++) acc[q*8+i]+=dq[i]*bs; }
}

// Reduce-scatter (split path): reduce this rank's shard across all peers, requantize once into `my`.
// Peer order is rotated by rank; each peer read is one 16-byte load + one 2-byte scale load.
__global__ void reducescatter_kernel(uint8_t** peer, uint8_t* my, Layout L, int world, int rank, size_t shard, float SFr) {
    const size_t lt = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (lt >= shard/EPT32) return;
    const size_t g = size_t(rank)*shard + lt*EPT32, po = g/2, si = g/SF_VEC_SIZE; float acc[32];
#pragma unroll
    for (int i=0;i<32;i++) acc[i]=0;
    for (int ii=0;ii<world;ii++){ int r=(rank+ii)%world; uint8_t* b=peer[r];
        uint4 e=*reinterpret_cast<const uint4*>(&L.packed(b)[po]);
        uint16_t s2=*reinterpret_cast<const uint16_t*>(&L.scales(b)[si]);
        dq32_acc(e,s2,recip_ftz(*L.gscale(b)),acc); }
    uint32_t ow[4]; uint8_t sfb[2];
    q16(acc,SFr,ow,sfb[0]); q16(acc+16,SFr,ow+2,sfb[1]);
    *reinterpret_cast<uint4*>(&L.packed(my)[po]) = make_uint4(ow[0],ow[1],ow[2],ow[3]);
    *reinterpret_cast<uint16_t*>(&L.scales(my)[si]) = uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
    if (lt==0) *L.gscale(my) = SFr;
}

// All-gather (split path): pull each shard's reduced NVFP4 result from its owner -> BF16 output.
// Owner sequence rotated by rank so concurrent readers spread over all owners.
__global__ void allgather_kernel(uint8_t** peer_reduced, __nv_bfloat16* __restrict__ out, Layout L, int world, size_t shard, int rank) {
    const size_t tid = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (tid >= L.numel/EPT32) return;
    const size_t tps=shard/EPT32; const int slot=int(tid/tps); const int owner=(slot+rank)%world;
    const size_t g=size_t(owner)*shard+(tid-size_t(slot)*tps)*EPT32; uint8_t* b=peer_reduced[owner];
    uint4 e=*reinterpret_cast<const uint4*>(&L.packed(b)[g/2]);
    uint16_t s2=*reinterpret_cast<const uint16_t*>(&L.scales(b)[g/SF_VEC_SIZE]);
    float acc[32];
#pragma unroll
    for(int i=0;i<32;i++) acc[i]=0.f;
    dq32_acc(e,s2,recip_ftz(*L.gscale(b)),acc);
#pragma unroll
    for(int k=0;k<4;k++){ Packed16 pk;
#pragma unroll
        for(int j=0;j<4;j++) pk.unpacked[j]=__floats2bfloat162_rn(acc[k*8+2*j],acc[k*8+2*j+1]);
        *reinterpret_cast<int4*>(&out[g+k*8])=pk.packed; }
}

// All-to-all counter barrier (split path). Use a MONOTONIC increasing `flag` per call.
__global__ void barrier_kernel(uint64_t** sig, int rank, int world, uint64_t flag) {
    int t = threadIdx.x;
    if (blockIdx.x==0 && t<world) {
        asm volatile("st.global.release.sys.b64 [%1], %0;" ::"l"(flag), "l"(&sig[t][rank]));
        uint64_t f; do { asm volatile("ld.global.acquire.sys.b64 %0,[%1];" : "=l"(f) : "l"(&sig[rank][t])); } while (f != flag);
    }
    __syncthreads();
}

// ---- fused path: TRT-LLM block_barrier (flag parity ping-pong, per-block offset) ----
__device__ __forceinline__ void st_flag_release(uint32_t flag, uint32_t* addr){ asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr)); }
__device__ __forceinline__ uint32_t ld_flag_acquire(uint32_t* addr){ uint32_t f; asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(f) : "l"(addr)); return f; }
__device__ __forceinline__ void block_barrier(uint32_t** signals, uint32_t flag, int local_rank, int world, int tidx, int bidx){
    // All threads of the block must have issued their data stores before one thread publishes the flag:
    // bar.sync makes them "observed" by the flag-storing thread and st.release.sys (cumulative) orders them at
    // system scope. TRT-LLM's block_barrier omits this __syncthreads(); with many blocks on PCIe that shows up
    // as run-to-run non-determinism (see README).
    __syncthreads();
    if (tidx < world) {
        uint32_t off = (bidx + 1) * world;
        if (flag % 2 == 1) off += (MAX_BLOCKS + 1) * world;
        st_flag_release(flag, signals[tidx] + off + local_rank);
        uint32_t* peer = signals[local_rank] + off + tidx;
        while (ld_flag_acquire(peer) != flag) {}
    }
    __syncthreads();
}
// barrier words per rank for the fused kernel (two arrays: in / out)
__host__ __device__ constexpr size_t fused_barrier_words(int world){ return size_t(2)*(MAX_BLOCKS+1)*world + 8; }
// comm buffer per rank for the fused kernel: [2*world columns][slot_bytes]
__host__ __device__ inline size_t fused_slot_bytes(size_t shard){ return ((shard/2 + shard/SF_VEC_SIZE) + 15) & ~size_t(15); }
__host__ __device__ inline size_t fused_comm_bytes(size_t shard, int world){ return size_t(2)*world*fused_slot_bytes(shard); }

// Fused two-shot, TRT-LLM twoShotAllReduceKernel structure with NVFP4 payload:
//  1. quantize my copy of shard ranks[ii] and PUSH it (16 B + 2 B) into owner's buffer, column local_rank
//  2. block_barrier
//  3. reduce my shard across the world columns of MY buffer (local reads), requantize into column 0
//  4. block_barrier
//  5. pull every owner's column 0, dequantize -> BF16 output
// Columns are [flag parity]*world + col, so consecutive calls never overwrite each other.
template <int RANKS>
__global__ void twoshot_fused_kernel(
    const __nv_bfloat16* __restrict__ local_input, __nv_bfloat16* __restrict__ local_output,
    uint8_t** comm_bufs, uint32_t** barrier_in, uint32_t** barrier_out,
    int local_rank, size_t elts_per_rank, size_t slot_bytes, size_t elts_per_block, float SF, float SFr, uint32_t flag)
{
    const int bidx=blockIdx.x, tidx=threadIdx.x;
    const int buffer_offset=(flag%2==0)?0:RANKS;
    const size_t packed_bytes=elts_per_rank/2;
    uint8_t* local_shared=comm_bufs[local_rank];
    uint8_t* buffers[RANKS]; int ranks[RANKS];
#pragma unroll
    for(int ii=0;ii<RANKS;ii++){ int rank=(local_rank+ii)%RANKS; ranks[ii]=rank; buffers[ii]=comm_bufs[rank]; }
    const size_t chunk_start=bidx*elts_per_block+tidx*EPT32;
    const size_t chunk_end=min(chunk_start+elts_per_block, elts_per_rank);
    const float invSF=recip_ftz(SF), invSFr=recip_ftz(SFr);

    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
#pragma unroll
        for(int ii=0;ii<RANKS;ii++){
            const __nv_bfloat16* src=local_input+(size_t)ranks[ii]*elts_per_rank+lo;
            float v[32];
#pragma unroll
            for(int k=0;k<4;k++){ Packed16 pk; pk.packed=*reinterpret_cast<const int4*>(src+k*8);
#pragma unroll
                for(int j=0;j<4;j++){ float2 f=__bfloat1622float2(pk.unpacked[j]); v[k*8+2*j]=f.x; v[k*8+2*j+1]=f.y; } }
            uint32_t ow[4]; uint8_t sfb[2];
            q16(v,SF,ow,sfb[0]); q16(v+16,SF,ow+2,sfb[1]);
            uint8_t* slot=buffers[ii]+(size_t)(buffer_offset+local_rank)*slot_bytes;
            *reinterpret_cast<uint4*>(slot+lo/2)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
            *reinterpret_cast<uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE)=uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
        }
    }
    block_barrier(barrier_in, flag, local_rank, RANKS, tidx, bidx);
    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
        float acc[32];
#pragma unroll
        for(int i=0;i<32;i++) acc[i]=0.f;
#pragma unroll
        for(int rank=0;rank<RANKS;rank++){ int ii=(rank+RANKS-local_rank)%RANKS;
            const uint8_t* slot=local_shared+(size_t)(buffer_offset+ii)*slot_bytes;
            uint4 e=*reinterpret_cast<const uint4*>(slot+lo/2);
            uint16_t s2=*reinterpret_cast<const uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE);
            dq32_acc(e,s2,invSF,acc); }
        uint32_t ow[4]; uint8_t sfb[2];
        q16(acc,SFr,ow,sfb[0]); q16(acc+16,SFr,ow+2,sfb[1]);
        uint8_t* slot=local_shared+(size_t)buffer_offset*slot_bytes;
        *reinterpret_cast<uint4*>(slot+lo/2)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
        *reinterpret_cast<uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE)=uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
    }
    block_barrier(barrier_out, flag, local_rank, RANKS, tidx, bidx);
    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
#pragma unroll
        for(int ii=0;ii<RANKS;ii++){
            const uint8_t* slot=buffers[ii]+(size_t)buffer_offset*slot_bytes;
            uint4 e=*reinterpret_cast<const uint4*>(slot+lo/2);
            uint16_t s2=*reinterpret_cast<const uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE);
            float acc[32];
#pragma unroll
            for(int i=0;i<32;i++) acc[i]=0.f;
            dq32_acc(e,s2,invSFr,acc);
            __nv_bfloat16* dst=local_output+(size_t)ranks[ii]*elts_per_rank+lo;
#pragma unroll
            for(int k=0;k<4;k++){ Packed16 pk;
#pragma unroll
                for(int j=0;j<4;j++) pk.unpacked[j]=__floats2bfloat162_rn(acc[k*8+2*j],acc[k*8+2*j+1]);
                *reinterpret_cast<int4*>(dst+k*8)=pk.packed; }
        }
    }
}

// ======================= NVFP4 fused one-shot — TRT-LLM oneShotAllReduceKernel structure =======================
// quantize my chunk once, PUSH it (16 B + 2 B) into every peer's buffer at column local_rank (rotated order),
// block_barrier, then reduce the RANKS columns of my own buffer locally -> BF16 output. Column = whole tensor.
template <int RANKS>
__global__ void oneshot_fused_kernel(
    const __nv_bfloat16* __restrict__ local_input, __nv_bfloat16* __restrict__ local_output,
    uint8_t** comm_bufs, uint32_t** barrier, int local_rank, size_t numel, size_t slot_bytes, size_t elts_per_block,
    float SF, uint32_t flag)
{
    const int bidx=blockIdx.x, tidx=threadIdx.x;
    const int buffer_offset=(flag%2==0)?0:RANKS;
    const size_t packed_bytes=numel/2;
    uint8_t* local_shared=comm_bufs[local_rank];
    uint8_t* buffers[RANKS];
#pragma unroll
    for(int ii=0;ii<RANKS;ii++) buffers[ii]=comm_bufs[(local_rank+ii)%RANKS];
    const size_t chunk_start=bidx*elts_per_block+tidx*EPT32;
    const size_t chunk_end=min(chunk_start+elts_per_block, numel);
    const float invSF=recip_ftz(SF);
    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
        float v[32];
#pragma unroll
        for(int k=0;k<4;k++){ Packed16 pk; pk.packed=*reinterpret_cast<const int4*>(local_input+lo+k*8);
#pragma unroll
            for(int j=0;j<4;j++){ float2 f=__bfloat1622float2(pk.unpacked[j]); v[k*8+2*j]=f.x; v[k*8+2*j+1]=f.y; } }
        uint32_t ow[4]; uint8_t sfb[2];
        q16(v,SF,ow,sfb[0]); q16(v+16,SF,ow+2,sfb[1]);
        uint4 pk4=make_uint4(ow[0],ow[1],ow[2],ow[3]); uint16_t sc=uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
#pragma unroll
        for(int ii=0;ii<RANKS;ii++){
            uint8_t* slot=buffers[ii]+(size_t)(buffer_offset+local_rank)*slot_bytes;
            *reinterpret_cast<uint4*>(slot+lo/2)=pk4;
            *reinterpret_cast<uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE)=sc;
        }
    }
    block_barrier(barrier, flag, local_rank, RANKS, tidx, bidx);
    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
        float acc[32];
#pragma unroll
        for(int i=0;i<32;i++) acc[i]=0.f;
#pragma unroll
        for(int rank=0;rank<RANKS;rank++){ int ii=(rank+RANKS-local_rank)%RANKS;
            const uint8_t* slot=local_shared+(size_t)(buffer_offset+ii)*slot_bytes;
            uint4 e=*reinterpret_cast<const uint4*>(slot+lo/2);
            uint16_t s2=*reinterpret_cast<const uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE);
            dq32_acc(e,s2,invSF,acc); }
        __nv_bfloat16* dst=local_output+lo;
#pragma unroll
        for(int k=0;k<4;k++){ Packed16 pk;
#pragma unroll
            for(int j=0;j<4;j++) pk.unpacked[j]=__floats2bfloat162_rn(acc[k*8+2*j],acc[k*8+2*j+1]);
            *reinterpret_cast<int4*>(dst+k*8)=pk.packed; }
    }
}


// -----------------------------------------------------------------------------
// Host launchers. Precondition: caller already cudaSetDevice(rank); launch ALL `world` ranks
// before synchronizing any of them (the in-kernel barriers need every rank in flight).
// numel must satisfy numel % (world*EPT32) == 0.
//
// launch_fused_rank (recommended, one kernel):
//   comm          : this rank's buffer of fused_comm_bytes(shard, world) bytes, peer-visible
//   peer_comm     : device array[world] of every rank's comm pointer
//   barrier_in/out: device arrays[world] of every rank's uint32 barrier buffer,
//                   fused_barrier_words(world) words each, zero-initialized once
//   flag          : MONOTONIC uint32, +1 per call (parity selects the comm column set)
//   grid          : blocks (<= MAX_BLOCKS); 0 = NVLink default rule (see below); PCIe: pass 4..8
// -----------------------------------------------------------------------------
template <int RANKS>
inline void launch_fused_rank(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint8_t** peer_comm, uint32_t** barrier_in, uint32_t** barrier_out,
    int rank, size_t numel, float SFScaleVal, uint32_t flag,
    cudaStream_t stream = 0, int TPB = 256, int grid = 0)
{
    const size_t shard = numel / RANKS;
    const size_t slot  = fused_slot_bytes(shard);
    const float  SFr   = SFScaleVal / float(RANKS);
    if (grid <= 0) {   // NVLink default (B200 sweep): one chunk/block up to 256 blocks, then 4 chunks/thread up to MAX_BLOCKS; floor 16.
        size_t chunks = (shard + size_t(TPB)*EPT32 - 1) / (size_t(TPB)*EPT32);
        size_t g = chunks < 256 ? chunks : 256; if (chunks/4 > g) g = chunks/4;
        if (g < 16) g = 16; if (g > size_t(MAX_BLOCKS)) g = MAX_BLOCKS; grid = int(g); }
    // PCIe: pass grid = 4..8 explicitly (measured 1.3-1.8x faster than the NVLink default there).
    const size_t epb = ((shard + size_t(grid)*TPB*EPT32 - 1) / (size_t(grid)*TPB*EPT32)) * (size_t(TPB)*EPT32);
    twoshot_fused_kernel<RANKS><<<grid, TPB, 0, stream>>>(input, output, peer_comm, barrier_in, barrier_out,
                                                         rank, shard, slot, epb, SFScaleVal, SFr, flag);
}

// launch_oneshot_fused_rank (one kernel, TRT-LLM oneShotAllReduceKernel structure):
//   comm : oneshot_comm_bytes(numel, RANKS) bytes per rank, peer-visible (column = whole tensor)
//   barrier : fused_barrier_words(RANKS) uint32 per rank; flag +1 per call
//   Wins over two-shot below ~2M elements on NVLink is NOT the case (both sit on the latency floor);
//   on PCIe use grid = 1 and prefer the two-shot fused kernel unless the message is tiny.
__host__ __device__ inline size_t oneshot_slot_bytes(size_t numel){ return ((numel/2 + numel/SF_VEC_SIZE) + 15) & ~size_t(15); }
__host__ __device__ inline size_t oneshot_comm_bytes(size_t numel, int world){ return size_t(2)*world*oneshot_slot_bytes(numel); }
template <int RANKS>
inline void launch_oneshot_fused_rank(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint8_t** peer_comm, uint32_t** barrier,
    int rank, size_t numel, float SFScaleVal, uint32_t flag,
    cudaStream_t stream = 0, int TPB = 256, int grid = 0)
{
    const size_t slot = oneshot_slot_bytes(numel);
    if (grid <= 0) {
        size_t chunks = (numel + size_t(TPB)*EPT32 - 1) / (size_t(TPB)*EPT32);
        size_t g = chunks < 256 ? chunks : 256; if (chunks/4 > g) g = chunks/4;
        if (g < 16) g = 16; if (g > size_t(MAX_BLOCKS)) g = MAX_BLOCKS; grid = int(g); }
    const size_t epb = ((numel + size_t(grid)*TPB*EPT32 - 1) / (size_t(grid)*TPB*EPT32)) * (size_t(TPB)*EPT32);
    oneshot_fused_kernel<RANKS><<<grid, TPB, 0, stream>>>(input, output, peer_comm, barrier, rank, numel, slot, epb, SFScaleVal, flag);
}

// launch_twoshot_rank (split path, five launches; `flag` consumes flag and flag+1):
//   payload/reduced : this rank's NVFP4 buffers (Layout::total_bytes each), peer-visible
//   peer_payloads / peer_reduced / barrier_sig : device arrays[world] of every rank's pointers
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
    const int grid    = int((numel/ELTS_PER_THREAD + TPB - 1) / TPB);
    const int grid_s  = int((shard/EPT32 + TPB - 1) / TPB);
    const int grid_ag = int((numel/EPT32 + TPB - 1) / TPB);

    quantize_kernel     <<<grid,    TPB, 0, stream>>>(input, payload, L, SFScaleVal);
    barrier_kernel      <<<1, MAX_RANKS, 0, stream>>>(barrier_sig, rank, world, flag);
    reducescatter_kernel<<<grid_s,  TPB, 0, stream>>>(peer_payloads, reduced, L, world, rank, shard, SFr);
    barrier_kernel      <<<1, MAX_RANKS, 0, stream>>>(barrier_sig, rank, world, flag + 1);
    allgather_kernel    <<<grid_ag, TPB, 0, stream>>>(peer_reduced, output, L, world, shard, rank);
}

} // namespace nvfp4_ar
