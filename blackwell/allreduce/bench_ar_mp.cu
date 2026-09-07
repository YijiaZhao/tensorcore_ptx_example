// bench_ar.cu
// Apples-to-apples: TRT-LLM's REAL custom-AR kernels (oneShot / twoShot, PUSH_MODE,
// fused block_barrier, add128b) — ported VERBATIM from
//   TensorRT-LLM/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu
// vs NVFP4-compressed one-shot / two-shot, in ONE identical single-process P2P harness.
//
// The TRT-LLM kernels here are the literal algorithm from that file (not a hand-rolled
// naive BF16): each rank PUSHes its chunk into peers' buffers, a fused flag-based
// block_barrier synchronizes corresponding blocks across GPUs (ONE kernel launch per
// allreduce — no separate barrier kernel, no cudaDeviceSynchronize between phases),
// then each rank reduces its buffer columns with vectorized add128b.
//
// Build: nvcc -arch=sm_120a -O3 -std=c++17 bench_ar.cu -o bench_ar
// Run:   ./bench_ar <numel>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

static constexpr int MAX_RANKS = 8;
#ifndef NRANKS
#define NRANKS 8          // world size (GPUs/node); override with -DNRANKS=4 for GB300 NVL (4/node)
#endif
static constexpr int MAX_BLOCKS = 64;               // TRT-LLM MAX_ALL_REDUCE_BLOCKS
static constexpr int SF_VEC_SIZE = 16;
static constexpr int ELTS_PER_THREAD = 8;
static constexpr float E2M1_MAX = 6.0f;

// ======================= TRT-LLM verbatim device helpers =======================
static inline __device__ void st_flag_release(uint32_t const& flag, uint32_t* addr) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
}
static inline __device__ uint32_t ld_flag_acquire(uint32_t* addr) {
    uint32_t flag; asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr));
    return flag;
}
using PackedBFloat16 = union { int4 packed; __nv_bfloat162 unpacked[4]; };
template <typename T> struct PackedOn16 {};
template <> struct PackedOn16<__nv_bfloat16> { using Type = PackedBFloat16; };

template <typename T>
inline __device__ int4 add128b(T& a, T& b) {
    T c;
    c.unpacked[0] = a.unpacked[0] + b.unpacked[0];
    c.unpacked[1] = a.unpacked[1] + b.unpacked[1];
    c.unpacked[2] = a.unpacked[2] + b.unpacked[2];
    c.unpacked[3] = a.unpacked[3] + b.unpacked[3];
    return c.packed;
}

// TRT-LLM block_barrier VERBATIM (flag ping-pong + per-block offset). signals: [world][ ... ]
__inline__ __device__ void block_barrier(uint32_t** signals, uint32_t const flag, size_t const local_rank,
    size_t const world_size, int const tidx, int const bidx) {
    if (tidx < world_size) {
        uint32_t flag_block_offset = (bidx + 1) * world_size;
        if (flag % 2 == 1)
            flag_block_offset += (MAX_BLOCKS + 1) * world_size;
        st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);
        uint32_t* peer = signals[local_rank] + flag_block_offset + tidx;
        while (ld_flag_acquire(peer) != flag) {}
    }
    __syncthreads();
}

// ======================= TRT-LLM one-shot (PUSH_MODE) — verbatim algorithm =======================
// comm_bufs[rank] : per-rank buffer of size [buffer_slots(=2*world)][elts_total] bf16.
//   ping (flag even) uses columns [0..world), pong (odd) uses [world..2world).
template <int RANKS>
__global__ void trt_oneshot_push(
    const __nv_bfloat16* __restrict__ local_input,
    __nv_bfloat16* __restrict__ local_output,
    __nv_bfloat16** comm_bufs,          // [world] peer comm buffers
    uint32_t** barrier_ptrs,            // [world] peer barrier signal buffers
    int local_rank, size_t elts_total, size_t elts_per_block, uint32_t flag)
{
    const int bidx = blockIdx.x, tidx = threadIdx.x;
    const int buffer_offset = (flag % 2 == 0) ? 0 : RANKS;     // ping/pong column base
    static constexpr int PACKED = 8;                            // 16/sizeof(bf16)
    using PK = PackedBFloat16;

    __nv_bfloat16* buffers[RANKS];
#pragma unroll
    for (int ii = 0; ii < RANKS; ++ii) {
        int rank = (local_rank + ii) % RANKS;
        buffers[ii] = comm_bufs[rank];                          // peer rank's buffer
    }
    const size_t chunk_start = bidx * elts_per_block + tidx * PACKED;
    const size_t chunk_end   = min((size_t)(bidx + 1) * elts_per_block, elts_total);

    // PUSH: write my chunk into every peer's buffer at my column (local_rank)
    for (size_t off = chunk_start; off < chunk_end; off += blockDim.x * PACKED) {
#pragma unroll
        for (int ii = 0; ii < RANKS; ++ii)
            *reinterpret_cast<int4*>(&buffers[ii][(size_t)(buffer_offset + local_rank) * elts_total + off])
                = *reinterpret_cast<const int4*>(&local_input[off]);
    }
    block_barrier(barrier_ptrs, flag, local_rank, RANKS, tidx, bidx);
    // reduce my own buffer's world columns
    __nv_bfloat16* mybuf = comm_bufs[local_rank];
    for (size_t off = chunk_start; off < chunk_end; off += blockDim.x * PACKED) {
        PK vals[RANKS];
#pragma unroll
        for (int ii = 0; ii < RANKS; ++ii)
            vals[ii].packed = *reinterpret_cast<const int4*>(&mybuf[(size_t)(buffer_offset + ii) * elts_total + off]);
        PK sums; sums.packed = {0,0,0,0};
#pragma unroll
        for (int rank = 0; rank < RANKS; ++rank) {
            int ii = (rank + RANKS - local_rank) % RANKS;
            sums.packed = add128b(sums, vals[ii]);
        }
        *reinterpret_cast<int4*>(&local_output[off]) = sums.packed;
    }
}

// ======================= TRT-LLM two-shot (PUSH_MODE) — verbatim algorithm =======================
// comm_bufs[rank] : per-rank buffer [2*world][elts_per_rank] bf16 (ping/pong).
template <int RANKS>
__global__ void trt_twoshot_push(
    const __nv_bfloat16* __restrict__ local_input,
    __nv_bfloat16* __restrict__ local_output,
    __nv_bfloat16** comm_bufs,
    uint32_t** barrier_in, uint32_t** barrier_out,
    int local_rank, size_t elts_total, size_t elts_per_rank, size_t elts_per_block, uint32_t flag)
{
    const int bidx = blockIdx.x, tidx = threadIdx.x;
    const int buffer_offset = (flag % 2 == 0) ? 0 : RANKS;
    static constexpr int PACKED = 8;
    using PK = PackedBFloat16;

    __nv_bfloat16* local_shared = comm_bufs[local_rank];
    __nv_bfloat16* buffers[RANKS];
    int ranks[RANKS];
#pragma unroll
    for (int ii = 0; ii < RANKS; ++ii) {
        int rank = (local_rank + ii) % RANKS;
        ranks[ii] = rank;
        buffers[ii] = comm_bufs[rank];
    }
    const size_t chunk_start = bidx * elts_per_block + tidx * PACKED;
    const size_t chunk_end   = min(chunk_start + elts_per_block, elts_per_rank);

    // 1. push each responsible shard into its owner's buffer at my column
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED) {
#pragma unroll
        for (int ii = 0; ii < RANKS; ++ii) {
            size_t offset_rank = ranks[ii] * elts_per_rank + local_offset;
            if (offset_rank >= elts_total) continue;
            *reinterpret_cast<int4*>(&buffers[ii][(size_t)(buffer_offset + local_rank) * elts_per_rank + local_offset])
                = *reinterpret_cast<const int4*>(&local_input[offset_rank]);
        }
    }
    block_barrier(barrier_in, flag, local_rank, RANKS, tidx, bidx);
    // 2. reduce my shard across ranks -> write column 0 of my buffer
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED) {
        PK vals[RANKS];
#pragma unroll
        for (int ii = 0; ii < RANKS; ++ii)
            vals[ii].packed = *reinterpret_cast<const int4*>(
                &local_shared[(size_t)(buffer_offset + ii) * elts_per_rank + local_offset]);
        PK sums; sums.packed = {0,0,0,0};
#pragma unroll
        for (int rank = 0; rank < RANKS; ++rank) {
            int ii = (rank + RANKS - local_rank) % RANKS;
            sums.packed = add128b(sums, vals[ii]);
        }
        *reinterpret_cast<int4*>(&local_shared[(size_t)buffer_offset * elts_per_rank + local_offset]) = sums.packed;
    }
    block_barrier(barrier_out, flag, local_rank, RANKS, tidx, bidx);
    // 3. all-gather: pull each shard's reduced result (col 0 of its owner) into output
    for (size_t local_offset = chunk_start; local_offset < chunk_end; local_offset += blockDim.x * PACKED) {
#pragma unroll
        for (int ii = 0; ii < RANKS; ++ii) {
            size_t offset_rank = ranks[ii] * elts_per_rank + local_offset;
            if (offset_rank >= elts_total) continue;
            *reinterpret_cast<int4*>(&local_output[offset_rank]) =
                *reinterpret_cast<const int4*>(&buffers[ii][(size_t)buffer_offset * elts_per_rank + local_offset]);
        }
    }
}

// ======================= NVFP4 (my compressed AR) =======================
struct Nvfp4Layout {
    size_t numel, packed_bytes, scale_bytes, total_bytes;
    __host__ __device__ static Nvfp4Layout make(size_t numel) {
        Nvfp4Layout L; L.numel=numel; L.packed_bytes=numel/2; L.scale_bytes=numel/SF_VEC_SIZE;
        size_t raw=L.packed_bytes+L.scale_bytes+sizeof(float); L.total_bytes=(raw+15)&~size_t(15); return L; }
    __host__ __device__ uint8_t* packed(uint8_t* b) const { return b; }
    __host__ __device__ uint8_t* scales(uint8_t* b) const { return b+packed_bytes; }
    __host__ __device__ float*   gscale(uint8_t* b) const { return reinterpret_cast<float*>(b+packed_bytes+scale_bytes); }
};
__device__ __forceinline__ float recip_ftz(float a){ float b; asm volatile("rcp.approx.ftz.f32 %0,%1;":"=f"(b):"f"(a)); return b; }
__device__ __forceinline__ float e2m1_to_float(uint8_t n){ const float lut[8]={0,.5f,1,1.5f,2,3,4,6}; float m=lut[n&7]; return (n&8)?-m:m; }
__device__ __forceinline__ uint8_t float_to_e2m1(float v){ uint8_t s=(v<0)?8:0; float a=fabsf(v); uint8_t c;
    if(a<.25f)c=0; else if(a<.75f)c=1; else if(a<1.25f)c=2; else if(a<1.75f)c=3; else if(a<2.5f)c=4; else if(a<3.5f)c=5; else if(a<5.f)c=6; else c=7; return s|c; }
__device__ __forceinline__ float ue4m3_to_float(uint8_t code){ __nv_fp8_e4m3 f; f.__x=code; return (float)f; }

// ===================== FAST NVFP4 CODEC =====================
// decode8: 8 e2m1 nibbles (uint32, nibble i = elt i) -> 8 floats.
//   2x PRMT (__byte_perm) map e2m1 magnitude -> e4m3 byte via a register LUT (same trick as
//   hopper/prmt_decode_mxfp4_to_e4m3; the e2m1 element format is identical for MXFP4/NVFP4),
//   then hardware e4m3x2 -> half2 -> float2. ~12 instr / 8 elts, works on every arch (sm_89+).
//   kMagLo/Hi: e2m1 mag 0..7 -> e4m3 bytes {0,.5,1,1.5 | 2,3,4,6}
__device__ __forceinline__ void decode8_e2m1(uint32_t uq, float d[8]){
// Decode path is chosen per arch from measurement (see README): the hardware cvt.f16x2.e2m1x2 is
// ~1-2% faster on sm_100/sm_103 (full-rate conversion pipe) but 13-37% SLOWER than the PRMT
// register-LUT on sm_120 (consumer die, low-throughput e2m1 cvt pipe). So: hardware on sm_100/103,
// PRMT on sm_120 and everything else.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=1000) && (__CUDA_ARCH__<1200)
    // TRUE hardware e2m1 decode (all Blackwell, needs explicit -gencode ...a):
    // cvt.rn.f16x2.e2m1x2 natively converts one byte (2 e2m1 nibbles) -> f16x2; low nibble ->
    // .x (elt 2k), high nibble -> .y (elt 2k+1). 4 cvt for 8 elts, no LUT, no e4m3 intermediate.
    uint32_t h0,h1,h2,h3;
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
    // Fallback (non-Blackwell): PRMT register-LUT -> e4m3 bytes -> hardware fp8 cvt. Still a
    // table lookup (software trick), just 4-wide and memory-free.
    const uint32_t kMagLo=0x3C383000u, kMagHi=0x4C484440u;
    uint32_t sel_lo=(uq&0x7u)|((uq>>4)&0x70u)|((uq>>8)&0x700u)|((uq>>12)&0x7000u);      // elts 0,2,4,6
    uint32_t sel_hi=((uq>>4)&0x7u)|((uq>>8)&0x70u)|((uq>>12)&0x700u)|((uq>>16)&0x7000u); // elts 1,3,5,7
    uint32_t olo=__byte_perm(kMagLo,kMagHi,sel_lo)|((uq<<4)&0x80808080u);
    uint32_t ohi=__byte_perm(kMagLo,kMagHi,sel_hi)|( uq     &0x80808080u);
    float2 f;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(olo&0xFFFFu),__NV_E4M3))); d[0]=f.x; d[2]=f.y;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(olo>>16),    __NV_E4M3))); d[4]=f.x; d[6]=f.y;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(ohi&0xFFFFu),__NV_E4M3))); d[1]=f.x; d[3]=f.y;
    f=__half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(ohi>>16),    __NV_E4M3))); d[5]=f.x; d[7]=f.y;
#endif
}
// encode8: 8 floats (already scaled into e2m1 range) -> uint32 of 8 e2m1 nibbles (nibble i = elt i).
//   ALL Blackwell (sm_100 B200, sm_101, sm_103 GB300, sm_120 RTX PRO): hardware
//   cvt.rn.satfinite.e2m1x2.f32, operand order per TRT-LLM fp32_vec_to_e2m1 (%2 -> upper
//   nibble, %1 -> lower). Authoritative arch matrix = cuda_fp4.hpp guard:
//     __CUDA_ARCH__>=1000 && (HAS_FEATURE(SM100_ALL)||SM101_ALL||SM120_ALL)  (+SM103 in 13.x)
//   GOTCHA (cost us a wrong conclusion): this is an arch-SPECIFIC ("a") feature, and the
//   `-arch=sm_XXXa` shorthand does NOT enable it in nvcc 13.x (PTX lands in compute_XXX and
//   ptxas says "Feature 'cvt.e2m1x2.f32' not supported on .target 'sm_XXX'"). You MUST build
//   with the explicit form, e.g.:
//       -gencode arch=compute_120a,code=sm_120a   (likewise 100a / 103a)
//   Non-Blackwell archs fall back to a branchless compare-sum (predicated, no divergence).
__device__ __forceinline__ uint32_t encode8_e2m1(const float v[8]){
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=1000)
    uint32_t val;
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

__global__ void nvfp4_quantize(const __nv_bfloat16* __restrict__ in, uint8_t* __restrict__ ob, Nvfp4Layout L, float SF){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/ELTS_PER_THREAD) return;
    const size_t base=tid*ELTS_PER_THREAD; float2 fp2[4]; float mx=0;
#pragma unroll
    for(int i=0;i<4;i++){ __nv_bfloat162 v=*reinterpret_cast<const __nv_bfloat162*>(&in[base+i*2]); fp2[i]=__bfloat1622float2(v); mx=fmaxf(mx,fmaxf(fabsf(fp2[i].x),fabsf(fp2[i].y))); }
    mx=fmaxf(__shfl_xor_sync(0xffffffffu,mx,1),mx);
    float SFv=SF*(mx*recip_ftz(E2M1_MAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0;
    float sv[8];
#pragma unroll
    for(int i=0;i<4;i++){ sv[2*i]=fp2[i].x*os; sv[2*i+1]=fp2[i].y*os; }
    uint32_t e=encode8_e2m1(sv);
    *reinterpret_cast<uint32_t*>(&L.packed(ob)[base/2])=e;
    if((tid&1)==0) L.scales(ob)[base/SF_VEC_SIZE]=sf.__x;
    if(tid==0) *L.gscale(ob)=SF;
}
// NVFP4 one-shot: each rank reads every peer's payload, dequant->fp32 accumulate, bf16 out.
__global__ void nvfp4_oneshot(uint8_t** peer, __nv_bfloat16* __restrict__ out, Nvfp4Layout L, int world){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/ELTS_PER_THREAD) return;
    const size_t base=tid*ELTS_PER_THREAD, po=base/2, si=base/SF_VEC_SIZE; float acc[8];
#pragma unroll
    for(int i=0;i<8;i++) acc[i]=0;
    for(int r=0;r<world;r++){ uint8_t* b=peer[r]; uint32_t e=*reinterpret_cast<const uint32_t*>(&L.packed(b)[po]);
        float bs=ue4m3_to_float(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
        float dq[8]; decode8_e2m1(e,dq);
        for(int i=0;i<8;i++) acc[i]+=dq[i]*bs; }
#pragma unroll
    for(int i=0;i<8;i++) out[base+i]=__float2bfloat16(acc[i]);
}
static constexpr int EPT32 = 32;   // RS/AG: 32 elts/thread = one 16-byte packed load + one 2-byte scale load
__global__ void nvfp4_reducescatter(uint8_t** peer, uint8_t* my, Nvfp4Layout L, int world, int rank, size_t shard, float SF){
    const size_t lt=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(lt>=shard/EPT32) return;
    const size_t g=size_t(rank)*shard+lt*EPT32, po=g/2, si=g/SF_VEC_SIZE; float acc[32];
#pragma unroll
    for(int i=0;i<32;i++) acc[i]=0;
    // peer order rotated by rank so the 8 ranks never all hit the same peer at the same time (TRT-LLM does the same)
    for(int ii=0;ii<world;ii++){ int r=(rank+ii)%world; uint8_t* b=peer[r];
        uint4 e=*reinterpret_cast<const uint4*>(&L.packed(b)[po]);
        uint16_t s2=*reinterpret_cast<const uint16_t*>(&L.scales(b)[si]);
        float gs=recip_ftz(*L.gscale(b));
        float bs0=ue4m3_to_float(uint8_t(s2&0xFF))*gs, bs1=ue4m3_to_float(uint8_t(s2>>8))*gs;
        uint32_t w[4]={e.x,e.y,e.z,e.w};
#pragma unroll
        for(int q=0;q<4;q++){ float dq[8]; decode8_e2m1(w[q],dq); float bs=(q<2)?bs0:bs1;
#pragma unroll
            for(int i=0;i<8;i++) acc[q*8+i]+=dq[i]*bs; }
    }
    uint32_t ow[4]; uint8_t sfb[2];
#pragma unroll
    for(int h=0;h<2;h++){ float mx=0;
#pragma unroll
        for(int i=0;i<16;i++) mx=fmaxf(mx,fabsf(acc[h*16+i]));
        float SFv=SF*(mx*recip_ftz(E2M1_MAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
        float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0; sfb[h]=sf.__x;
#pragma unroll
        for(int q=0;q<2;q++){ float sv[8];
#pragma unroll
            for(int i=0;i<8;i++) sv[i]=acc[h*16+q*8+i]*os;
            ow[h*2+q]=encode8_e2m1(sv); }
    }
    *reinterpret_cast<uint4*>(&L.packed(my)[po])=make_uint4(ow[0],ow[1],ow[2],ow[3]);
    *reinterpret_cast<uint16_t*>(&L.scales(my)[si])=uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
    if(lt==0) *L.gscale(my)=SF;
}
__global__ void nvfp4_allgather(uint8_t** peer, __nv_bfloat16* __restrict__ out, Nvfp4Layout L, int world, size_t shard, int rank){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/EPT32) return;
    // owner sequence rotated by rank: otherwise every rank reads owner 0's shard first (7 readers on one NVLink egress)
    const size_t tps=shard/EPT32; const int slot=int(tid/tps); const int owner=(slot+rank)%world;
    const size_t g=size_t(owner)*shard+(tid-size_t(slot)*tps)*EPT32; uint8_t* b=peer[owner];
    uint4 e=*reinterpret_cast<const uint4*>(&L.packed(b)[g/2]);
    uint16_t s2=*reinterpret_cast<const uint16_t*>(&L.scales(b)[g/SF_VEC_SIZE]);
    float gs=recip_ftz(*L.gscale(b));
    float bs0=ue4m3_to_float(uint8_t(s2&0xFF))*gs, bs1=ue4m3_to_float(uint8_t(s2>>8))*gs;
    uint32_t w[4]={e.x,e.y,e.z,e.w};
#pragma unroll
    for(int q=0;q<4;q++){ float dq[8]; decode8_e2m1(w[q],dq); float bs=(q<2)?bs0:bs1;
        __nv_bfloat162 o[4];
#pragma unroll
        for(int i=0;i<4;i++) o[i]=__floats2bfloat162_rn(dq[2*i]*bs,dq[2*i+1]*bs);
        *reinterpret_cast<uint4*>(&out[g+q*8])=*reinterpret_cast<uint4*>(o); }
}


// ======================= NVFP4 fused two-shot — TRT-LLM twoShotAllReduceKernel structure =======================
// Same skeleton as trt_twoshot_push above (PUSH_MODE, rank-rotated peer order, 16-byte accesses, two in-kernel
// block_barriers, flag-parity ping-pong columns). Payload is NVFP4: 32 elts/thread = 16 B packed + 2 B block scales.
// comm buffer per rank: [2*RANKS columns][slot_bytes], slot = shard/2 packed bytes followed by shard/16 scale bytes.
static constexpr int NV_MAX_BLOCKS = 2048;
__inline__ __device__ void block_barrier_mb(uint32_t** signals, uint32_t const flag, size_t const local_rank,
    size_t const world_size, int const tidx, int const bidx, int const max_blocks) {
    if (tidx < world_size) {
        uint32_t flag_block_offset = (bidx + 1) * world_size;
        if (flag % 2 == 1) flag_block_offset += (max_blocks + 1) * world_size;
        st_flag_release(flag, signals[tidx] + flag_block_offset + local_rank);
        uint32_t* peer = signals[local_rank] + flag_block_offset + tidx;
        while (ld_flag_acquire(peer) != flag) {}
    }
    __syncthreads();
}
__device__ __forceinline__ void nv_q16(const float* v, float SF, uint32_t* ow, uint8_t& sfb){
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
__device__ __forceinline__ void nv_dq32_acc(uint4 e, uint16_t s2, float invSF, float* acc){
    float bs0=ue4m3_to_float(uint8_t(s2&0xFF))*invSF, bs1=ue4m3_to_float(uint8_t(s2>>8))*invSF;
    uint32_t w[4]={e.x,e.y,e.z,e.w};
#pragma unroll
    for(int q=0;q<4;q++){ float dq[8]; decode8_e2m1(w[q],dq); float bs=(q<2)?bs0:bs1;
#pragma unroll
        for(int i=0;i<8;i++) acc[q*8+i]+=dq[i]*bs; }
}
template <int RANKS>
__global__ void nvfp4_twoshot_fused(
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

    // 1. quantize my copy of shard ranks[ii] and push it into the owner's buffer at my column
    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
#pragma unroll
        for(int ii=0;ii<RANKS;ii++){
            const __nv_bfloat16* src=local_input+(size_t)ranks[ii]*elts_per_rank+lo;
            float v[32];
#pragma unroll
            for(int k=0;k<4;k++){ PackedBFloat16 pk; pk.packed=*reinterpret_cast<const int4*>(src+k*8);
#pragma unroll
                for(int j=0;j<4;j++){ float2 f=__bfloat1622float2(pk.unpacked[j]); v[k*8+2*j]=f.x; v[k*8+2*j+1]=f.y; } }
            uint32_t ow[4]; uint8_t sfb[2];
            nv_q16(v,SF,ow,sfb[0]); nv_q16(v+16,SF,ow+2,sfb[1]);
            uint8_t* slot=buffers[ii]+(size_t)(buffer_offset+local_rank)*slot_bytes;
            *reinterpret_cast<uint4*>(slot+lo/2)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
            *reinterpret_cast<uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE)=uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
        }
    }
    block_barrier_mb(barrier_in, flag, local_rank, RANKS, tidx, bidx, NV_MAX_BLOCKS);
    // 2. reduce my shard across the RANKS columns of my own buffer (local reads), requantize into column 0
    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
        float acc[32];
#pragma unroll
        for(int i=0;i<32;i++) acc[i]=0.f;
#pragma unroll
        for(int rank=0;rank<RANKS;rank++){ int ii=(rank+RANKS-local_rank)%RANKS;
            const uint8_t* slot=local_shared+(size_t)(buffer_offset+ii)*slot_bytes;
            uint4 e=*reinterpret_cast<const uint4*>(slot+lo/2);
            uint16_t s2=*reinterpret_cast<const uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE);
            nv_dq32_acc(e,s2,invSF,acc); }
        uint32_t ow[4]; uint8_t sfb[2];
        nv_q16(acc,SFr,ow,sfb[0]); nv_q16(acc+16,SFr,ow+2,sfb[1]);
        uint8_t* slot=local_shared+(size_t)buffer_offset*slot_bytes;
        *reinterpret_cast<uint4*>(slot+lo/2)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
        *reinterpret_cast<uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE)=uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
    }
    block_barrier_mb(barrier_out, flag, local_rank, RANKS, tidx, bidx, NV_MAX_BLOCKS);
    // 3. all-gather: pull each owner's reduced column 0, dequantize, write bf16 output
    for(size_t lo=chunk_start; lo<chunk_end; lo+=blockDim.x*EPT32){
#pragma unroll
        for(int ii=0;ii<RANKS;ii++){
            const uint8_t* slot=buffers[ii]+(size_t)buffer_offset*slot_bytes;
            uint4 e=*reinterpret_cast<const uint4*>(slot+lo/2);
            uint16_t s2=*reinterpret_cast<const uint16_t*>(slot+packed_bytes+lo/SF_VEC_SIZE);
            float acc[32];
#pragma unroll
            for(int i=0;i<32;i++) acc[i]=0.f;
            nv_dq32_acc(e,s2,invSFr,acc);
            __nv_bfloat16* dst=local_output+(size_t)ranks[ii]*elts_per_rank+lo;
#pragma unroll
            for(int k=0;k<4;k++){ PackedBFloat16 pk;
#pragma unroll
                for(int j=0;j<4;j++) pk.unpacked[j]=__floats2bfloat162_rn(acc[k*8+2*j],acc[k*8+2*j+1]);
                *reinterpret_cast<int4*>(dst+k*8)=pk.packed; }
        }
    }
}

// simple counter barrier for the NVFP4 multi-phase path (separate-launch, like the standalone kernel)
__global__ void ctr_barrier(uint64_t** sig, int rank, int world, uint64_t flag){
    int t=threadIdx.x;
    if(blockIdx.x==0 && t<world){
        asm volatile("st.global.release.sys.b64 [%1], %0;"::"l"(flag),"l"(&sig[t][rank]));
        uint64_t f; do{ asm volatile("ld.global.acquire.sys.b64 %0,[%1];":"=l"(f):"l"(&sig[rank][t])); }while(f!=flag);
    }
    __syncthreads();
}

// ======================= FP8 E4M3 (TRT-LLM low-precision AR recipe) =======================
// Same two-shot algorithm & same per-16 UE4M3 block-scale layout as NVFP4; only the value
// format (E4M3, 1 byte) and packed width differ. Payload = 1.0 + 0.0625 = 1.0625 B/elt.
static constexpr float FP8_QMAX = 448.0f;
struct Fp8Layout {
    size_t numel, packed_bytes, scale_bytes, total_bytes;
    __host__ __device__ static Fp8Layout make(size_t n){ Fp8Layout L; L.numel=n; L.packed_bytes=n; L.scale_bytes=n/SF_VEC_SIZE;
        size_t raw=L.packed_bytes+L.scale_bytes+sizeof(float); L.total_bytes=(raw+15)&~size_t(15); return L; }
    __host__ __device__ uint8_t* packed(uint8_t* b) const { return b; }
    __host__ __device__ uint8_t* scales(uint8_t* b) const { return b+packed_bytes; }
    __host__ __device__ float*   gscale(uint8_t* b) const { return reinterpret_cast<float*>(b+packed_bytes+scale_bytes); }
};
__device__ __forceinline__ uint8_t f2e4m3(float v){ return __nv_fp8_e4m3(v).__x; }
__device__ __forceinline__ float e4m3f(uint8_t x){ __nv_fp8_e4m3 f; f.__x=x; return (float)f; }

__global__ void fp8_quantize(const __nv_bfloat16* __restrict__ in, uint8_t* __restrict__ ob, Fp8Layout L, float SF){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/ELTS_PER_THREAD) return;
    const size_t base=tid*ELTS_PER_THREAD; float2 fp2[4]; float mx=0;
#pragma unroll
    for(int i=0;i<4;i++){ __nv_bfloat162 v=*reinterpret_cast<const __nv_bfloat162*>(&in[base+i*2]); fp2[i]=__bfloat1622float2(v); mx=fmaxf(mx,fmaxf(fabsf(fp2[i].x),fabsf(fp2[i].y))); }
    mx=fmaxf(__shfl_xor_sync(0xffffffffu,mx,1),mx);
    float SFv=SF*(mx*recip_ftz(FP8_QMAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0; uint64_t e=0;
#pragma unroll
    for(int i=0;i<4;i++){ e|=uint64_t(f2e4m3(fp2[i].x*os))<<((2*i)*8); e|=uint64_t(f2e4m3(fp2[i].y*os))<<((2*i+1)*8); }
    *reinterpret_cast<uint64_t*>(&L.packed(ob)[base])=e;
    if((tid&1)==0) L.scales(ob)[base/SF_VEC_SIZE]=sf.__x;
    if(tid==0) *L.gscale(ob)=SF;
}
__global__ void fp8_oneshot(uint8_t** peer, __nv_bfloat16* __restrict__ out, Fp8Layout L, int world){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/ELTS_PER_THREAD) return;
    const size_t base=tid*ELTS_PER_THREAD, si=base/SF_VEC_SIZE; float acc[8];
#pragma unroll
    for(int i=0;i<8;i++) acc[i]=0;
    for(int r=0;r<world;r++){ uint8_t* b=peer[r]; uint64_t e=*reinterpret_cast<const uint64_t*>(&L.packed(b)[base]);
        float bs=e4m3f(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
        for(int i=0;i<8;i++) acc[i]+=e4m3f((e>>(i*8))&0xFF)*bs; }
#pragma unroll
    for(int i=0;i<8;i++) out[base+i]=__float2bfloat16(acc[i]);
}
__global__ void fp8_reducescatter(uint8_t** peer, uint8_t* my, Fp8Layout L, int world, int rank, size_t shard, float SFr){
    const size_t lt=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(lt>=shard/ELTS_PER_THREAD) return;
    const size_t g=size_t(rank)*shard+lt*ELTS_PER_THREAD, si=g/SF_VEC_SIZE; float acc[8];
#pragma unroll
    for(int i=0;i<8;i++) acc[i]=0;
    for(int ii=0;ii<world;ii++){ int r=(rank+ii)%world; uint8_t* b=peer[r]; uint64_t e=*reinterpret_cast<const uint64_t*>(&L.packed(b)[g]);
        float bs=e4m3f(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
        for(int i=0;i<8;i++) acc[i]+=e4m3f((e>>(i*8))&0xFF)*bs; }
    float mx=0;
#pragma unroll
    for(int i=0;i<8;i++) mx=fmaxf(mx,fabsf(acc[i]));
    mx=fmaxf(__shfl_xor_sync(0xffffffffu,mx,1),mx);
    float SFv=SFr*(mx*recip_ftz(FP8_QMAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SFr)):0; uint64_t e=0;
#pragma unroll
    for(int i=0;i<8;i++) e|=uint64_t(f2e4m3(acc[i]*os))<<(i*8);
    *reinterpret_cast<uint64_t*>(&L.packed(my)[g])=e;
    if((lt&1)==0) L.scales(my)[si]=sf.__x;
    if(lt==0) *L.gscale(my)=SFr;
}
__global__ void fp8_allgather(uint8_t** peer_reduced, __nv_bfloat16* __restrict__ out, Fp8Layout L, int world, size_t shard, int rank){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/ELTS_PER_THREAD) return;
    const size_t tps=shard/ELTS_PER_THREAD; const int slot=int(tid/tps); const int owner=(slot+rank)%world;
    const size_t g=size_t(owner)*shard+(tid-size_t(slot)*tps)*ELTS_PER_THREAD; uint8_t* b=peer_reduced[owner];
    const size_t si=g/SF_VEC_SIZE; uint64_t e=*reinterpret_cast<const uint64_t*>(&L.packed(b)[g]);
    float bs=e4m3f(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
    for(int i=0;i<8;i++) out[g+i]=__float2bfloat16(e4m3f((e>>(i*8))&0xFF)*bs);
}


// ===================== TRT-LLM FP8 LOW-PRECISION AR — VERBATIM two-shot =====================
// Ported from TensorRT-LLM communicationKernels/customLowPrecisionAllReduceKernels.cu:
//   lowPrecisionPreprocessKernel + lowPrecisionTwoShotAllReduceKernel (First/SecondStage).
// PULL mode. Each warp: 31 lanes x 16 fp8 = 496 data elts + 1 scale slot (lane 30 writes the
// FP32 scale at warp offset 496; lane 31's 16-byte load reads it back). block=512 (16 warps),
// elts_per_block (data)=7936, with-scale=8192. Dequant -> FP32 reduce -> requant ONCE, in place
// in the owner's buffer, then all-gather + dequant to BF16.
// NOTE: TRT-LLM's dispatch uses this kernel for RANKS<=4 (grid=16); for 8 ranks it switches to
// a NUMA-hierarchical 3-stage variant built for PCIe 2-NUMA boxes. We instantiate this exact
// two-shot for NRANKS so the same TRT-LLM algorithm is measured at every rank count.
static constexpr int LP_WARPSIZE=32, LP_BLOCK=512, LP_WARPS=16, LP_MAX_BLOCKS=8;
static constexpr int LP_ELTS_PER_BLOCK=16*(LP_WARPSIZE-1)*LP_WARPS;   // 7936 data elts / block
static constexpr int LP_ELTS_PER_BLOCK_WS=16*LP_BLOCK;                // 8192 slots incl. scale
using LP_PackedF8  = union { int4 packed; __nv_fp8_e4m3 unpacked[16]; };
using LP_PackedF8_8= union { int2 packed; __nv_fp8_e4m3 unpacked[8]; };
using LP_PackedBF  = union { int4 packed; __nv_bfloat16 unpacked[8]; };
__device__ __forceinline__ float lp_warp_reduce_max(float v){
    v=fmaxf(__shfl_xor_sync(~0u,v,16),v); v=fmaxf(__shfl_xor_sync(~0u,v,8),v); v=fmaxf(__shfl_xor_sync(~0u,v,4),v);
    v=fmaxf(__shfl_xor_sync(~0u,v,2),v);  v=fmaxf(__shfl_xor_sync(~0u,v,1),v); return v; }
// TRT-LLM low-precision multi_gpu_barrier (volatile flag version, verbatim)
__device__ __forceinline__ void lp_multi_gpu_barrier(uint64_t** signals, uint64_t flag, size_t rank, size_t world, int tidx, int bidx){
    uint64_t volatile* my=signals[rank];
    if(tidx<(int)world){ if(bidx==0) signals[tidx][rank]=flag; while(my[tidx]!=flag){} }
    __syncthreads();
}
// preprocess: BF16 -> FP8 with per-warp scale (T_IN=bf16: 8/int4, T_OUT=fp8: 16/int4)
template<int RANKS>
__global__ void trt_fp8_preprocess(const __nv_bfloat16* __restrict__ input, size_t elts_per_rank_in, size_t elts_per_rank_out, __nv_fp8_e4m3* __restrict__ output){
    constexpr float QUANT_MAX=448.f;
    constexpr int output_rounds=2, elts_per_thread=16, elts_per_round=8, elts_per_warp_per_round=elts_per_round*LP_WARPSIZE;
    constexpr int NUM_IN=(LP_WARPSIZE-1)*elts_per_thread, NUM_OUT=LP_WARPSIZE*elts_per_thread;
    const int target_rank=blockIdx.x/(gridDim.x/RANKS), local_bid=blockIdx.x%(gridDim.x/RANKS);
    input+=elts_per_rank_in*target_rank; output+=elts_per_rank_out*target_rank;
    const int lane=threadIdx.x%LP_WARPSIZE, wid=threadIdx.x/LP_WARPSIZE;
    LP_PackedBF vals[output_rounds];
    const size_t start_in=(size_t)NUM_IN*LP_WARPS*local_bid+(size_t)wid*NUM_IN;
    const size_t start_out=(size_t)NUM_OUT*LP_WARPS*local_bid+(size_t)wid*NUM_OUT;
#pragma unroll
    for(int i=0;i<output_rounds;i++){ int lo=lane*elts_per_round+elts_per_warp_per_round*i; size_t go=start_in+lo;
        if(lo<NUM_IN && go<elts_per_rank_in) vals[i].packed=*reinterpret_cast<const int4*>(input+start_in+lo);
        else { for(int j=0;j<elts_per_round;j++) vals[i].unpacked[j]=__float2bfloat16(0.f); } }
    float scalar=0.f;
    for(int i=0;i<output_rounds;i++) for(int j=0;j<elts_per_round;j++) scalar=fmaxf(fabsf(__bfloat162float(vals[i].unpacked[j])),scalar);
    scalar=lp_warp_reduce_max(scalar); if(scalar!=0.f) scalar=QUANT_MAX/scalar;
    LP_PackedF8_8 ov[output_rounds]; for(int i=0;i<output_rounds;i++) ov[i].packed=make_int2(0,0);
    for(int i=0;i<output_rounds;i++){ int lw=lane*elts_per_round+elts_per_warp_per_round*i;
        if(lw<NUM_IN){ for(int j=0;j<elts_per_round;j++){ float o=__bfloat162float(vals[i].unpacked[j]); if(scalar!=0.f) o*=scalar; ov[i].unpacked[j]=__nv_fp8_e4m3(o);} }
        else if(lw==NUM_IN){ *reinterpret_cast<float*>(&ov[i])=scalar; } }
#pragma unroll
    for(int i=0;i<output_rounds;i++){ int lw=lane*elts_per_round+elts_per_warp_per_round*i; *reinterpret_cast<int2*>(output+start_out+lw)=ov[i].packed; }
}
// first stage: reduce-scatter my shard (dequant -> FP32 sum -> requant in place into own buffer)
template<int RANKS>
__device__ void trt_fp8_first_stage(int myrank, size_t elts_per_rank, __nv_fp8_e4m3** input, float* smem){
    constexpr float QUANT_MAX=448.f; constexpr int elts_per_thread=16, NUM_IN=LP_WARPSIZE*elts_per_thread;
    const int lane=threadIdx.x%LP_WARPSIZE, bid=blockIdx.x, wid=threadIdx.x/LP_WARPSIZE;
    const size_t in_start=((size_t)bid*LP_WARPS+wid)*NUM_IN+(size_t)lane*elts_per_thread;
    float* sm=&smem[RANKS*wid]; const size_t rank_offset=elts_per_rank*myrank;
    for(size_t lo=in_start; lo<elts_per_rank; lo+=(size_t)gridDim.x*blockDim.x*elts_per_thread){
        float sums[elts_per_thread];
#pragma unroll
        for(int i=0;i<elts_per_thread;i++) sums[i]=0.f;
        LP_PackedF8 vals[RANKS];
#pragma unroll
        for(int ii=0;ii<RANKS;ii++) vals[ii].packed=*reinterpret_cast<const int4*>(&input[ii][lo+rank_offset]);
        if(lane==LP_WARPSIZE-1){
#pragma unroll
            for(int ii=0;ii<RANKS;ii++){ float* ts=(float*)(&vals[ii]); sm[ii]=ts[0]; } }
        __syncwarp();
        if(lane<LP_WARPSIZE-1){
            for(int ii=0;ii<RANKS;ii++){
#pragma unroll
                for(int j=0;j<elts_per_thread;j++){ float v=(float)vals[ii].unpacked[j]; sums[j]+= (sm[ii]!=0.f)? v/sm[ii] : v; } } }
        float scalar=0.f;
        if(lane<LP_WARPSIZE-1){
#pragma unroll
            for(int i=0;i<elts_per_thread;i++) scalar=fmaxf(fabsf(sums[i]),scalar); }
        scalar=lp_warp_reduce_max(scalar); if(scalar!=0.f) scalar=QUANT_MAX/scalar;
        LP_PackedF8 tmp;
        if(lane<LP_WARPSIZE-1){
#pragma unroll
            for(int i=0;i<elts_per_thread;i++){ float t=sums[i]; if(scalar!=0.f) t*=scalar; tmp.unpacked[i]=__nv_fp8_e4m3(t); } }
        else { tmp.packed=make_int4(0,0,0,0); ((float*)(&tmp))[0]=scalar; }
        *reinterpret_cast<int4*>(input[0]+lo+rank_offset)=tmp.packed;
    }
}
// second stage: all-gather every shard from its owner + dequant -> BF16 output
template<int RANKS>
__device__ void trt_fp8_second_stage(size_t in_elts_per_rank, size_t out_elts_per_rank, __nv_fp8_e4m3** input, __nv_bfloat16* output, float* smem, const int* dst_rank){
    constexpr int elts_per_thread=16, output_rounds=2, depack=8;
    constexpr int NUM_IN=LP_WARPSIZE*elts_per_thread, NUM_OUT=(LP_WARPSIZE-1)*elts_per_thread;
    const int lane=threadIdx.x%LP_WARPSIZE, bid=blockIdx.x, wid=threadIdx.x/LP_WARPSIZE;
    const size_t in_start=((size_t)bid*LP_WARPS+wid)*NUM_IN+(size_t)lane*elts_per_thread;
    const size_t out_start=((size_t)bid*LP_WARPS+wid)*NUM_OUT+(size_t)lane*elts_per_thread;
    float* sm=&smem[RANKS*wid];
    LP_PackedF8 vals[RANKS];
    for(size_t io=in_start, oo=out_start; io<in_elts_per_rank; io+=(size_t)gridDim.x*LP_WARPS*NUM_IN, oo+=(size_t)gridDim.x*LP_WARPS*NUM_OUT){
#pragma unroll
        for(int ii=0;ii<RANKS;ii++){ size_t to=(size_t)dst_rank[ii]*in_elts_per_rank+io; if(io<in_elts_per_rank) vals[ii].packed=*reinterpret_cast<const int4*>(&input[ii][to]); }
        if(lane==LP_WARPSIZE-1){
#pragma unroll
            for(int ii=0;ii<RANKS;ii++){ float* ts=(float*)(&vals[ii]); sm[ii]=ts[0]; } }
        __syncwarp();
        for(int ii=0;ii<RANKS;ii++){
            float scale=sm[ii]; size_t too=(size_t)dst_rank[ii]*out_elts_per_rank+oo;
            if(oo<out_elts_per_rank && lane<LP_WARPSIZE-1){
                for(int jj=0;jj<output_rounds;jj++){ LP_PackedBF o;
#pragma unroll
                    for(int kk=0;kk<depack;kk++){ float t=(float)vals[ii].unpacked[kk+jj*depack]; if(scale!=0.f) t/=scale; o.unpacked[kk]=__float2bfloat16(t); }
                    *reinterpret_cast<int4*>(output+too+jj*depack)=o.packed; } } }
    }
}
// fused driver (verbatim structure): multi_gpu_barrier -> first stage -> per-block flag barrier -> second stage
template<int RANKS>
__global__ void trt_fp8_twoshot(__nv_fp8_e4m3** comm, uint64_t** barrier_in, __nv_bfloat16* output,
                                int local_rank, size_t elts_per_rank, size_t buffer_elts_per_rank, uint64_t flag){
    const int bidx=blockIdx.x, tidx=threadIdx.x;
    extern __shared__ float smem[];
    lp_multi_gpu_barrier(barrier_in, flag, local_rank, RANKS, tidx, bidx);
    __nv_fp8_e4m3* src[RANKS]; int dst[RANKS];
#pragma unroll
    for(int ii=0;ii<RANKS;ii++){ int r=(local_rank+ii)%RANKS; src[ii]=comm[r]; dst[ii]=r; }
    trt_fp8_first_stage<RANKS>(local_rank, buffer_elts_per_rank, src, smem);
    __syncthreads();
    if(tidx<RANKS){
        uint32_t off=RANKS+bidx*RANKS;
        asm volatile("st.global.release.sys.b64 [%1], %0;"::"l"(flag),"l"(barrier_in[tidx]+off+local_rank));
        uint64_t rb=0; uint64_t* pb=barrier_in[local_rank]+off+tidx;
        do{ asm volatile("ld.global.acquire.sys.b64 %0,[%1];":"=l"(rb):"l"(pb)); }while(rb!=flag);
    }
    __syncthreads();
    trt_fp8_second_stage<RANKS>(buffer_elts_per_rank, elts_per_rank, src, output, smem+RANKS*LP_WARPS, dst);
}

static float host_global_scale(const float* h, size_t n){ float a=0; for(size_t i=0;i<n;i++) a=fmaxf(a,fabsf(h[i])); if(a==0) return 1.f; return (E2M1_MAX*448.f)/a; }
static float host_global_scale_fp8(const float* h, size_t n){ float a=0; for(size_t i=0;i<n;i++) a=fmaxf(a,fabsf(h[i])); if(a==0) return 1.f; return (FP8_QMAX*448.f)/a; }


// ============================ multi-process main (fabric handles) ============================
#include <cuda.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sys/stat.h>
#include <string>
#define CU_CHECK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ const char* _s=nullptr; cuGetErrorString(_r,&_s); \
    fprintf(stderr,"CU %s:%d %s -> %s\n",__FILE__,__LINE__,#x,_s?_s:"?"); exit(1);} }while(0)

static int LOCAL=0, NPROC=1, PROC=0, WORLD=0;
struct ShBuf { CUdeviceptr va=0; size_t size=0; CUmemGenericAllocationHandle h=0; CUmemFabricHandle fh{}; };
static void set_access_local(CUdeviceptr va, size_t size){
    std::vector<CUmemAccessDesc> acc(LOCAL);
    for(int d=0;d<LOCAL;d++){ acc[d].location.type=CU_MEM_LOCATION_TYPE_DEVICE; acc[d].location.id=d; acc[d].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
    CU_CHECK(cuMemSetAccess(va,size,acc.data(),LOCAL));
}
static ShBuf alloc_shared(int local_dev, size_t bytes){
    CUmemAllocationProp p={}; p.type=CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id=local_dev; p.requestedHandleTypes=CU_MEM_HANDLE_TYPE_FABRIC;
    size_t gran=0; CU_CHECK(cuMemGetAllocationGranularity(&gran,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    ShBuf b; b.size=((bytes+gran-1)/gran)*gran; if(!b.size) b.size=gran;
    CU_CHECK(cuMemCreate(&b.h,b.size,&p,0));
    CU_CHECK(cuMemAddressReserve(&b.va,b.size,0,0,0));
    CU_CHECK(cuMemMap(b.va,b.size,0,b.h,0));
    set_access_local(b.va,b.size);
    CU_CHECK(cuMemExportToShareableHandle(&b.fh,b.h,CU_MEM_HANDLE_TYPE_FABRIC,0));
    return b;
}
static CUdeviceptr map_remote(const CUmemFabricHandle& fh, size_t size){
    CUmemGenericAllocationHandle h; CUmemFabricHandle tmp=fh;
    CU_CHECK(cuMemImportFromShareableHandle(&h,(void*)&tmp,CU_MEM_HANDLE_TYPE_FABRIC));
    CUdeviceptr va; CU_CHECK(cuMemAddressReserve(&va,size,0,0,0)); CU_CHECK(cuMemMap(va,size,0,h,0));
    set_access_local(va,size); return va;
}
static void fs_barrier(const std::string& dir, const std::string& name){
    { std::ofstream f(dir+"/"+name+"."+std::to_string(PROC)); f<<"1"; }
    for(;;){ int n=0; for(int p=0;p<NPROC;p++){ struct stat st; if(stat((dir+"/"+name+"."+std::to_string(p)).c_str(),&st)==0) n++; }
        if(n==NPROC) return; std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
}
enum { B_PL=0,B_RED,B_F8,B_F8RED,B_C1,B_C2,B_BA,B_BB,B_CTR,B_LPC,B_LPB,B_NVC,B_NVBA,B_NVBB,NB };
struct RankHandles { CUmemFabricHandle fh[NB]; size_t sz[NB]; };

int main(int argc,char**argv){
    size_t numel=(argc>1)?strtoull(argv[1],0,10):(1u<<20);
    const char* e; NPROC=(e=getenv("SLURM_NTASKS"))?atoi(e):1; PROC=(e=getenv("SLURM_PROCID"))?atoi(e):0;
    CUDA_CHECK(cudaGetDeviceCount(&LOCAL)); WORLD=NPROC*LOCAL;
    if(WORLD!=NRANKS){ printf("[p%d] WORLD=%d (NPROC %d x LOCAL %d) != NRANKS=%d\n",PROC,WORLD,NPROC,LOCAL,NRANKS); return 1; }
    if(numel%(size_t(WORLD)*EPT32)!=0){ printf("numel must be %d-aligned\n",WORLD*EPT32); return 1; }
    const int world=WORLD; const size_t shard=numel/world;
    std::string jid=(e=getenv("SLURM_JOB_ID"))?e:"local"; std::string dir="/work/hs_"+jid+"_"+std::to_string(numel); mkdir(dir.c_str(),0777);
    CU_CHECK(cuInit(0));
    for(int d=0;d<LOCAL;d++){ CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaFree(0));
        for(int j=0;j<LOCAL;j++) if(j!=d){ int can=0; cudaDeviceCanAccessPeer(&can,d,j); if(can){ cudaError_t r=cudaDeviceEnablePeerAccess(j,0); if(r!=cudaSuccess&&r!=cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(r);} } }
    CUDA_CHECK(cudaSetDevice(0));

    Nvfp4Layout L=Nvfp4Layout::make(numel); Fp8Layout FL=Fp8Layout::make(numel);
    // host data for ALL ranks (same seed on every process) -> identical reference everywhere
    std::vector<std::vector<float>> hin(world,std::vector<float>(numel)); std::vector<float> ref(numel,0);
    srand(1234); for(int r=0;r<world;r++) for(size_t i=0;i<numel;i++){ float v=(rand()/float(RAND_MAX))*2-1; hin[r][i]=v; ref[i]+=v; }
    float SF=host_global_scale(hin[0].data(),numel), SFr=SF/float(world);
    float SF8=host_global_scale_fp8(hin[0].data(),numel), SFr8=SF8/float(world);
    const size_t lp_elts_per_rank=numel/world, lp_rounds=(lp_elts_per_rank+LP_ELTS_PER_BLOCK-1)/LP_ELTS_PER_BLOCK;
    const size_t lp_buf_elts_per_rank=(size_t)LP_ELTS_PER_BLOCK_WS*lp_rounds;
    int lp_grid=(int)std::min<size_t>(std::max<size_t>(lp_rounds,(size_t)LP_MAX_BLOCKS*2),2048); if(const char* g=getenv("TRT_FP8_GRID")) lp_grid=atoi(g);
    const size_t lp_barrier_words=(size_t)(1+lp_grid)*world+8, barwords=2*(MAX_BLOCKS+1)*world+8;
    const size_t nv_slot_bytes=((shard/2+shard/SF_VEC_SIZE)+15)&~size_t(15), nv_barwords=2*(NV_MAX_BLOCKS+1)*world+8;

    // ---- per LOCAL rank: private in/out + shared buffers (fabric) ----
    std::vector<__nv_bfloat16*> d_in(LOCAL),d_out(LOCAL);
    std::vector<std::vector<ShBuf>> sh(LOCAL,std::vector<ShBuf>(NB));
    size_t need[NB]={L.total_bytes,L.total_bytes,FL.total_bytes,FL.total_bytes,(size_t)2*world*numel*2,(size_t)2*world*shard*2,barwords*4,barwords*4,MAX_RANKS*8,(size_t)world*lp_buf_elts_per_rank,lp_barrier_words*8,(size_t)2*world*nv_slot_bytes,nv_barwords*4,nv_barwords*4};
    for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; CUDA_CHECK(cudaSetDevice(lr));
        CUDA_CHECK(cudaMalloc(&d_in[lr],numel*2)); CUDA_CHECK(cudaMalloc(&d_out[lr],numel*2));
        std::vector<__nv_bfloat16> tmp(numel); for(size_t i=0;i<numel;i++) tmp[i]=__float2bfloat16(hin[r][i]);
        CUDA_CHECK(cudaMemcpy(d_in[lr],tmp.data(),numel*2,cudaMemcpyHostToDevice));
        for(int b=0;b<NB;b++){ sh[lr][b]=alloc_shared(lr,need[b]); CUDA_CHECK(cudaMemset((void*)sh[lr][b].va,0,sh[lr][b].size)); }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // ---- exchange fabric handles through the shared filesystem ----
    for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; RankHandles rh{}; for(int b=0;b<NB;b++){ rh.fh[b]=sh[lr][b].fh; rh.sz[b]=sh[lr][b].size; }
        std::string tmpn=dir+"/rank"+std::to_string(r)+".tmp", fin=dir+"/rank"+std::to_string(r)+".bin";
        { std::ofstream f(tmpn,std::ios::binary); f.write((char*)&rh,sizeof rh); } rename(tmpn.c_str(),fin.c_str()); }
    std::vector<RankHandles> all(world);
    for(int r=0;r<world;r++){ std::string fin=dir+"/rank"+std::to_string(r)+".bin";
        for(;;){ std::ifstream f(fin,std::ios::binary); if(f && f.read((char*)&all[r],sizeof(RankHandles))) break; std::this_thread::sleep_for(std::chrono::milliseconds(200)); } }
    // ---- resolve every rank's buffers to a VA usable from this process ----
    std::vector<std::vector<CUdeviceptr>> va(world,std::vector<CUdeviceptr>(NB));
    for(int r=0;r<world;r++) for(int b=0;b<NB;b++){
        if(r/LOCAL==PROC) va[r][b]=sh[r%LOCAL][b].va; else va[r][b]=map_remote(all[r].fh[b],all[r].sz[b]); }
    fs_barrier(dir,"mapped");   // everyone has zeroed its buffers and mapped everyone else's before any kernel runs
    if(PROC==0) printf("world=%d (%d procs x %d gpus)  numel=%zu  NVFP4 payload/rank=%zu B (BF16 %zu B, %.2fx)\n",world,NPROC,LOCAL,numel,L.total_bytes,numel*2,double(numel*2)/L.total_bytes);

    // ---- per-local-rank device arrays of world pointers ----
    auto mk_pp=[&](int lr,int b)->void*{ std::vector<void*> h(world); for(int r=0;r<world;r++) h[r]=(void*)va[r][b];
        void** d; CUDA_CHECK(cudaSetDevice(lr)); CUDA_CHECK(cudaMalloc(&d,world*sizeof(void*))); CUDA_CHECK(cudaMemcpy(d,h.data(),world*sizeof(void*),cudaMemcpyHostToDevice)); return (void*)d; };
    std::vector<uint8_t**> pp_pl(LOCAL),pp_red(LOCAL),pp_f8(LOCAL),pp_f8red(LOCAL);
    std::vector<__nv_bfloat16**> pp_c1(LOCAL),pp_c2(LOCAL); std::vector<uint32_t**> pp_ba(LOCAL),pp_bb(LOCAL);
    std::vector<uint64_t**> pp_ctr(LOCAL),pp_lpb(LOCAL); std::vector<__nv_fp8_e4m3**> pp_lpc(LOCAL);
    for(int lr=0;lr<LOCAL;lr++){ pp_pl[lr]=(uint8_t**)mk_pp(lr,B_PL); pp_red[lr]=(uint8_t**)mk_pp(lr,B_RED); pp_f8[lr]=(uint8_t**)mk_pp(lr,B_F8); pp_f8red[lr]=(uint8_t**)mk_pp(lr,B_F8RED);
        pp_c1[lr]=(__nv_bfloat16**)mk_pp(lr,B_C1); pp_c2[lr]=(__nv_bfloat16**)mk_pp(lr,B_C2); pp_ba[lr]=(uint32_t**)mk_pp(lr,B_BA); pp_bb[lr]=(uint32_t**)mk_pp(lr,B_BB);
        pp_ctr[lr]=(uint64_t**)mk_pp(lr,B_CTR); pp_lpc[lr]=(__nv_fp8_e4m3**)mk_pp(lr,B_LPC); pp_lpb[lr]=(uint64_t**)mk_pp(lr,B_LPB); }
    std::vector<uint8_t**> pp_nvc(LOCAL); std::vector<uint32_t**> pp_nvba(LOCAL),pp_nvbb(LOCAL);
    for(int lr=0;lr<LOCAL;lr++){ pp_nvc[lr]=(uint8_t**)mk_pp(lr,B_NVC); pp_nvba[lr]=(uint32_t**)mk_pp(lr,B_NVBA); pp_nvbb[lr]=(uint32_t**)mk_pp(lr,B_NVBB); }
    auto own=[&](int lr,int b){ return (void*)sh[lr][b].va; };

    const int TPB=256, PACKED=8; size_t nthr=numel/ELTS_PER_THREAD; int grid=int((nthr+TPB-1)/TPB); size_t sthr=shard/ELTS_PER_THREAD; int grid_s=int((sthr+TPB-1)/TPB);
    int grid_s32=int((shard/EPT32+TPB-1)/TPB), grid_ag32=int((numel/EPT32+TPB-1)/TPB);
    size_t nv_chunks=(shard+(size_t)TPB*EPT32-1)/((size_t)TPB*EPT32);
    // NVLink rule from the B200 sweep: one chunk per block up to 256 blocks, then 4 chunks/thread up to 2048; floor 16.
    // PCIe wants 4-8 blocks: set NV_GRID=8 (or NV_PCIE=1).
    int g_nv=(int)std::min<size_t>(std::max<size_t>(std::max<size_t>(std::min<size_t>(nv_chunks,256),nv_chunks/4),(size_t)16),(size_t)NV_MAX_BLOCKS);
    if(getenv("NV_PCIE")) g_nv=8;
    if(const char* ge=getenv("NV_GRID")) g_nv=atoi(ge);
    size_t epb_nv=((shard+(size_t)g_nv*TPB*EPT32-1)/((size_t)g_nv*TPB*EPT32))*((size_t)TPB*EPT32);
    auto trt_grid=[&](size_t elts){ size_t need2=(elts+TPB*PACKED-1)/(TPB*PACKED); int g=(int)std::min<size_t>(need2,MAX_BLOCKS); return g<1?1:g; };
    int g1=trt_grid(numel); size_t epb1=((numel+(size_t)g1*TPB*PACKED-1)/((size_t)g1*TPB*PACKED))*(TPB*PACKED);
    int g2=trt_grid(shard); size_t epb2=((shard+(size_t)g2*TPB*PACKED-1)/((size_t)g2*TPB*PACKED))*(TPB*PACKED);
    auto sync_local=[&](){ for(int lr=0;lr<LOCAL;lr++){ CUDA_CHECK(cudaSetDevice(lr)); CUDA_CHECK(cudaDeviceSynchronize()); } };
    const bool big=numel>=(size_t(1)<<27); const int WARMUP=big?5:20, ITERS=big?20:100;
    auto rel_rmse=[&](std::vector<__nv_bfloat16>&h){ double s=0,sr=0; for(size_t i=0;i<numel;i++){ float o=__bfloat162float(h[i]); double d=o-ref[i]; s+=d*d; sr+=double(ref[i])*ref[i]; } return sqrt(s/(sr+1e-12)); };
    auto check=[&](const char* name){ if(PROC!=0) return; std::vector<__nv_bfloat16> h(numel); CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h.data(),d_out[0],numel*2,cudaMemcpyDeviceToHost)); printf("  %-18s rel_rmse=%.6f\n",name,rel_rmse(h)); };

    uint32_t trtflag=1, nvfflag=1; uint64_t nvflag=1, lpflag=1;
    auto run_nvfp4_fused=[&](){ for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr);
        nvfp4_twoshot_fused<NRANKS><<<g_nv,TPB>>>(d_in[lr],d_out[lr],pp_nvc[lr],pp_nvba[lr],pp_nvbb[lr],r,shard,nv_slot_bytes,epb_nv,SF,SFr,nvfflag); } nvfflag++; }; const size_t lp_smem=(size_t)LP_WARPS*world*sizeof(float)*2;
    auto run_trt_oneshot=[&](){ for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr);
        trt_oneshot_push<NRANKS><<<g1,TPB>>>(d_in[lr],d_out[lr],pp_c1[lr],pp_ba[lr],r,numel,epb1,trtflag); } trtflag++; };
    auto run_trt_twoshot=[&](){ for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr);
        trt_twoshot_push<NRANKS><<<g2,TPB>>>(d_in[lr],d_out[lr],pp_c2[lr],pp_ba[lr],pp_bb[lr],r,numel,shard,epb2,trtflag); } trtflag++; };
    auto run_nvfp4_oneshot=[&](){
        for(int lr=0;lr<LOCAL;lr++){ cudaSetDevice(lr); nvfp4_quantize<<<grid,TPB>>>(d_in[lr],(uint8_t*)own(lr,B_PL),L,SF); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[lr],r,world,nvflag); }
        for(int lr=0;lr<LOCAL;lr++){ cudaSetDevice(lr); nvfp4_oneshot<<<grid,TPB>>>(pp_pl[lr],d_out[lr],L,world); } nvflag++; };
    auto run_nvfp4_twoshot=[&](){
        for(int lr=0;lr<LOCAL;lr++){ cudaSetDevice(lr); nvfp4_quantize<<<grid,TPB>>>(d_in[lr],(uint8_t*)own(lr,B_PL),L,SF); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[lr],r,world,nvflag); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); nvfp4_reducescatter<<<grid_s32,TPB>>>(pp_pl[lr],(uint8_t*)own(lr,B_RED),L,world,r,shard,SFr); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[lr],r,world,nvflag+1); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); nvfp4_allgather<<<grid_ag32,TPB>>>(pp_red[lr],d_out[lr],L,world,shard,r); } nvflag+=2; };
    auto run_fp8_oneshot=[&](){
        for(int lr=0;lr<LOCAL;lr++){ cudaSetDevice(lr); fp8_quantize<<<grid,TPB>>>(d_in[lr],(uint8_t*)own(lr,B_F8),FL,SF8); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[lr],r,world,nvflag); }
        for(int lr=0;lr<LOCAL;lr++){ cudaSetDevice(lr); fp8_oneshot<<<grid,TPB>>>(pp_f8[lr],d_out[lr],FL,world); } nvflag++; };
    auto run_fp8_twoshot=[&](){
        for(int lr=0;lr<LOCAL;lr++){ cudaSetDevice(lr); fp8_quantize<<<grid,TPB>>>(d_in[lr],(uint8_t*)own(lr,B_F8),FL,SF8); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[lr],r,world,nvflag); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); fp8_reducescatter<<<grid_s,TPB>>>(pp_f8[lr],(uint8_t*)own(lr,B_F8RED),FL,world,r,shard,SFr8); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[lr],r,world,nvflag+1); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); fp8_allgather<<<grid,TPB>>>(pp_f8red[lr],d_out[lr],FL,world,shard,r); } nvflag+=2; };
    auto run_trt_fp8=[&](){
        for(int lr=0;lr<LOCAL;lr++){ cudaSetDevice(lr); trt_fp8_preprocess<NRANKS><<<(unsigned)(lp_rounds*world),LP_BLOCK>>>(d_in[lr],lp_elts_per_rank,lp_buf_elts_per_rank,(__nv_fp8_e4m3*)own(lr,B_LPC)); }
        for(int lr=0;lr<LOCAL;lr++){ int r=PROC*LOCAL+lr; cudaSetDevice(lr); trt_fp8_twoshot<NRANKS><<<lp_grid,LP_BLOCK,lp_smem>>>(pp_lpc[lr],pp_lpb[lr],d_out[lr],r,lp_elts_per_rank,lp_buf_elts_per_rank,lpflag); } lpflag++; };

    if(PROC==0) printf("correctness (rank 0):\n");
    run_trt_oneshot();  sync_local(); check("TRT-BF16-oneshot");
    run_trt_twoshot();  sync_local(); check("TRT-BF16-twoshot");
    run_fp8_oneshot();  sync_local(); check("FP8-oneshot");
    run_fp8_twoshot();  sync_local(); check("FP8-twoshot");
    run_trt_fp8();      sync_local(); check("TRT-FP8-twoshot");
    run_nvfp4_oneshot(); sync_local(); check("NVFP4-oneshot");
    run_nvfp4_twoshot(); sync_local(); check("NVFP4-twoshot");
    run_nvfp4_fused();   sync_local(); check("NVFP4-fused");

    auto time_it=[&](auto fn)->double{ for(int it=0;it<WARMUP;it++) fn(); sync_local();
        cudaSetDevice(0); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
        for(int it=0;it<ITERS;it++) fn(); cudaSetDevice(0); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b); return ms*1e3/ITERS; };
    double t_trt1=time_it(run_trt_oneshot), t_f81=time_it(run_fp8_oneshot), t_nv1=time_it(run_nvfp4_oneshot);
    double t_trt2=time_it(run_trt_twoshot), t_f82=time_it(run_fp8_twoshot), t_trtf8=time_it(run_trt_fp8), t_nv2=time_it(run_nvfp4_twoshot), t_nvf=time_it(run_nvfp4_fused);
    sync_local(); fs_barrier(dir,"done");
    if(PROC==0){
        printf("PAYLOAD bytes/elt  BF16=2.000  FP8=%.4f (%.2fx)  NVFP4=%.4f (%.2fx)\n",double(FL.total_bytes)/numel,numel*2.0/FL.total_bytes,double(L.total_bytes)/numel,numel*2.0/L.total_bytes);
        printf("TIMING (us/allreduce, %d ranks over %d nodes):\n",world,NPROC);
        printf("  ONE-SHOT  BF16=%.2f  FP8=%.2f  NVFP4=%.2f  | NVFP4/BF16=%.2fx  NVFP4/FP8=%.2fx\n",t_trt1,t_f81,t_nv1,t_trt1/t_nv1,t_f81/t_nv1);
        printf("  (TRT-FP8 grid=%d blocks x %d thr)\n",lp_grid,LP_BLOCK);
        printf("  TWO-SHOT  BF16=%.2f  myFP8=%.2f  TRT-FP8=%.2f  NVFP4=%.2f  | NVFP4/BF16=%.2fx  NVFP4/myFP8=%.2fx  NVFP4/TRT-FP8=%.2fx  myFP8/TRT-FP8=%.2fx\n",t_trt2,t_f82,t_trtf8,t_nv2,t_trt2/t_nv2,t_f82/t_nv2,t_trtf8/t_nv2,t_f82/t_trtf8);
        printf("  FUSED     NVFP4-fused=%.2f (grid=%d x %d thr)  | fused/BF16=%.2fx  fused/TRT-FP8=%.2fx  fused vs 5-launch=%.2fx\n",t_nvf,g_nv,TPB,t_trt2/t_nvf,t_trtf8/t_nvf,t_nv2/t_nvf);
    } else {
        printf("[p%d] TIMING  one: BF16=%.2f FP8=%.2f NVFP4=%.2f | two: BF16=%.2f myFP8=%.2f TRT-FP8=%.2f NVFP4=%.2f fused=%.2f\n",PROC,t_trt1,t_f81,t_nv1,t_trt2,t_f82,t_trtf8,t_nv2,t_nvf);
    }
    return 0;
}
