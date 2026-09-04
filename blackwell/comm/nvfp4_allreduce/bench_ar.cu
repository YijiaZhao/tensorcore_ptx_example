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
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

static constexpr int MAX_RANKS = 8;
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

__global__ void nvfp4_quantize(const __nv_bfloat16* __restrict__ in, uint8_t* __restrict__ ob, Nvfp4Layout L, float SF){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/ELTS_PER_THREAD) return;
    const size_t base=tid*ELTS_PER_THREAD; float2 fp2[4]; float mx=0;
#pragma unroll
    for(int i=0;i<4;i++){ __nv_bfloat162 v=*reinterpret_cast<const __nv_bfloat162*>(&in[base+i*2]); fp2[i]=__bfloat1622float2(v); mx=fmaxf(mx,fmaxf(fabsf(fp2[i].x),fabsf(fp2[i].y))); }
    mx=fmaxf(__shfl_xor_sync(0xffffffffu,mx,1),mx);
    float SFv=SF*(mx*recip_ftz(E2M1_MAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0; uint32_t e=0;
#pragma unroll
    for(int i=0;i<4;i++){ e|=uint32_t(float_to_e2m1(fp2[i].x*os))<<((2*i)*4); e|=uint32_t(float_to_e2m1(fp2[i].y*os))<<((2*i+1)*4); }
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
        for(int i=0;i<8;i++) acc[i]+=e2m1_to_float((e>>(i*4))&0xF)*bs; }
#pragma unroll
    for(int i=0;i<8;i++) out[base+i]=__float2bfloat16(acc[i]);
}
__global__ void nvfp4_reducescatter(uint8_t** peer, uint8_t* my, Nvfp4Layout L, int world, int rank, size_t shard, float SF){
    const size_t lt=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(lt>=shard/ELTS_PER_THREAD) return;
    const size_t g=size_t(rank)*shard+lt*ELTS_PER_THREAD, po=g/2, si=g/SF_VEC_SIZE; float acc[8];
#pragma unroll
    for(int i=0;i<8;i++) acc[i]=0;
    for(int r=0;r<world;r++){ uint8_t* b=peer[r]; uint32_t e=*reinterpret_cast<const uint32_t*>(&L.packed(b)[po]);
        float bs=ue4m3_to_float(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
        for(int i=0;i<8;i++) acc[i]+=e2m1_to_float((e>>(i*4))&0xF)*bs; }
    float mx=0;
#pragma unroll
    for(int i=0;i<8;i++) mx=fmaxf(mx,fabsf(acc[i]));
    mx=fmaxf(__shfl_xor_sync(0xffffffffu,mx,1),mx);
    float SFv=SF*(mx*recip_ftz(E2M1_MAX)); __nv_fp8_e4m3 sf=__nv_fp8_e4m3(SFv); float SFq=(float)sf;
    float os=(mx!=0)?recip_ftz(SFq*recip_ftz(SF)):0; uint32_t e=0;
#pragma unroll
    for(int i=0;i<8;i++) e|=uint32_t(float_to_e2m1(acc[i]*os))<<(i*4);
    *reinterpret_cast<uint32_t*>(&L.packed(my)[po])=e;
    if((lt&1)==0) L.scales(my)[si]=sf.__x;
    if(lt==0) *L.gscale(my)=SF;
}
__global__ void nvfp4_allgather(uint8_t** peer, __nv_bfloat16* __restrict__ out, Nvfp4Layout L, int world, size_t shard){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/ELTS_PER_THREAD) return;
    const size_t g=tid*ELTS_PER_THREAD; const int owner=int(g/shard); uint8_t* b=peer[owner];
    const size_t po=g/2, si=g/SF_VEC_SIZE; uint32_t e=*reinterpret_cast<const uint32_t*>(&L.packed(b)[po]);
    float bs=ue4m3_to_float(L.scales(b)[si])*recip_ftz(*L.gscale(b));
#pragma unroll
    for(int i=0;i<8;i++) out[g+i]=__float2bfloat16(e2m1_to_float((e>>(i*4))&0xF)*bs);
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

static float host_global_scale(const float* h, size_t n){ float a=0; for(size_t i=0;i<n;i++) a=fmaxf(a,fabsf(h[i])); if(a==0) return 1.f; return (E2M1_MAX*448.f)/a; }

int main(int argc,char**argv){
    int world=8; size_t numel=(argc>1)?strtoull(argv[1],0,10):(1u<<20);
    int ndev=0; CUDA_CHECK(cudaGetDeviceCount(&ndev)); if(ndev<world) world=ndev;
    if(world!=8){ printf("need 8 GPUs (RANKS templated=8), have %d\n",world); return 1; }
    if(numel%(size_t(world)*SF_VEC_SIZE)!=0){ printf("numel must be %d-aligned\n",world*SF_VEC_SIZE); return 1; }
    const size_t shard=numel/world;

    for(int i=0;i<world;i++){ CUDA_CHECK(cudaSetDevice(i));
        for(int j=0;j<world;j++) if(i!=j){ int can=0; cudaDeviceCanAccessPeer(&can,i,j);
            if(can){ cudaError_t e=cudaDeviceEnablePeerAccess(j,0); if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);} } }

    Nvfp4Layout L=Nvfp4Layout::make(numel);
    std::vector<std::vector<float>> hin(world,std::vector<float>(numel)); std::vector<float> ref(numel,0);
    srand(1234);
    for(int r=0;r<world;r++) for(size_t i=0;i<numel;i++){ float v=(rand()/float(RAND_MAX))*2-1; hin[r][i]=v; ref[i]+=v; }
    float SF=host_global_scale(hin[0].data(),numel), SFr=SF/float(world);

    // per-rank buffers
    std::vector<__nv_bfloat16*> d_in(world),d_out(world);
    std::vector<uint8_t*> d_pl(world),d_red(world);
    std::vector<__nv_bfloat16*> d_comm1(world),d_comm2(world);      // TRT one/two-shot comm buffers
    std::vector<uint32_t*> d_bar_a(world),d_bar_b(world);           // TRT fused-barrier signals (in/out)
    std::vector<uint64_t*> d_ctr(world);                           // NVFP4 counter barrier
    const size_t barwords=2*(MAX_BLOCKS+1)*world + 8;
    for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_in[r],numel*2)); CUDA_CHECK(cudaMalloc(&d_out[r],numel*2));
        CUDA_CHECK(cudaMalloc(&d_pl[r],L.total_bytes)); CUDA_CHECK(cudaMalloc(&d_red[r],L.total_bytes)); CUDA_CHECK(cudaMemset(d_red[r],0,L.total_bytes));
        CUDA_CHECK(cudaMalloc(&d_comm1[r],(size_t)2*world*numel*2)); // [2*world][elts_total]
        CUDA_CHECK(cudaMalloc(&d_comm2[r],(size_t)2*world*shard*2)); // [2*world][elts_per_rank]
        CUDA_CHECK(cudaMalloc(&d_bar_a[r],barwords*4)); CUDA_CHECK(cudaMemset(d_bar_a[r],0,barwords*4));
        CUDA_CHECK(cudaMalloc(&d_bar_b[r],barwords*4)); CUDA_CHECK(cudaMemset(d_bar_b[r],0,barwords*4));
        CUDA_CHECK(cudaMalloc(&d_ctr[r],MAX_RANKS*8)); CUDA_CHECK(cudaMemset(d_ctr[r],0,MAX_RANKS*8));
        std::vector<__nv_bfloat16> tmp(numel); for(size_t i=0;i<numel;i++) tmp[i]=__float2bfloat16(hin[r][i]);
        CUDA_CHECK(cudaMemcpy(d_in[r],tmp.data(),numel*2,cudaMemcpyHostToDevice));
    }
    // peer pointer arrays (device)
    std::vector<uint8_t**> pp_pl(world),pp_red(world);
    std::vector<__nv_bfloat16**> pp_c1(world),pp_c2(world);
    std::vector<uint32_t**> pp_ba(world),pp_bb(world);
    std::vector<uint64_t**> pp_ctr(world);
    for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&pp_pl[r],world*8));  CUDA_CHECK(cudaMemcpy(pp_pl[r],d_pl.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_red[r],world*8)); CUDA_CHECK(cudaMemcpy(pp_red[r],d_red.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_c1[r],world*8));  CUDA_CHECK(cudaMemcpy(pp_c1[r],d_comm1.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_c2[r],world*8));  CUDA_CHECK(cudaMemcpy(pp_c2[r],d_comm2.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_ba[r],world*8));  CUDA_CHECK(cudaMemcpy(pp_ba[r],d_bar_a.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_bb[r],world*8));  CUDA_CHECK(cudaMemcpy(pp_bb[r],d_bar_b.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_ctr[r],world*8)); CUDA_CHECK(cudaMemcpy(pp_ctr[r],d_ctr.data(),world*8,cudaMemcpyHostToDevice));
    }

    const int TPB=256; const int PACKED=8;
    size_t nthr=numel/ELTS_PER_THREAD; int grid=int((nthr+TPB-1)/TPB);
    size_t sthr=shard/ELTS_PER_THREAD; int grid_s=int((sthr+TPB-1)/TPB);
    // TRT grids: cap at MAX_BLOCKS, elts_per_block multiple of TPB*PACKED
    auto trt_grid=[&](size_t elts){ size_t need=(elts+TPB*PACKED-1)/(TPB*PACKED); int g=(int)std::min<size_t>(need,MAX_BLOCKS); return g<1?1:g; };
    int g1=trt_grid(numel);  size_t epb1=((numel+ (size_t)g1*TPB*PACKED-1)/((size_t)g1*TPB*PACKED))*(TPB*PACKED);
    int g2=trt_grid(shard);  size_t epb2=((shard+ (size_t)g2*TPB*PACKED-1)/((size_t)g2*TPB*PACKED))*(TPB*PACKED);

    auto sync_all=[&](){ for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); CUDA_CHECK(cudaDeviceSynchronize()); } };
    const int WARMUP=20,ITERS=100;
    auto rel_rmse=[&](std::vector<__nv_bfloat16>&h){ double s=0,sr=0; for(size_t i=0;i<numel;i++){ float o=__bfloat162float(h[i]); double e=o-ref[i]; s+=e*e; sr+=double(ref[i])*ref[i]; } return sqrt(s/(sr+1e-12)); };

    auto check=[&](const char*name){ std::vector<__nv_bfloat16> h(numel); CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h.data(),d_out[0],numel*2,cudaMemcpyDeviceToHost));
        printf("  %-18s rel_rmse=%.6f\n",name,rel_rmse(h)); };

    printf("world=%d numel=%zu  NVFP4 payload/rank=%zu B (BF16 %zu B, %.2fx smaller)\n",
           world,numel,L.total_bytes,numel*2,double(numel*2)/L.total_bytes);

    // ---------- lambdas for one allreduce pass ----------
    // single monotonic flag shared by all TRT calls that touch pp_ba (stale flags are
    // always strictly lower, so block_barrier never false-passes on a repeated value).
    uint32_t trtflag=1;
    auto run_trt_oneshot=[&](){
        for(int r=0;r<world;r++){ cudaSetDevice(r);
            trt_oneshot_push<8><<<g1,TPB>>>(d_in[r],d_out[r],pp_c1[r],pp_ba[r],r,numel,epb1,trtflag); }
        trtflag++;
    };
    auto run_trt_twoshot=[&](){
        for(int r=0;r<world;r++){ cudaSetDevice(r);
            trt_twoshot_push<8><<<g2,TPB>>>(d_in[r],d_out[r],pp_c2[r],pp_ba[r],pp_bb[r],r,numel,shard,epb2,trtflag); }
        trtflag++;
    };
    uint64_t nvflag=1;
    auto run_nvfp4_oneshot=[&](){
        for(int r=0;r<world;r++){ cudaSetDevice(r); nvfp4_quantize<<<grid,TPB>>>(d_in[r],d_pl[r],L,SF); }
        for(int r=0;r<world;r++){ cudaSetDevice(r); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[r],r,world,nvflag); }
        for(int r=0;r<world;r++){ cudaSetDevice(r); nvfp4_oneshot<<<grid,TPB>>>(pp_pl[r],d_out[r],L,world); }
        nvflag++;
    };
    auto run_nvfp4_twoshot=[&](){
        for(int r=0;r<world;r++){ cudaSetDevice(r); nvfp4_quantize<<<grid,TPB>>>(d_in[r],d_pl[r],L,SF); }
        for(int r=0;r<world;r++){ cudaSetDevice(r); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[r],r,world,nvflag); }
        for(int r=0;r<world;r++){ cudaSetDevice(r); nvfp4_reducescatter<<<grid_s,TPB>>>(pp_pl[r],d_red[r],L,world,r,shard,SFr); }
        for(int r=0;r<world;r++){ cudaSetDevice(r); ctr_barrier<<<1,MAX_RANKS>>>(pp_ctr[r],r,world,nvflag+1); }
        for(int r=0;r<world;r++){ cudaSetDevice(r); nvfp4_allgather<<<grid,TPB>>>(pp_red[r],d_out[r],L,world,shard); }
        nvflag+=2;
    };

    // ---------- correctness ----------
    printf("correctness:\n");
    run_trt_oneshot();  sync_all(); check("TRT-oneshot");
    run_trt_twoshot();  sync_all(); check("TRT-twoshot");
    run_nvfp4_oneshot(); sync_all(); check("NVFP4-oneshot");
    run_nvfp4_twoshot(); sync_all(); check("NVFP4-twoshot");

    // ---------- timing ----------
    auto bench=[&](void(*)(),  const char*){}; (void)bench;
    auto time_it=[&](auto fn)->double{
        for(int it=0;it<WARMUP;it++) fn(); sync_all();
        cudaSetDevice(0); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a); for(int it=0;it<ITERS;it++) fn(); cudaSetDevice(0); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b); return ms*1e3/ITERS;
    };
    double t_trt1=time_it(run_trt_oneshot);
    double t_nv1 =time_it(run_nvfp4_oneshot);
    double t_trt2=time_it(run_trt_twoshot);
    double t_nv2 =time_it(run_nvfp4_twoshot);

    printf("TIMING (us/allreduce):\n");
    printf("  ONE-SHOT  TRT-LLM-BF16=%.2f  NVFP4=%.2f  | NVFP4 %.2fx %s\n",
           t_trt1,t_nv1, t_trt1/t_nv1, (t_nv1<t_trt1?"FASTER":"slower"));
    printf("  TWO-SHOT  TRT-LLM-BF16=%.2f  NVFP4=%.2f  | NVFP4 %.2fx %s\n",
           t_trt2,t_nv2, t_trt2/t_nv2, (t_nv2<t_trt2?"FASTER":"slower"));
    return 0;
}
