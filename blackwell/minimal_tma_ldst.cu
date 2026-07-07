/*
 * 最小 TMA load/store 单元(单卡,无 P2P)
 * 数据路径:global --cp.async.bulk--> smem --cp.async.bulk--> global
 *   ① tma_load_1d  : global → smem,异步,mbarrier 追踪完成
 *   ② tma_store_1d : smem → global,bulk_group,wait_group 等完成
 * 这是 TMA 1D bulk 的最小骨架,不需要 tensor-map 描述符,只用裸指针 + mbarrier。
 *
 * 编译(B200): nvcc -arch=sm_100a -O3 -o minimal_tma_ldst minimal_tma_ldst.cu
 * (Hopper:    nvcc -arch=sm_90a  ...)
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do{ cudaError_t e=(x); if(e){ \
  printf("%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- TMA / mbarrier PTX(与 deep_gemm/ptx/tma.cuh 一致)----
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c){
    uint32_t a=__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(a),"r"(c));
}
__device__ __forceinline__ void tma_load_1d(void* dst_smem,const void* src_g,uint64_t* mbar,uint32_t n){
    uint32_t d=__cvta_generic_to_shared(dst_smem), m=__cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0],[%1],%2,[%3];"::"r"(d),"l"(src_g),"r"(n),"r"(m):"memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* b,uint32_t n){
    uint32_t a=__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%1],%0;"::"r"(n),"r"(a));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b,uint32_t ph){
    uint32_t a=__cvta_generic_to_shared(b);
    asm volatile("{.reg .pred P; L: mbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1,%2;"
                 "@P bra D; bra L; D:}"::"r"(a),"r"(ph),"r"(0x989680));
}
__device__ __forceinline__ void tma_store_1d(void* dst_g,const void* src_smem,uint32_t n){
    uint32_t s=__cvta_generic_to_shared(src_smem);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0],[%1],%2;"
                 ::"l"(dst_g),"r"(s),"r"(n):"memory");
}
__device__ __forceinline__ void tma_store_commit(){ asm volatile("cp.async.bulk.commit_group;":::"memory"); }
__device__ __forceinline__ void tma_store_wait(){ asm volatile("cp.async.bulk.wait_group 0;":::"memory"); }

// 一个 CTA,把 nbytes 按 CHUNK 分块 global→smem→global 搬运
template<uint32_t CHUNK>
__global__ void tma_copy(const uint8_t* src, uint8_t* dst, uint32_t nbytes){
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t mbar;
    if(threadIdx.x==0) mbar_init(&mbar,1);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;":::"memory");

    uint32_t phase=0;
    for(uint32_t off=0; off<nbytes; off+=CHUNK){
        uint32_t b = (nbytes-off<CHUNK)?(nbytes-off):CHUNK;
        if(threadIdx.x==0){
            tma_load_1d(smem, src+off, &mbar, b);   // ① global → smem
            mbar_expect(&mbar, b);
            mbar_wait(&mbar, phase);                 // 等 load 完成
            tma_store_1d(dst+off, smem, b);          // ② smem → global
            tma_store_commit();
            tma_store_wait();
        }
        __syncthreads();
        phase^=1;
    }
}

int main(){
    constexpr uint32_t CHUNK=16*1024, N=1u<<20, BYTES=N*sizeof(float);
    float *ds,*dd; CHECK(cudaMalloc(&ds,BYTES)); CHECK(cudaMalloc(&dd,BYTES));
    CHECK(cudaMemset(dd,0,BYTES));
    float* h=(float*)malloc(BYTES);
    for(uint32_t i=0;i<N;++i) h[i]=(float)(i%1000)+0.5f;
    CHECK(cudaMemcpy(ds,h,BYTES,cudaMemcpyHostToDevice));

    cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    tma_copy<CHUNK><<<1,32,CHUNK>>>((const uint8_t*)ds,(uint8_t*)dd,BYTES);
    CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1)); CHECK(cudaGetLastError());
    float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));

    float* o=(float*)malloc(BYTES); CHECK(cudaMemcpy(o,dd,BYTES,cudaMemcpyDeviceToHost));
    uint32_t err=0; for(uint32_t i=0;i<N;++i) if(o[i]!=h[i]){ if(err<5) printf("  @%u got %f want %f\n",i,o[i],h[i]); ++err; }
    printf("[TMA global->smem->global] %u bytes, chunk=%u : %s (%.3f ms, %.1f GB/s)\n",
           BYTES,CHUNK,err?"FAIL":"OK",ms,BYTES/(ms*1e6));
    free(h); free(o); cudaFree(ds); cudaFree(dd);
    return err?1:0;
}
