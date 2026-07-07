// Test tcgen05.mma with different warp counts
// cta_group::1 = expected 4 warps (128 threads)
// cta_group::2 = expected 8 warps (256 threads)
// Also test with fewer/more warps to find limits

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ uint32_t su32(void const* p){return static_cast<uint32_t>(__cvta_generic_to_shared(p));}

__device__ uint64_t mkdesc(void const* p, int stride){
    uint64_t d=0;
    d|=(uint64_t)((su32(p)>>4)&0x3FFF);
    d|=(uint64_t)(((stride>>4)&0x3FFF))<<16;
    d|=(uint64_t)(1)<<46;
    return d;
}

// cta_group::1 kernel
template<int NTHREADS>
__global__ void test_cg1(float* out, int* status) {
    extern __shared__ char sm[];
    uint8_t* sa=(uint8_t*)sm;
    uint8_t* sb=sa+4096;
    uint32_t* stm=(uint32_t*)(sb+256+64);
    int tid=threadIdx.x, wid=tid/32;

    for(int i=tid;i<4096;i+=NTHREADS) sa[i]=0x22;
    for(int i=tid;i<256;i+=NTHREADS) sb[i]=0x22;
    __syncthreads();

    if(wid==0) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"
                            ::"r"(su32(stm)),"r"(32));
    __syncthreads();
    uint32_t tb=*stm, tc=tb, tsfa=tb+8, tsfb=tb+12;

    uint32_t sv=0x38383838u;
    if(wid==0){
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0],{%1,%2,%3,%4};\n"
                     ::"r"(tsfa),"r"(sv),"r"(sv),"r"(sv),"r"(sv));
    }
    __syncthreads();
    if(wid==0){
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0],{%1,%2,%3,%4};\n"
                     ::"r"(tsfb),"r"(sv),"r"(sv),"r"(sv),"r"(sv));
    }
    __syncthreads();

    uint64_t da=mkdesc(sa,32), db=mkdesc(sb,32);
    uint32_t id=(1<<7)|(1<<10)|(1<<17)|(8<<24);
    id|=((tsfa>>30)&3)<<29;
    id|=((tsfb>>30)&3)<<4;
    __syncthreads();

    if(tid==0){
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p,%4,0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0],%1,%2,%3,[%5],[%6],p;\n\t}\n"
            ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(0u),"r"(tsfa),"r"(tsfb));
    }
    __syncthreads();

    if(wid==0){
        uint32_t r0,r1,r2,r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3},[%4];\n"
                     :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(tc));
        if(tid==0){
            out[0]=__uint_as_float(r0);
            *status = (out[0]==64.0f) ? 1 : 0;
        }
    }

    __syncthreads();
    if(wid==0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"
                            ::"r"(tb),"r"(32));
}

// cta_group::2 kernel
template<int NTHREADS>
__global__ void test_cg2(float* out, int* status) {
    extern __shared__ char sm[];
    uint8_t* sa=(uint8_t*)sm;
    uint8_t* sb=sa+4096;
    uint32_t* stm=(uint32_t*)(sb+256+64);
    int tid=threadIdx.x, wid=tid/32;

    for(int i=tid;i<4096;i+=NTHREADS) sa[i]=0x22;
    for(int i=tid;i<256;i+=NTHREADS) sb[i]=0x22;
    __syncthreads();

    if(wid==0) asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;"
                            ::"r"(su32(stm)),"r"(32));
    __syncthreads();
    uint32_t tb=*stm, tc=tb, tsfa=tb+8, tsfb=tb+12;

    uint32_t sv=0x38383838u;
    if(wid==0){
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0],{%1,%2,%3,%4};\n"
                     ::"r"(tsfa),"r"(sv),"r"(sv),"r"(sv),"r"(sv));
    }
    __syncthreads();
    if(wid==0){
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0],{%1,%2,%3,%4};\n"
                     ::"r"(tsfb),"r"(sv),"r"(sv),"r"(sv),"r"(sv));
    }
    __syncthreads();

    uint64_t da=mkdesc(sa,32), db=mkdesc(sb,32);
    // cta_group::2 supports M=128 or 256. Use M=128 (m_dim=8)
    uint32_t id=(1<<7)|(1<<10)|(1<<17)|(8<<24);
    id|=((tsfa>>30)&3)<<29;
    id|=((tsfb>>30)&3)<<4;
    __syncthreads();

    if(tid==0){
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p,%4,0;\n\t"
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0],%1,%2,%3,[%5],[%6],p;\n\t}\n"
            ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(0u),"r"(tsfa),"r"(tsfb));
    }
    __syncthreads();

    if(wid==0){
        uint32_t r0,r1,r2,r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3},[%4];\n"
                     :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(tc));
        if(tid==0){
            out[0]=__uint_as_float(r0);
            *status = (out[0]==64.0f) ? 1 : 0;
        }
    }

    __syncthreads();
    if(wid==0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;"
                            ::"r"(tb),"r"(32));
}

int main(){
    float *d; int *st, h_st;
    cudaMalloc(&d,4096);
    cudaMalloc(&st,sizeof(int));

    printf("=== cta_group::1 (expect 4 warps = 128 threads) ===\n");
    int warps_cg1[] = {1,2,3,4,5,6,7,8};
    for(int w : warps_cg1){
        int threads = w * 32;
        cudaMemset(d,0,4096);
        cudaMemset(st,0,sizeof(int));

        // Can't template at runtime, use switch
        cudaError_t e;
        switch(threads){
            case 32:  test_cg1<32><<<1,32,8192>>>(d,st); break;
            case 64:  test_cg1<64><<<1,64,8192>>>(d,st); break;
            case 96:  test_cg1<96><<<1,96,8192>>>(d,st); break;
            case 128: test_cg1<128><<<1,128,8192>>>(d,st); break;
            case 160: test_cg1<160><<<1,160,8192>>>(d,st); break;
            case 192: test_cg1<192><<<1,192,8192>>>(d,st); break;
            case 224: test_cg1<224><<<1,224,8192>>>(d,st); break;
            case 256: test_cg1<256><<<1,256,8192>>>(d,st); break;
        }
        e = cudaDeviceSynchronize();
        if(e!=cudaSuccess){
            printf("  %d warps (%3d threads): FAIL (%s)\n", w, threads, cudaGetErrorString(e));
            cudaGetLastError(); // clear error
        } else {
            cudaMemcpy(&h_st,st,sizeof(int),cudaMemcpyDeviceToHost);
            float h; cudaMemcpy(&h,d,sizeof(float),cudaMemcpyDeviceToHost);
            printf("  %d warps (%3d threads): %s (D[0]=%.1f)\n", w, threads,
                   h_st?"PASS":"WRONG", h);
        }
    }

    printf("\n=== cta_group::2 (expect 8 warps = 256 threads) ===\n");
    int warps_cg2[] = {4,6,8,10,12};
    for(int w : warps_cg2){
        int threads = w * 32;
        cudaMemset(d,0,4096);
        cudaMemset(st,0,sizeof(int));

        cudaError_t e;
        switch(threads){
            case 128: test_cg2<128><<<1,128,8192>>>(d,st); break;
            case 192: test_cg2<192><<<1,192,8192>>>(d,st); break;
            case 256: test_cg2<256><<<1,256,8192>>>(d,st); break;
            case 320: test_cg2<320><<<1,320,8192>>>(d,st); break;
            case 384: test_cg2<384><<<1,384,8192>>>(d,st); break;
        }
        e = cudaDeviceSynchronize();
        if(e!=cudaSuccess){
            printf("  %d warps (%3d threads): FAIL (%s)\n", w, threads, cudaGetErrorString(e));
            cudaGetLastError();
        } else {
            cudaMemcpy(&h_st,st,sizeof(int),cudaMemcpyDeviceToHost);
            float h; cudaMemcpy(&h,d,sizeof(float),cudaMemcpyDeviceToHost);
            printf("  %d warps (%3d threads): %s (D[0]=%.1f)\n", w, threads,
                   h_st?"PASS":"WRONG", h);
        }
    }

    cudaFree(d); cudaFree(st);
}
