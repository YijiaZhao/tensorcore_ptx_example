// bench_ar_pcie.cu — NVFP4 two-shot all-reduce on PCIe hosts with three wire transports:
//   sm   : in-kernel peer stores (the fused kernel from nvfp4_allreduce.cuh; SM threads write across PCIe)
//   ce   : copy engines — cudaMemcpyPeerAsync for every slice, stream-ordered with cross-device events
//   rdma : GPUDirect RDMA loopback through the host's own NICs — each GPU uses the NIC on its PCIe bridge,
//          RC QPs between ranks, RDMA_WRITE (+_WITH_IMM for the last slice) into the peer's landing buffer,
//          host polls completions between phases (structure follows FlashInfer PR #4876 / Ulysses PCIe).
// The quantize / reduce+requantize / dequantize math is identical for all three; only the bytes-on-the-wire
// mechanism changes. Single process, one thread, NRANKS GPUs.
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -DNRANKS=8 bench_ar_pcie.cu -o bench_ar_pcie -lcuda -libverbs
// Run:   ./bench_ar_pcie <numel> [AR_TRANSPORT=sm,ce,rdma] [NV_GRID=1] [RDMA_GID=3] [NIC_MAP=2,3,0,1,6,7,4,5]
#include "nvfp4_allreduce.cuh"
#include <cuda.h>
#include <infiniband/verbs.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <limits.h>

#ifndef NRANKS
#define NRANKS 8
#endif
using namespace nvfp4_ar;

#define CUDA_CHECK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ fprintf(stderr,"CUDA %s @%s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
#define CU_CHECK(x) do{ CUresult r_=(x); if(r_!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r_,&s); fprintf(stderr,"CU %s @%d: %s\n",#x,__LINE__,s); exit(1);} }while(0)
#define IBV_CHECK(cond,msg) do{ if(!(cond)){ fprintf(stderr,"ibv: %s failed (%s) @%d\n",msg,strerror(errno),__LINE__); exit(1);} }while(0)

// ------------------------------------------------------------------ device kernels (local only)
// landing layout per rank: [world columns][slot_bytes]; column c = rank c's NVFP4 slice of MY shard
//   slot: packed shard/2 bytes, then scales shard/16 bytes (16-byte aligned slot)
__global__ void rs_reduce_kernel(const uint8_t* __restrict__ landing, size_t slot_bytes, size_t shard, int world,
                                 float SF, float SFr, uint8_t* __restrict__ reduced){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=shard/EPT32) return;
    const size_t lo=tid*EPT32; float acc[32];
#pragma unroll
    for(int i=0;i<32;i++) acc[i]=0.f;
    const float invSF=recip_ftz(SF);
    for(int r=0;r<world;r++){ const uint8_t* col=landing+(size_t)r*slot_bytes;
        uint4 e=*reinterpret_cast<const uint4*>(col+lo/2);
        uint16_t s2=*reinterpret_cast<const uint16_t*>(col+shard/2+lo/SF_VEC_SIZE);
        dq32_acc(e,s2,invSF,acc); }
    uint32_t ow[4]; uint8_t sfb[2];
    q16(acc,SFr,ow,sfb[0]); q16(acc+16,SFr,ow+2,sfb[1]);
    *reinterpret_cast<uint4*>(reduced+lo/2)=make_uint4(ow[0],ow[1],ow[2],ow[3]);
    *reinterpret_cast<uint16_t*>(reduced+shard/2+lo/SF_VEC_SIZE)=uint16_t(sfb[0])|(uint16_t(sfb[1])<<8);
}
// ag landing per rank: [world columns][slot_bytes]; column c = rank c's reduced shard -> out[c*shard ...]
__global__ void ag_dequant_kernel(const uint8_t* __restrict__ landing, size_t slot_bytes, size_t shard, size_t numel,
                                  float SFr, __nv_bfloat16* __restrict__ out){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=numel/EPT32) return;
    const size_t g=tid*EPT32; const int c=int(g/shard); const size_t lo=g-(size_t)c*shard;
    const uint8_t* col=landing+(size_t)c*slot_bytes;
    uint4 e=*reinterpret_cast<const uint4*>(col+lo/2);
    uint16_t s2=*reinterpret_cast<const uint16_t*>(col+shard/2+lo/SF_VEC_SIZE);
    float acc[32];
#pragma unroll
    for(int i=0;i<32;i++) acc[i]=0.f;
    dq32_acc(e,s2,recip_ftz(SFr),acc);
#pragma unroll
    for(int k=0;k<4;k++){ Packed16 pk;
#pragma unroll
        for(int j=0;j<4;j++) pk.unpacked[j]=__floats2bfloat162_rn(acc[k*8+2*j],acc[k*8+2*j+1]);
        *reinterpret_cast<int4*>(&out[g+k*8])=pk.packed; }
}

// one-shot landing per rank: [world columns][L.total_bytes]; column c = rank c's whole quantized tensor -> reduce -> BF16 out
__global__ void os_reduce_kernel(const uint8_t* __restrict__ landing, size_t col_bytes, Layout L, int world, float SF, __nv_bfloat16* __restrict__ out){
    const size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(tid>=L.numel/EPT32) return;
    const size_t g=tid*EPT32; float acc[32];
#pragma unroll
    for(int i=0;i<32;i++) acc[i]=0.f;
    const float invSF=recip_ftz(SF);
    for(int r=0;r<world;r++){ uint8_t* col=const_cast<uint8_t*>(landing)+(size_t)r*col_bytes;
        uint4 e=*reinterpret_cast<const uint4*>(L.packed(col)+g/2);
        uint16_t s2=*reinterpret_cast<const uint16_t*>(L.scales(col)+g/SF_VEC_SIZE);
        dq32_acc(e,s2,invSF,acc); }
#pragma unroll
    for(int k=0;k<4;k++){ Packed16 pk;
#pragma unroll
        for(int j=0;j<4;j++) pk.unpacked[j]=__floats2bfloat162_rn(acc[k*8+2*j],acc[k*8+2*j+1]);
        *reinterpret_cast<int4*>(&out[g+k*8])=pk.packed; }
}

// ------------------------------------------------------------------ RDMA plumbing
struct RdmaRank {
    ibv_context* ctx=nullptr; ibv_pd* pd=nullptr; ibv_cq* cq=nullptr;
    ibv_qp* qp[NRANKS]={};
    ibv_mr *mr_payload=nullptr,*mr_rs=nullptr,*mr_reduced=nullptr,*mr_ag=nullptr,*mr_os=nullptr;
    ibv_gid gid{}; uint16_t lid=0; ibv_mtu mtu=IBV_MTU_1024; std::string nic;
    int pending_send=0, pending_recv=0;
};
static std::string sys_realpath(const std::string& p){ char buf[PATH_MAX]; if(realpath(p.c_str(),buf)) return buf; return ""; }
static size_t common_prefix(const std::string& a,const std::string& b){ size_t n=std::min(a.size(),b.size()),i=0; while(i<n&&a[i]==b[i]) i++; return i; }

static ibv_mr* reg_gpu_mr(ibv_pd* pd, void* ptr, size_t bytes, int access){
    // 1) nvidia-peermem path
    ibv_mr* mr=ibv_reg_mr(pd,ptr,bytes,access);
    if(mr) return mr;
    int e1=errno;
    // 2) dma-buf path (open kernel module)
    int fd=-1; CUresult r=cuMemGetHandleForAddressRange(&fd,(CUdeviceptr)ptr,bytes,CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,0);
    if(r==CUDA_SUCCESS){ mr=ibv_reg_dmabuf_mr(pd,0,bytes,(uint64_t)ptr,fd,access); int e2=errno; close(fd);
        if(mr) return mr; fprintf(stderr,"ibv_reg_dmabuf_mr failed: %s\n",strerror(e2)); }
    fprintf(stderr,"GPU MR registration failed: ibv_reg_mr: %s; dma-buf export CUresult=%d (load nvidia-peermem?)\n",strerror(e1),(int)r);
    exit(1);
}
static void post_recv(RdmaRank& R,int peer){ ibv_recv_wr wr{}; wr.wr_id=1000+peer; ibv_recv_wr* bad=nullptr; IBV_CHECK(ibv_post_recv(R.qp[peer],&wr,&bad)==0,"post_recv"); R.pending_recv++; }
static void post_write(RdmaRank& R,int peer,ibv_mr* lmr,const void* laddr,size_t bytes,uint64_t raddr,uint32_t rkey,bool imm){
    ibv_sge sge{}; sge.addr=(uint64_t)laddr; sge.length=(uint32_t)bytes; sge.lkey=lmr->lkey;
    ibv_send_wr wr{}; wr.wr_id=peer; wr.sg_list=&sge; wr.num_sge=1; wr.opcode=imm?IBV_WR_RDMA_WRITE_WITH_IMM:IBV_WR_RDMA_WRITE;
    wr.send_flags=IBV_SEND_SIGNALED; wr.wr.rdma.remote_addr=raddr; wr.wr.rdma.rkey=rkey; wr.imm_data=0;
    ibv_send_wr* bad=nullptr; IBV_CHECK(ibv_post_send(R.qp[peer],&wr,&bad)==0,"post_send"); R.pending_send++;
}
static void drain(std::vector<RdmaRank>& RR){
    for(;;){ bool any=false; for(auto& R:RR){ if(R.pending_send==0&&R.pending_recv==0) continue; any=true;
            ibv_wc wc[16]; int n=ibv_poll_cq(R.cq,16,wc);
            for(int i=0;i<n;i++){ if(wc[i].status!=IBV_WC_SUCCESS){ fprintf(stderr,"wc error %s (opcode %d, wr_id %llu) on %s\n",ibv_wc_status_str(wc[i].status),wc[i].opcode,(unsigned long long)wc[i].wr_id,R.nic.c_str()); exit(1); }
                if(wc[i].opcode==IBV_WC_RECV_RDMA_WITH_IMM) R.pending_recv--; else R.pending_send--; } }
        if(!any) break; }
}

int main(int argc,char** argv){
    const int world=NRANKS; size_t numel=(argc>1)?strtoull(argv[1],0,10):(1u<<20);
    int ndev=0; CUDA_CHECK(cudaGetDeviceCount(&ndev)); if(ndev<world){ printf("need %d GPUs, have %d\n",world,ndev); return 1; }
    if(numel%(size_t(world)*EPT32)!=0){ printf("numel must be %d-aligned\n",world*EPT32); return 1; }
    const size_t shard=numel/world, slot=((shard/2+shard/SF_VEC_SIZE)+15)&~size_t(15);
    std::string transports=getenv("AR_TRANSPORT")?getenv("AR_TRANSPORT"):"sm,ce,rdma";
    auto want=[&](const char* t){ return transports.find(t)!=std::string::npos; };
    const int TPB=256;
    CU_CHECK(cuInit(0));
    for(int i=0;i<world;i++){ CUDA_CHECK(cudaSetDevice(i)); for(int j=0;j<world;j++) if(i!=j){ int can=0; cudaDeviceCanAccessPeer(&can,i,j); if(can){ cudaError_t e=cudaDeviceEnablePeerAccess(j,0); if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);} } }

    // ---- data ----
    Layout L=Layout::make(numel);
    std::vector<std::vector<float>> hin(world,std::vector<float>(numel)); std::vector<float> ref(numel,0);
    srand(1234); for(int r=0;r<world;r++) for(size_t i=0;i<numel;i++){ float v=(rand()/float(RAND_MAX))*2-1; hin[r][i]=v; ref[i]+=v; }
    float amax=0; for(size_t i=0;i<numel;i++) amax=fmaxf(amax,fabsf(hin[0][i])); const float SF=(E2M1_MAX*448.f)/amax, SFr=SF/float(world);

    std::vector<__nv_bfloat16*> d_in(world),d_out(world);
    std::vector<uint8_t*> d_payload(world),d_rs(world),d_reduced(world),d_ag(world),d_comm(world),d_os(world),d_comm1(world);
    std::vector<uint32_t*> d_b1(world); std::vector<uint8_t**> pp_comm1(world); std::vector<uint32_t**> pp_b1(world);
    std::vector<uint32_t*> d_bin(world),d_bout(world); std::vector<uint8_t**> pp_comm(world); std::vector<uint32_t**> pp_bin(world),pp_bout(world);
    std::vector<cudaStream_t> st(world); std::vector<cudaEvent_t> ev_a(world),ev_b(world),ev_c(world);
    const size_t commB=fused_comm_bytes(shard,world), barW=fused_barrier_words(world);
    for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaStreamCreateWithFlags(&st[r],cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_a[r],cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&ev_b[r],cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&ev_c[r],cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&d_in[r],numel*2)); CUDA_CHECK(cudaMalloc(&d_out[r],numel*2));
        CUDA_CHECK(cudaMalloc(&d_payload[r],L.total_bytes)); CUDA_CHECK(cudaMalloc(&d_rs[r],slot*world)); CUDA_CHECK(cudaMemset(d_rs[r],0,slot*world));
        CUDA_CHECK(cudaMalloc(&d_reduced[r],slot)); CUDA_CHECK(cudaMalloc(&d_ag[r],slot*world)); CUDA_CHECK(cudaMemset(d_ag[r],0,slot*world));
        CUDA_CHECK(cudaMalloc(&d_comm[r],commB)); CUDA_CHECK(cudaMemset(d_comm[r],0,commB));
        CUDA_CHECK(cudaMalloc(&d_os[r],L.total_bytes*world)); CUDA_CHECK(cudaMemset(d_os[r],0,L.total_bytes*world));
        CUDA_CHECK(cudaMalloc(&d_comm1[r],oneshot_comm_bytes(numel,world))); CUDA_CHECK(cudaMemset(d_comm1[r],0,oneshot_comm_bytes(numel,world)));
        CUDA_CHECK(cudaMalloc(&d_b1[r],barW*4)); CUDA_CHECK(cudaMemset(d_b1[r],0,barW*4));
        CUDA_CHECK(cudaMalloc(&d_bin[r],barW*4)); CUDA_CHECK(cudaMemset(d_bin[r],0,barW*4)); CUDA_CHECK(cudaMalloc(&d_bout[r],barW*4)); CUDA_CHECK(cudaMemset(d_bout[r],0,barW*4));
        std::vector<__nv_bfloat16> tmp(numel); for(size_t i=0;i<numel;i++) tmp[i]=__float2bfloat16(hin[r][i]);
        CUDA_CHECK(cudaMemcpy(d_in[r],tmp.data(),numel*2,cudaMemcpyHostToDevice)); }
    for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&pp_comm[r],world*8)); CUDA_CHECK(cudaMemcpy(pp_comm[r],d_comm.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_bin[r],world*8)); CUDA_CHECK(cudaMemcpy(pp_bin[r],d_bin.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_bout[r],world*8)); CUDA_CHECK(cudaMemcpy(pp_bout[r],d_bout.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_comm1[r],world*8)); CUDA_CHECK(cudaMemcpy(pp_comm1[r],d_comm1.data(),world*8,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&pp_b1[r],world*8)); CUDA_CHECK(cudaMemcpy(pp_b1[r],d_b1.data(),world*8,cudaMemcpyHostToDevice)); }
    auto sync_all=[&](){ for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); CUDA_CHECK(cudaDeviceSynchronize()); } };
    auto rel_rmse=[&](int r){ std::vector<__nv_bfloat16> h(numel); CUDA_CHECK(cudaSetDevice(r)); CUDA_CHECK(cudaMemcpy(h.data(),d_out[r],numel*2,cudaMemcpyDeviceToHost));
        double s=0,sr=0; for(size_t i=0;i<numel;i++){ double e=__bfloat162float(h[i])-ref[i]; s+=e*e; sr+=double(ref[i])*ref[i]; } return sqrt(s/(sr+1e-12)); };
    const int grid_q=int((numel/ELTS_PER_THREAD+TPB-1)/TPB), grid_rs=int((shard/EPT32+TPB-1)/TPB), grid_ag=int((numel/EPT32+TPB-1)/TPB);
    int g_sm=getenv("NV_GRID")?atoi(getenv("NV_GRID")):1;   // PCIe optimum for the in-kernel transport
    int g_sm1=getenv("NV_GRID1")?atoi(getenv("NV_GRID1")):1;
    const bool big=numel>=(size_t(1)<<23); const int WARMUP=big?3:10, ITERS=big?10:50;
    printf("world=%d numel=%zu shard=%zu  NVFP4 slot/rank=%zu B  transports=%s\n",world,numel,shard,slot,transports.c_str());

    // ================= sm: fused in-kernel transport =================
    uint32_t flag=1;
    auto run_sm=[&](){ for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); launch_fused_rank<NRANKS>(d_in[r],d_out[r],pp_comm[r],pp_bin[r],pp_bout[r],r,numel,SF,flag,st[r],TPB,g_sm); } flag++; };

    uint32_t flag1=1;
    auto run_sm1=[&](){ for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); launch_oneshot_fused_rank<NRANKS>(d_in[r],d_out[r],pp_comm1[r],pp_b1[r],r,numel,SF,flag1,st[r],TPB,g_sm1); } flag1++; };
    // ce one-shot: quantize, push whole payload to every peer's column, wait all, local reduce
    auto run_ce1=[&](){
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); quantize_kernel<<<grid_q,TPB,0,st[r]>>>(d_in[r],d_payload[r],L,SF);
            for(int p=0;p<world;p++){ uint8_t* dst=d_os[p]+(size_t)r*L.total_bytes;
                if(p==r) CUDA_CHECK(cudaMemcpyAsync(dst,d_payload[r],L.total_bytes,cudaMemcpyDeviceToDevice,st[r]));
                else CUDA_CHECK(cudaMemcpyPeerAsync(dst,p,d_payload[r],r,L.total_bytes,st[r])); }
            CUDA_CHECK(cudaEventRecord(ev_a[r],st[r])); }
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); for(int p=0;p<world;p++) if(p!=r) CUDA_CHECK(cudaStreamWaitEvent(st[r],ev_a[p],0));
            os_reduce_kernel<<<grid_ag,TPB,0,st[r]>>>(d_os[r],L.total_bytes,L,world,SF,d_out[r]); }
    };

    // ================= ce: copy-engine transport, stream-ordered =================
    auto run_ce=[&](){
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r));
            quantize_kernel<<<grid_q,TPB,0,st[r]>>>(d_in[r],d_payload[r],L,SF);
            for(int p=0;p<world;p++){ uint8_t* dst=d_rs[p]+(size_t)r*slot;
                const uint8_t* sp=L.packed(d_payload[r])+(size_t)p*shard/2; const uint8_t* ss=L.scales(d_payload[r])+(size_t)p*shard/SF_VEC_SIZE;
                if(p==r){ CUDA_CHECK(cudaMemcpyAsync(dst,sp,shard/2,cudaMemcpyDeviceToDevice,st[r])); CUDA_CHECK(cudaMemcpyAsync(dst+shard/2,ss,shard/SF_VEC_SIZE,cudaMemcpyDeviceToDevice,st[r])); }
                else { CUDA_CHECK(cudaMemcpyPeerAsync(dst,p,sp,r,shard/2,st[r])); CUDA_CHECK(cudaMemcpyPeerAsync(dst+shard/2,p,ss,r,shard/SF_VEC_SIZE,st[r])); } }
            CUDA_CHECK(cudaEventRecord(ev_a[r],st[r])); }
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r));
            for(int p=0;p<world;p++) if(p!=r) CUDA_CHECK(cudaStreamWaitEvent(st[r],ev_a[p],0));
            rs_reduce_kernel<<<grid_rs,TPB,0,st[r]>>>(d_rs[r],slot,shard,world,SF,SFr,d_reduced[r]);
            for(int p=0;p<world;p++){ uint8_t* dst=d_ag[p]+(size_t)r*slot;
                if(p==r) CUDA_CHECK(cudaMemcpyAsync(dst,d_reduced[r],slot,cudaMemcpyDeviceToDevice,st[r]));
                else CUDA_CHECK(cudaMemcpyPeerAsync(dst,p,d_reduced[r],r,slot,st[r])); }
            CUDA_CHECK(cudaEventRecord(ev_b[r],st[r])); }
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r));
            for(int p=0;p<world;p++) if(p!=r) CUDA_CHECK(cudaStreamWaitEvent(st[r],ev_b[p],0));
            ag_dequant_kernel<<<grid_ag,TPB,0,st[r]>>>(d_ag[r],slot,shard,numel,SFr,d_out[r]); }
    };

    // ================= rdma: GPUDirect RDMA through the local NICs =================
    std::vector<RdmaRank> RR(world); bool rdma_ok=want("rdma");
    double t_rd_q=0,t_rd_rs=0,t_rd_red=0,t_rd_ag=0,t_rd_dq=0; int rd_iters=0; bool need_flush=false;
    if(rdma_ok){
        int ndevs=0; ibv_device** list=ibv_get_device_list(&ndevs); IBV_CHECK(list&&ndevs>0,"ibv_get_device_list");
        std::vector<std::string> nic_names, nic_paths; std::vector<ibv_device*> nic_dev;
        for(int i=0;i<ndevs;i++){ std::string n=ibv_get_device_name(list[i]); if(n.find("bond")!=std::string::npos) continue;
            std::string st_; { FILE* f=fopen(("/sys/class/infiniband/"+n+"/ports/1/state").c_str(),"r"); char b[64]={0}; if(f){ fgets(b,63,f); fclose(f);} st_=b; }
            if(st_.find("ACTIVE")==std::string::npos) continue;
            nic_names.push_back(n); nic_paths.push_back(sys_realpath("/sys/class/infiniband/"+n+"/device")); nic_dev.push_back(list[i]); }
        std::vector<int> nic_of(world,-1); std::vector<bool> used(nic_names.size(),false);
        if(const char* m=getenv("NIC_MAP")){ std::string s=m; size_t pos=0; for(int r=0;r<world;r++){ size_t c=s.find(',',pos); std::string tok=s.substr(pos,c==std::string::npos?std::string::npos:c-pos); std::string name="mlx5_"+tok; for(size_t i=0;i<nic_names.size();i++) if(nic_names[i]==name) nic_of[r]=(int)i; pos=(c==std::string::npos)?s.size():c+1; } }
        else for(int r=0;r<world;r++){ char bus[32]; CUDA_CHECK(cudaDeviceGetPCIBusId(bus,32,r));
            std::string b=bus; for(auto& ch:b) ch=tolower(ch); size_t c1=b.find(':'); std::string addr="0000:"+b.substr(c1+1);   // "00000000:06:00.0" -> "0000:06:00.0"
            std::string gp=sys_realpath("/sys/bus/pci/devices/"+addr); if(gp.empty()){ fprintf(stderr,"no sysfs entry for GPU %d (%s)\n",r,addr.c_str()); exit(1); }
            for(auto& ch:gp) ch=tolower(ch); int best=-1; size_t bl=0;
            for(size_t i=0;i<nic_names.size();i++){ if(used[i]) continue; std::string np=nic_paths[i]; for(auto& ch:np) ch=tolower(ch); size_t l=common_prefix(gp,np); if(l>bl){ bl=l; best=(int)i; } }
            IBV_CHECK(best>=0,"nic mapping"); nic_of[r]=best; used[best]=true; }
        int gid_index=getenv("RDMA_GID")?atoi(getenv("RDMA_GID")):3;
        printf("RDMA: gid_index=%d  GPU->NIC:",gid_index); for(int r=0;r<world;r++) printf(" %d->%s",r,nic_names[nic_of[r]].c_str()); printf("\n");
        for(int r=0;r<world;r++){ RdmaRank& R=RR[r]; R.nic=nic_names[nic_of[r]];
            R.ctx=ibv_open_device(nic_dev[nic_of[r]]); IBV_CHECK(R.ctx,"ibv_open_device");
            R.pd=ibv_alloc_pd(R.ctx); IBV_CHECK(R.pd,"ibv_alloc_pd");
            R.cq=ibv_create_cq(R.ctx,256,nullptr,nullptr,0); IBV_CHECK(R.cq,"ibv_create_cq");
            ibv_port_attr pa{}; IBV_CHECK(ibv_query_port(R.ctx,1,&pa)==0,"query_port"); R.lid=pa.lid; R.mtu=pa.active_mtu;
            IBV_CHECK(ibv_query_gid(R.ctx,1,gid_index,&R.gid)==0,"query_gid");
            CUDA_CHECK(cudaSetDevice(r));
            int ord=0,fl=0; cudaDeviceGetAttribute(&ord,cudaDevAttrGPUDirectRDMAWritesOrdering,r); cudaDeviceGetAttribute(&fl,cudaDevAttrGPUDirectRDMAFlushWritesOptions,r);
            if(r==0) printf("RDMA: GPUDirectRDMAWritesOrdering=%d flushOptions=%d -> %s\n",ord,fl,(ord<cudaGPUDirectRDMAWritesOrderingOwner)?"host flush before consuming":"no flush needed");
            if(ord<cudaGPUDirectRDMAWritesOrderingOwner) need_flush=true;
            const int acc=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
            R.mr_payload=reg_gpu_mr(R.pd,d_payload[r],L.total_bytes,acc); R.mr_rs=reg_gpu_mr(R.pd,d_rs[r],slot*world,acc);
            R.mr_reduced=reg_gpu_mr(R.pd,d_reduced[r],slot,acc); R.mr_ag=reg_gpu_mr(R.pd,d_ag[r],slot*world,acc); R.mr_os=reg_gpu_mr(R.pd,d_os[r],L.total_bytes*world,acc);
            for(int p=0;p<world;p++){ if(p==r) continue; ibv_qp_init_attr qa{}; qa.send_cq=R.cq; qa.recv_cq=R.cq; qa.qp_type=IBV_QPT_RC;
                qa.cap.max_send_wr=64; qa.cap.max_recv_wr=64; qa.cap.max_send_sge=1; qa.cap.max_recv_sge=1;
                R.qp[p]=ibv_create_qp(R.pd,&qa); IBV_CHECK(R.qp[p],"ibv_create_qp");
                ibv_qp_attr a{}; a.qp_state=IBV_QPS_INIT; a.pkey_index=0; a.port_num=1; a.qp_access_flags=IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
                IBV_CHECK(ibv_modify_qp(R.qp[p],&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)==0,"qp INIT"); } }
        // connect qp[r][p] <-> qp[p][r]
        for(int r=0;r<world;r++) for(int p=0;p<world;p++){ if(p==r) continue; RdmaRank& R=RR[r]; RdmaRank& P=RR[p];
            ibv_qp_attr a{}; a.qp_state=IBV_QPS_RTR; a.path_mtu=(ibv_mtu)std::min((int)R.mtu,(int)P.mtu); a.dest_qp_num=P.qp[r]->qp_num; a.rq_psn=0;
            a.max_dest_rd_atomic=1; a.min_rnr_timer=12; a.ah_attr.is_global=1; a.ah_attr.grh.dgid=P.gid; a.ah_attr.grh.sgid_index=gid_index; a.ah_attr.grh.hop_limit=64;
            a.ah_attr.dlid=P.lid; a.ah_attr.sl=0; a.ah_attr.src_path_bits=0; a.ah_attr.port_num=1;
            IBV_CHECK(ibv_modify_qp(R.qp[p],&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)==0,"qp RTR");
            ibv_qp_attr b{}; b.qp_state=IBV_QPS_RTS; b.timeout=14; b.retry_cnt=7; b.rnr_retry=7; b.sq_psn=0; b.max_rd_atomic=1;
            IBV_CHECK(ibv_modify_qp(R.qp[p],&b,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)==0,"qp RTS"); }
        ibv_free_device_list(list);
    }
    auto flush_all=[&](){ if(!need_flush) return; for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); CUDA_CHECK(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,cudaFlushGPUDirectRDMAWritesToOwner)); } };
    auto now=[](){ return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    auto run_rdma=[&](bool timed){
        double t0=now();
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); quantize_kernel<<<grid_q,TPB,0,st[r]>>>(d_in[r],d_payload[r],L,SF);
            uint8_t* dst=d_rs[r]+(size_t)r*slot; CUDA_CHECK(cudaMemcpyAsync(dst,L.packed(d_payload[r])+(size_t)r*shard/2,shard/2,cudaMemcpyDeviceToDevice,st[r]));
            CUDA_CHECK(cudaMemcpyAsync(dst+shard/2,L.scales(d_payload[r])+(size_t)r*shard/SF_VEC_SIZE,shard/SF_VEC_SIZE,cudaMemcpyDeviceToDevice,st[r]));
            CUDA_CHECK(cudaEventRecord(ev_a[r],st[r])); }
        for(int r=0;r<world;r++) for(int p=0;p<world;p++) if(p!=r) post_recv(RR[r],p);      // one imm per incoming slice
        for(int r=0;r<world;r++) CUDA_CHECK(cudaEventSynchronize(ev_a[r]));
        double t1=now();
        for(int r=0;r<world;r++) for(int p=0;p<world;p++){ if(p==r) continue; RdmaRank& R=RR[r];
            uint64_t base=(uint64_t)d_rs[p]+(uint64_t)r*slot; uint32_t rkey=RR[p].mr_rs->rkey;
            post_write(R,p,R.mr_payload,L.packed(d_payload[r])+(size_t)p*shard/2,shard/2,base,rkey,false);
            post_write(R,p,R.mr_payload,L.scales(d_payload[r])+(size_t)p*shard/SF_VEC_SIZE,shard/SF_VEC_SIZE,base+shard/2,rkey,true); }
        drain(RR); flush_all();
        double t2=now();
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); rs_reduce_kernel<<<grid_rs,TPB,0,st[r]>>>(d_rs[r],slot,shard,world,SF,SFr,d_reduced[r]);
            CUDA_CHECK(cudaMemcpyAsync(d_ag[r]+(size_t)r*slot,d_reduced[r],slot,cudaMemcpyDeviceToDevice,st[r])); CUDA_CHECK(cudaEventRecord(ev_b[r],st[r])); }
        for(int r=0;r<world;r++) for(int p=0;p<world;p++) if(p!=r) post_recv(RR[r],p);
        for(int r=0;r<world;r++) CUDA_CHECK(cudaEventSynchronize(ev_b[r]));
        double t3=now();
        for(int r=0;r<world;r++) for(int p=0;p<world;p++){ if(p==r) continue; RdmaRank& R=RR[r];
            uint64_t base=(uint64_t)d_ag[p]+(uint64_t)r*slot; uint32_t rkey=RR[p].mr_ag->rkey;
            post_write(R,p,R.mr_reduced,d_reduced[r],shard/2,base,rkey,false);
            post_write(R,p,R.mr_reduced,d_reduced[r]+shard/2,shard/SF_VEC_SIZE,base+shard/2,rkey,true); }
        drain(RR); flush_all();
        double t4=now();
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); ag_dequant_kernel<<<grid_ag,TPB,0,st[r]>>>(d_ag[r],slot,shard,numel,SFr,d_out[r]); CUDA_CHECK(cudaEventRecord(ev_c[r],st[r])); }
        for(int r=0;r<world;r++) CUDA_CHECK(cudaEventSynchronize(ev_c[r]));
        double t5=now();
        if(timed){ t_rd_q+=t1-t0; t_rd_rs+=t2-t1; t_rd_red+=t3-t2; t_rd_ag+=t4-t3; t_rd_dq+=t5-t4; rd_iters++; }
    };

    double t_r1_q=0,t_r1_w=0,t_r1_red=0; int r1_iters=0;
    auto run_rdma1=[&](bool timed){
        double t0=now();
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); quantize_kernel<<<grid_q,TPB,0,st[r]>>>(d_in[r],d_payload[r],L,SF);
            CUDA_CHECK(cudaMemcpyAsync(d_os[r]+(size_t)r*L.total_bytes,d_payload[r],L.total_bytes,cudaMemcpyDeviceToDevice,st[r])); CUDA_CHECK(cudaEventRecord(ev_a[r],st[r])); }
        for(int r=0;r<world;r++) for(int p=0;p<world;p++) if(p!=r) post_recv(RR[r],p);
        for(int r=0;r<world;r++) CUDA_CHECK(cudaEventSynchronize(ev_a[r]));
        double t1=now();
        for(int r=0;r<world;r++) for(int p=0;p<world;p++){ if(p==r) continue; RdmaRank& R=RR[r];
            uint64_t base=(uint64_t)d_os[p]+(uint64_t)r*L.total_bytes; uint32_t rkey=RR[p].mr_os->rkey;
            post_write(R,p,R.mr_payload,L.packed(d_payload[r]),numel/2,base,rkey,false);
            post_write(R,p,R.mr_payload,L.scales(d_payload[r]),numel/SF_VEC_SIZE,base+numel/2,rkey,true); }
        drain(RR); flush_all();
        double t2=now();
        for(int r=0;r<world;r++){ CUDA_CHECK(cudaSetDevice(r)); os_reduce_kernel<<<grid_ag,TPB,0,st[r]>>>(d_os[r],L.total_bytes,L,world,SF,d_out[r]); CUDA_CHECK(cudaEventRecord(ev_c[r],st[r])); }
        for(int r=0;r<world;r++) CUDA_CHECK(cudaEventSynchronize(ev_c[r]));
        double t3=now();
        if(timed){ t_r1_q+=t1-t0; t_r1_w+=t2-t1; t_r1_red+=t3-t2; r1_iters++; }
    };

    // ---- correctness ----
    printf("correctness (rank 0 / rank %d):\n",world-1);
    if(want("sm"))  { run_sm(); sync_all(); printf("  NVFP4 sm    rel_rmse=%.6f / %.6f\n",rel_rmse(0),rel_rmse(world-1)); }
    if(want("ce"))  { run_ce(); sync_all(); printf("  NVFP4 ce    rel_rmse=%.6f / %.6f\n",rel_rmse(0),rel_rmse(world-1)); }
    if(rdma_ok)     { run_rdma(false); sync_all(); printf("  NVFP4 rdma  rel_rmse=%.6f / %.6f\n",rel_rmse(0),rel_rmse(world-1)); }
    if(want("sm"))  { run_sm1(); sync_all(); printf("  NVFP4 one-shot sm    rel_rmse=%.6f / %.6f\n",rel_rmse(0),rel_rmse(world-1)); }
    if(want("ce"))  { run_ce1(); sync_all(); printf("  NVFP4 one-shot ce    rel_rmse=%.6f / %.6f\n",rel_rmse(0),rel_rmse(world-1)); }
    if(rdma_ok)     { run_rdma1(false); sync_all(); printf("  NVFP4 one-shot rdma  rel_rmse=%.6f / %.6f\n",rel_rmse(0),rel_rmse(world-1)); }

    // ---- timing (device events on rank 0 for stream-ordered transports; host wall clock for rdma) ----
    auto time_dev=[&](auto fn)->double{ for(int i=0;i<WARMUP;i++) fn(); sync_all(); cudaSetDevice(0); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a,st[0]); for(int i=0;i<ITERS;i++) fn(); cudaSetDevice(0); cudaEventRecord(b,st[0]); cudaEventSynchronize(b); sync_all(); float ms=0; cudaEventElapsedTime(&ms,a,b); return ms*1e3/ITERS; };
    double t_sm=-1,t_ce=-1,t_rd=-1;
    if(want("sm")) t_sm=time_dev(run_sm);
    if(want("ce")) t_ce=time_dev(run_ce);
    if(rdma_ok){ for(int i=0;i<WARMUP;i++) run_rdma(false); sync_all(); double a=now(); for(int i=0;i<ITERS;i++) run_rdma(true); sync_all(); t_rd=(now()-a)/ITERS; }
    double t1_sm=-1,t1_ce=-1,t1_rd=-1;
    if(want("sm")) t1_sm=time_dev(run_sm1);
    if(want("ce")) t1_ce=time_dev(run_ce1);
    if(rdma_ok){ for(int i=0;i<WARMUP;i++) run_rdma1(false); sync_all(); double a=now(); for(int i=0;i<ITERS;i++) run_rdma1(true); sync_all(); t1_rd=(now()-a)/ITERS; }
    const double wire_bytes=2.0*(world-1)*(shard/2.0+shard/(double)SF_VEC_SIZE);   // per rank, RS push + AG push
    printf("TIMING us/allreduce (two-shot NVFP4, %d GPUs):", world);
    if(t_sm>0) printf("  sm=%.1f (grid %d, %.1f GB/s/rank)",t_sm,g_sm,wire_bytes/t_sm*1e-3);
    if(t_ce>0) printf("  ce=%.1f (%.1f GB/s/rank)",t_ce,wire_bytes/t_ce*1e-3);
    if(t_rd>0) printf("  rdma=%.1f (%.1f GB/s/rank)",t_rd,wire_bytes/t_rd*1e-3);
    printf("\n");
    if(t_rd>0&&rd_iters) printf("  rdma phases (host, us): quant+sync=%.1f  RS-write+cq=%.1f  reduce+sync=%.1f  AG-write+cq=%.1f  dequant+sync=%.1f\n",
        t_rd_q/rd_iters,t_rd_rs/rd_iters,t_rd_red/rd_iters,t_rd_ag/rd_iters,t_rd_dq/rd_iters);
    if(t_sm>0&&t_ce>0) printf("  ce/sm speedup=%.2fx",t_sm/t_ce); if(t_sm>0&&t_rd>0) printf("  rdma/sm speedup=%.2fx",t_sm/t_rd); if(t_ce>0&&t_rd>0) printf("  rdma/ce=%.2fx",t_ce/t_rd); printf("\n");
    const double wire1=(double)(world-1)*L.total_bytes;   // per rank, one-shot push to every peer
    printf("ONE-SHOT us/allreduce (NVFP4, %d GPUs):", world);
    if(t1_sm>0) printf("  sm=%.1f (grid %d)",t1_sm,g_sm1); if(t1_ce>0) printf("  ce=%.1f (%.1f GB/s/rank)",t1_ce,wire1/t1_ce*1e-3); if(t1_rd>0) printf("  rdma=%.1f (%.1f GB/s/rank)",t1_rd,wire1/t1_rd*1e-3); printf("\n");
    if(t1_rd>0&&r1_iters) printf("  one-shot rdma phases (host, us): quant+sync=%.1f  write+cq=%.1f  reduce+sync=%.1f\n",t_r1_q/r1_iters,t_r1_w/r1_iters,t_r1_red/r1_iters);
    return 0;
}
