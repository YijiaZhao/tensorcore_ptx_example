/*
 * Minimal GPU-to-GPU copy through the NICs (GPUDirect RDMA), single process.
 *
 * Third way to move bytes between GPUs, after SM ld/st and the Copy Engine:
 * the NIC's DMA engine reads GPU_SRC's memory over PCIe and writes it into
 * GPU_DST's memory, without touching host memory and without any SM
 * instruction. The GPU never "sends" anything: the host posts a work request
 * (ibv_post_send) and the NIC does the rest. Traffic never crosses the CPU
 * root complex when each GPU is paired with the NIC on its own PCIe bridge,
 * which is why this beats both SM stores and the Copy Engine on switch-less
 * PCIe hosts (see ../allreduce_sm100_sm120/README.md §4.6).
 *
 * What happens, in order:
 *   1. cudaMalloc on both GPUs; ibv_reg_mr registers the GPU memory with each NIC
 *      (needs the nvidia-peermem module; falls back to dma-buf export)
 *   2. one RC queue pair per side, connected to each other (same process, so the
 *      QP numbers / GIDs are exchanged in memory instead of over TCP)
 *   3. GPU_SRC side posts IBV_WR_RDMA_WRITE_WITH_IMM: NIC_SRC reads d_src, sends it
 *      to NIC_DST, which writes d_dst; the immediate makes a completion appear on
 *      the receiver so it knows the data has landed
 *   4. host polls both completion queues, flushes GPUDirect writes if the device
 *      requires it, then verifies d_dst
 *
 * Build:
 *   nvcc -std=c++17 -O2 -arch=sm_100 -o minimal_p2p_rdma_write minimal_p2p_rdma_write.cu -lcuda -libverbs
 * Run:
 *   ./minimal_p2p_rdma_write [src_gpu=0] [dst_gpu=1] [src_nic=mlx5_2] [dst_nic=mlx5_3] [gid_index=3] [bytes=67108864]
 *   Pick NICs that sit next to the GPUs: `nvidia-smi topo -m` (PIX/PXB column). RoCE v2
 *   usually needs a gid_index whose entry is IPv4-mapped; `show_gids` lists them.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call)                                                                     \
    do {                                                                                \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)
#define IBV_CHECK(cond, what)                                                           \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, what, std::strerror(errno)); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

// One end of the connection: a GPU, the NIC next to it, and the verbs objects on that NIC.
struct Endpoint {
    int          gpu = -1;
    std::string  nic_name;
    ibv_context* ctx = nullptr;
    ibv_pd*      pd  = nullptr;
    ibv_cq*      cq  = nullptr;
    ibv_qp*      qp  = nullptr;
    ibv_mr*      mr  = nullptr;     // registration of this side's GPU buffer
    uint8_t*     d_buf = nullptr;
    ibv_gid      gid{};
    uint16_t     lid = 0;
    ibv_mtu      mtu = IBV_MTU_1024;
};

static ibv_device* find_nic(ibv_device** list, int n, const std::string& name) {
    for (int i = 0; i < n; i++)
        if (name == ibv_get_device_name(list[i])) return list[i];
    std::fprintf(stderr, "NIC %s not found; available:", name.c_str());
    for (int i = 0; i < n; i++) std::fprintf(stderr, " %s", ibv_get_device_name(list[i]));
    std::fprintf(stderr, "\n");
    std::exit(1);
}

// Register GPU memory with the NIC. nvidia-peermem lets plain ibv_reg_mr accept a device
// pointer; without it, export the range as a dma-buf and register that.
static ibv_mr* register_gpu_memory(ibv_pd* pd, void* d_ptr, size_t bytes) {
    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    ibv_mr* mr = ibv_reg_mr(pd, d_ptr, bytes, access);
    if (mr) return mr;
    const int peermem_errno = errno;

    int fd = -1;
    CUresult r = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)d_ptr, bytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (r == CUDA_SUCCESS) {
        mr = ibv_reg_dmabuf_mr(pd, 0, bytes, (uint64_t)d_ptr, fd, access);
        close(fd);
        if (mr) return mr;
    }
    std::fprintf(stderr, "GPU memory registration failed: ibv_reg_mr: %s; dma-buf CUresult=%d (is nvidia-peermem loaded?)\n",
                 std::strerror(peermem_errno), int(r));
    std::exit(1);
}

static void open_endpoint(Endpoint& e, ibv_device* dev, int gid_index, size_t bytes) {
    e.ctx = ibv_open_device(dev);
    IBV_CHECK(e.ctx, "ibv_open_device");
    e.pd = ibv_alloc_pd(e.ctx);
    IBV_CHECK(e.pd, "ibv_alloc_pd");
    e.cq = ibv_create_cq(e.ctx, 64, nullptr, nullptr, 0);
    IBV_CHECK(e.cq, "ibv_create_cq");

    ibv_port_attr port{};
    IBV_CHECK(ibv_query_port(e.ctx, 1, &port) == 0, "ibv_query_port");
    e.lid = port.lid;
    e.mtu = port.active_mtu;
    IBV_CHECK(ibv_query_gid(e.ctx, 1, gid_index, &e.gid) == 0, "ibv_query_gid");

    CHECK(cudaSetDevice(e.gpu));
    CHECK(cudaMalloc(&e.d_buf, bytes));
    e.mr = register_gpu_memory(e.pd, e.d_buf, bytes);

    // Reliable-connected QP; send and receive completions share one CQ.
    ibv_qp_init_attr init{};
    init.send_cq = e.cq;
    init.recv_cq = e.cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr  = 32;
    init.cap.max_recv_wr  = 32;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    e.qp = ibv_create_qp(e.pd, &init);
    IBV_CHECK(e.qp, "ibv_create_qp");

    ibv_qp_attr a{};
    a.qp_state        = IBV_QPS_INIT;
    a.pkey_index      = 0;
    a.port_num        = 1;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    IBV_CHECK(ibv_modify_qp(e.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0, "qp -> INIT");
}

// Move `me` to RTR then RTS, pointing at `peer`. Both sides call this.
static void connect_qp(Endpoint& me, const Endpoint& peer, int gid_index) {
    ibv_qp_attr rtr{};
    rtr.qp_state           = IBV_QPS_RTR;
    rtr.path_mtu           = (ibv_mtu)std::min(int(me.mtu), int(peer.mtu));
    rtr.dest_qp_num        = peer.qp->qp_num;
    rtr.rq_psn             = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer      = 12;
    rtr.ah_attr.is_global       = 1;              // RoCE: route by GID
    rtr.ah_attr.grh.dgid        = peer.gid;
    rtr.ah_attr.grh.sgid_index  = gid_index;
    rtr.ah_attr.grh.hop_limit   = 64;
    rtr.ah_attr.dlid            = peer.lid;
    rtr.ah_attr.port_num        = 1;
    IBV_CHECK(ibv_modify_qp(me.qp, &rtr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0, "qp -> RTR");

    ibv_qp_attr rts{};
    rts.qp_state      = IBV_QPS_RTS;
    rts.timeout       = 14;
    rts.retry_cnt     = 7;
    rts.rnr_retry     = 7;
    rts.sq_psn        = 0;
    rts.max_rd_atomic = 1;
    IBV_CHECK(ibv_modify_qp(me.qp, &rts, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0, "qp -> RTS");
}

// The receiver must have a receive WR posted for the immediate to land in.
static void post_recv(Endpoint& e) {
    ibv_recv_wr wr{};
    wr.wr_id = 1;
    ibv_recv_wr* bad = nullptr;
    IBV_CHECK(ibv_post_recv(e.qp, &wr, &bad) == 0, "ibv_post_recv");
}

// THE SEND: one work request "write my [d_buf, +bytes) into peer's d_buf, then signal".
static void post_rdma_write_with_imm(Endpoint& src, const Endpoint& dst, size_t bytes) {
    ibv_sge sge{};
    sge.addr   = (uint64_t)src.d_buf;
    sge.length = (uint32_t)bytes;
    sge.lkey   = src.mr->lkey;

    ibv_send_wr wr{};
    wr.wr_id               = 2;
    wr.sg_list             = &sge;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags          = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = (uint64_t)dst.d_buf;
    wr.wr.rdma.rkey        = dst.mr->rkey;
    wr.imm_data            = htonl(0xC0FFEE);

    ibv_send_wr* bad = nullptr;
    IBV_CHECK(ibv_post_send(src.qp, &wr, &bad) == 0, "ibv_post_send");
}

// Block until one completion with the given opcode shows up on e.cq.
static void wait_completion(Endpoint& e, ibv_wc_opcode expect) {
    for (;;) {
        ibv_wc wc{};
        const int n = ibv_poll_cq(e.cq, 1, &wc);
        if (n == 0) continue;
        IBV_CHECK(n == 1, "ibv_poll_cq");
        if (wc.status != IBV_WC_SUCCESS) {
            std::fprintf(stderr, "completion error on %s: %s (opcode %d)\n", e.nic_name.c_str(), ibv_wc_status_str(wc.status), wc.opcode);
            std::exit(1);
        }
        if (wc.opcode == expect) return;
    }
}

static double now_us() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

int main(int argc, char** argv) {
    Endpoint src, dst;
    src.gpu      = argc > 1 ? std::atoi(argv[1]) : 0;
    dst.gpu      = argc > 2 ? std::atoi(argv[2]) : 1;
    src.nic_name = argc > 3 ? argv[3] : "mlx5_2";
    dst.nic_name = argc > 4 ? argv[4] : "mlx5_3";
    const int    gid_index = argc > 5 ? std::atoi(argv[5]) : 3;
    const size_t bytes     = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : (size_t(64) << 20);

    CHECK(cudaFree(0));   // create the CUDA contexts before touching cuMem* / peermem

    int n_nics = 0;
    ibv_device** nics = ibv_get_device_list(&n_nics);
    IBV_CHECK(nics && n_nics > 0, "ibv_get_device_list");
    open_endpoint(src, find_nic(nics, n_nics, src.nic_name), gid_index, bytes);
    open_endpoint(dst, find_nic(nics, n_nics, dst.nic_name), gid_index, bytes);
    ibv_free_device_list(nics);
    connect_qp(src, dst, gid_index);
    connect_qp(dst, src, gid_index);

    // Does the destination GPU need an explicit flush before the host/SM may read RDMA'd data?
    int ordering = 0;
    CHECK(cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, dst.gpu));
    const bool need_flush = ordering < cudaGPUDirectRDMAWritesOrderingOwner;

    // ---- fill source, clear destination
    std::vector<uint8_t> h_src(bytes);
    for (size_t i = 0; i < bytes; i++) h_src[i] = uint8_t(i * 2654435761u >> 13);
    CHECK(cudaSetDevice(src.gpu));
    CHECK(cudaMemcpy(src.d_buf, h_src.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaSetDevice(dst.gpu));
    CHECK(cudaMemset(dst.d_buf, 0, bytes));
    CHECK(cudaDeviceSynchronize());

    // ---- one transfer, verified
    post_recv(dst);
    post_rdma_write_with_imm(src, dst, bytes);
    wait_completion(src, IBV_WC_RDMA_WRITE);          // NIC_SRC finished reading d_src
    wait_completion(dst, IBV_WC_RECV_RDMA_WITH_IMM);  // NIC_DST finished writing d_dst
    if (need_flush) {
        CHECK(cudaSetDevice(dst.gpu));
        CHECK(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice, cudaFlushGPUDirectRDMAWritesToOwner));
    }
    std::vector<uint8_t> h_dst(bytes);
    CHECK(cudaSetDevice(dst.gpu));
    CHECK(cudaMemcpy(h_dst.data(), dst.d_buf, bytes, cudaMemcpyDeviceToHost));
    const bool ok = (h_dst == h_src);

    // ---- bandwidth: same transfer repeated
    const int iters = 20;
    for (int i = 0; i < 3; i++) { post_recv(dst); post_rdma_write_with_imm(src, dst, bytes); wait_completion(src, IBV_WC_RDMA_WRITE); wait_completion(dst, IBV_WC_RECV_RDMA_WITH_IMM); }
    const double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        post_recv(dst);
        post_rdma_write_with_imm(src, dst, bytes);
        wait_completion(src, IBV_WC_RDMA_WRITE);
        wait_completion(dst, IBV_WC_RECV_RDMA_WITH_IMM);
    }
    const double us = (now_us() - t0) / iters;

    std::printf("GPU %d (%s) -> GPU %d (%s), %zu bytes, gid_index %d, flush %s\n",
                src.gpu, src.nic_name.c_str(), dst.gpu, dst.nic_name.c_str(), bytes, gid_index, need_flush ? "required" : "not needed");
    std::printf("  RDMA_WRITE_WITH_IMM: %9.1f us  %6.1f GB/s  %s\n", us, double(bytes) / us * 1e-3, ok ? "OK" : "MISMATCH");

    for (Endpoint* e : {&src, &dst}) {
        ibv_destroy_qp(e->qp);
        ibv_dereg_mr(e->mr);
        ibv_destroy_cq(e->cq);
        ibv_dealloc_pd(e->pd);
        ibv_close_device(e->ctx);
        CHECK(cudaSetDevice(e->gpu));
        CHECK(cudaFree(e->d_buf));
    }
    return ok ? 0 : 2;
}
