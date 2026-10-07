# comm_primitives_sm100_sm120 — how bytes get from one GPU to another, one primitive per file

Minimal, single-process examples of every data-movement path a GPU has. Each file is
self-contained and prints a pass/fail plus, where it makes sense, a bandwidth. The
complete all-reduce built from these lives in `../allreduce_sm100_sm120/`.

The `sm100_sm120` suffix groups the Blackwell examples; it does **not** mean every
primitive is exclusive to, or verified on, both SM100 and SM120. Some PTX works
on sm_90+, while peer access, multicast, and RDMA additionally depend on the
GPU, interconnect, and NIC. Build device code for the actual GPU (for example,
`-arch=sm_100a` on B200 or `-arch=sm_120a` on supported SM120 hardware);
the build commands below are examples, not cross-architecture binaries.

| file | who moves the bytes | instruction / API | needs |
|---|---|---|---|
| `minimal_p2p_ldst.cu` | SM | `ld.global` / `st.global` on a peer pointer | P2P (NVLink or PCIe) |
| `minimal_p2p_acqrel_volatile.cu` | SM | `st.release.sys` / `ld.acquire.sys` flags vs `volatile` | P2P |
| `minimal_tma_ldst.cu` | SM (TMA unit) | `cp.async.bulk` global ↔ smem | sm_90+ |
| `minimal_p2p_tma.cu` | SM (TMA unit) | `cp.async.bulk` from a peer's global memory | sm_90+, P2P |
| `minimal_multimem_reduce.cu` | NVSwitch | `multimem.ld_reduce` (reduction in the switch) | NVLink multicast (`check_multimem_nvlink_multicast.cu` tells you) |
| `minimal_p2p_copy_engine.cu` | Copy Engine (DMA in the GPU) | `cudaMemcpyPeerAsync`; also SM push / SM pull for comparison | P2P |
| `minimal_p2p_rdma_write.cu` | NIC (DMA in the NIC) | `ibv_reg_mr` on GPU memory, `IBV_WR_RDMA_WRITE_WITH_IMM` | RoCE/IB NIC, `nvidia-peermem` or dma-buf |

Only the SM rows are GPU instructions. Copy Engine and NIC are DMA engines: the host writes a
descriptor / work request, the engine moves the data, no SM is involved. Neither can add, so a
reduction still needs an SM kernel (or the NVSwitch path) after the bytes land.

## Build

```bash
nvcc -std=c++17 -O2 -arch=sm_100 -o minimal_p2p_copy_engine minimal_p2p_copy_engine.cu
nvcc -std=c++17 -O2 -arch=sm_100 -o minimal_p2p_rdma_write minimal_p2p_rdma_write.cu -lcuda -libverbs
# the other files carry their own build line in the header comment
```

## Measured (8× RTX 6000D, PCIe, 2 sockets, no PCIe switch, one 400G RoCE port per GPU; 64 MB, 3 runs)

```
./minimal_p2p_copy_engine 0 4 67108864 4       # GPU0 -> GPU4 is cross-socket
  copy engine (cudaMemcpyPeerAsync):    1488 us   45.1 GB/s
  SM push (remote st.global)       :    1497 us   44.8 GB/s   (grid 4 x 256)
  SM pull (remote ld.global)       :    6714 us   10.0 GB/s   (grid 4 x 256)
./minimal_p2p_rdma_write 0 4 mlx5_2 mlx5_6 3 67108864
  RDMA_WRITE_WITH_IMM:                  1408 us   47.7 GB/s
```

| GPU0 → GPU4, 64 MB | 1 block | 4 | 16 | 64 | 1024 |
|---|---:|---:|---:|---:|---:|
| SM push | 4–6 GB/s | **45** | 22–45 | 22–42 | 20–45 |
| SM pull | 2 | 10 | 16–39 | 15 | 15 |
| copy engine | 45 (grid-independent) |||||
| RDMA write | 47.7 (grid-independent) |||||

Take-aways: remote stores are posted and reach link speed with a handful of blocks; remote loads are
round trips and need many more in flight; too many concurrent writers on a switch-less host hurt
(root-complex contention). The copy engine and the NIC do not care about any of this. The
all-reduce in `../allreduce_sm100_sm120/README.md` §4.6 shows how these single-pair numbers translate (or fail to)
into an 8-GPU collective.
