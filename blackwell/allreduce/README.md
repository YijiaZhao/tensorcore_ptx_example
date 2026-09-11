# NVFP4-compressed all-reduce vs TRT-LLM custom all-reduce (Blackwell)

An all-reduce that moves NVFP4 (E2M1 + per-16 UE4M3 block scale, 0.5625 B/elt) over the wire
instead of BF16 (2 B/elt) or FP8 (TRT-LLM low-precision AR, ~1.06 B/elt), built on the exact
kernel structure of TRT-LLM's `twoShotAllReduceKernel`, and benchmarked against **TRT-LLM's own
kernels compiled verbatim into the same binary**:

- `TRT-BF16` — TRT-LLM `customAllReduceKernels.cu` one-shot / two-shot (PUSH_MODE, `block_barrier`).
- `TRT-FP8`  — TRT-LLM `customLowPrecisionAllReduceKernels.cu` two-shot (preprocess + fused kernel).
- `NVFP4`    — this work: **fused two-shot** and **fused one-shot** (one kernel each, TRT-LLM structure) plus the split (multi-launch) variants used for phase timing.

Files:

| file | what |
|---|---|
| `nvfp4_allreduce.cuh` | standalone header: `launch_fused_rank<RANKS>()` (two-shot, recommended), `launch_oneshot_fused_rank<RANKS>()` (one-shot), `launch_twoshot_rank()` (split) |
| `bench_ar.cu` | single-process harness, 8 GPUs in one node; all baselines + NVFP4; `PHASE=1` per-phase timing |
| `bench_ar_mp.cu` | multi-process harness (one process per node, fabric handles + IMEX) for 2-node GB300 |
| `bench_ar_pcie.cu` | PCIe wire transports for the same NVFP4 two-shot: in-kernel stores (`sm`) vs copy engines (`ce`) vs **GPUDirect RDMA through the host's own NICs** (`rdma`) |

Headline (two-shot, 8 GPUs, clocks locked, µs; **fused NVFP4 vs TRT-LLM FP8**):

| GPU | interconnect | 32M / 512K | 128M / 8M | 512M | 2G |
|---|---|---|---|---|---|
| B200 ×8 | NVLink | **1.84×** | **1.91×** | **1.80×** | **1.91×** |
| GB300 2×4 (NVL72) | NVLink cross-node | **1.73×** | **1.78×** | **1.67×** | **1.78×** |
| RTX 6000D ×8 (PCIe), best grid both | PCIe | **6.1×** (32K) · **4.6×** (128K) · **2.9×** (512K) | **1.76×** (8M) | — | — |
| RTX PRO 6000 ×8 (PCIe), best grid both | PCIe | **4.8×** (32K) · **3.6×** (128K) · **2.6×** (512K) | **2.1×** (8M) | — | — |

vs TRT-LLM BF16 custom AR the fused NVFP4 kernel is 3.9–11.8× faster on NVLink (≥8M) and 2.3–6.3× on PCIe (best grid).
**PCIe caveat:** TRT-FP8 with 1 block instead of its fixed 16 is up to 2.6× faster on the 6000D; the PCIe ratios above give TRT-FP8 that best grid (§4.1).
One-shot: the fused NVFP4 one-shot beats TRT-LLM's BF16 one-shot at every size on NVLink (1.1× on the 16 µs latency floor, 5–8× from 8M up) and 4.4–5.6× on PCIe.
**PCIe wire transport (§2.5, §4.1, §4.6):** on the 8× RTX 6000D every in-kernel all-reduce — ours and TRT-LLM's — moves ~1 GB/s per rank because SM stores across the CPU root complex degrade to small PCIe transactions. Moving the same NVFP4 bytes with GPUDirect RDMA through the host's own NICs (one 400G port per GPU, RC QPs between ranks) reaches 32–40 GB/s per rank: two-shot **387 µs at 8M vs 7969 µs in-kernel, 14056 µs TRT-FP8 (best grid) and 49833 µs TRT-BF16**; one-shot 833 µs vs 158265 µs TRT-BF16 one-shot.
Accuracy cost: rel_rmse 0.14 (two-shot, two quantization passes) / 0.10 (one-shot) vs 0.037 (FP8) vs 0.004 (BF16).

---

## 1. Test environment

| GPU | sm | GPUs | interconnect | build toolchain | clock lock (verified by `nvidia-smi --query-gpu=clocks.sm` after each run) |
|---|---|---|---|---|---|
| RTX 6000D (cc 12.0) | sm_120 | 8 | PCIe | CUDA 13.3 nvcc, local | `nvidia-smi -lgc <max>` → 2347–2355 MHz |
| RTX PRO 6000 Blackwell Server Edition | sm_120 | 8 | PCIe | static binary (`-cudart static`) built with CUDA 13.3 on an x86 host; no toolkit on the node | slurm `--gpu-freq=high` → 2347 MHz (node 1), 2295 MHz (node 2) |
| B200 (cc 10.0) | sm_100 | 8 | NVLink | static binary built with CUDA 13.3 on an x86 host | slurm `--gpu-freq=1965` → 1965 MHz |
| GB300 NVL72 (cc 10.3, aarch64) | sm_103 | 8 = **2 nodes × 4** | NVLink (intra-rack, cross-node) | built inside `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` (CUDA 13.1, aarch64); no toolkit on the node | `--gpu-freq=high` requested (slurm reported `control_disabled`); read back 2070 MHz = GB300 max SM clock |

- Timing: CUDA events on rank 0; 20 warm-up + 100 timed iterations (5 + 20 for ≥128M elements). µs per all-reduce.
- Accuracy: rel_rmse vs an FP32 host reference of the same random inputs (fixed seed); NVFP4 two-shot is bit-reproducible run to run.
- Harness `bench_ar.cu`: one process, `cudaDeviceEnablePeerAccess` + raw peer pointers (no NCCL/MPI).
  `bench_ar_mp.cu`: one process per node, buffers from `cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC)`, handles
  exchanged through files on the shared `$HOME`, remote ranks mapped with `cuMemImportFromShareableHandle`
  + `cuMemMap`; needs IMEX (`/dev/nvidia-caps-imex-channels`) and both nodes in one NVL72 domain.
- GB300 8-GPU numbers were reproduced on three NVL72 racks (two sites) within 1–3 % at every size.
- Driver versions were not recorded.

---

## 2. What is being compared

### 2.1 TRT-LLM kernels (verbatim)

- **TRT-BF16 two-shot** (`twoShotAllReduceKernel`, PUSH_MODE): every block, for each peer in
  rank-rotated order `(local_rank+ii)%RANKS`, pushes its int4 slice into the peer's comm buffer
  (column = my rank), `block_barrier`, reduces the RANKS columns of its own buffer locally
  (`add128b`, rotated summation order), `block_barrier`, then pulls each owner's reduced column
  into the output. Grid ≤ 64 blocks (`MAX_ALL_REDUCE_BLOCKS`) — sized for small messages, which is
  why it sits at ~300 GB/s on large ones.
- **TRT-BF16 one-shot** (`oneShotAllReduceKernel`, PUSH_MODE).
- **TRT-FP8 two-shot** (`lowPrecisionTwoShotAllReduceKernel` + `preprocess`): BF16→FP8 with one
  FP32 scale per warp stored in lane 31's slot (496 data + 1 scale per 512 elements); fused kernel
  with in-kernel `lp_multi_gpu_barrier`, 16-element int4 loads, dequant→FP32 reduce→requant in place.
  TRT-LLM dispatches it with a fixed grid of 16 blocks (PCIe-tuned, ≤4 ranks; the 8-rank NUMA-hierarchical
  variant is not ported). On NVLink 16 blocks are occupancy-starved and run at BF16 speed, so the
  harness scales the grid to one block per 7936-element round, clamped to [16, 2048] (`TRT_FP8_GRID`
  env pins it; `TRT_FP8_GRID=16` reproduces the verbatim dispatch). On PCIe the grid makes ≤4 % difference.

### 2.2 NVFP4 (this work)

Wire format per rank: `numel/2` bytes of packed E2M1 + `numel/16` bytes of UE4M3 block scales
(+ one FP32 tensor scale) = 0.5625 B/elt, 3.56× smaller than BF16, 1.89× smaller than FP8.

**Fused kernel** `twoshot_fused_kernel<RANKS>` — TRT-LLM's two-shot skeleton, payload swapped:

1. for each peer `ranks[ii] = (local_rank+ii)%RANKS`: load 32 BF16 of my input belonging to that
   peer's shard (4 × int4), quantize to 2 × 16-element blocks (scale = `SF·amax/6` → UE4M3), **push**
   16 B packed + 2 B scales into the peer's comm buffer, column `local_rank`;
2. `block_barrier` (TRT-LLM's, flag-parity ping-pong, per-block offset);
3. reduce my shard across the RANKS columns of **my own** buffer (local reads, 16 B + 2 B per peer,
   dequant → FP32), requantize once with `SF/world`, write to column 0;
4. `block_barrier`;
5. pull each owner's column 0 (rotated order), dequantize, write 64 B of BF16 output per peer.

Comm buffer per rank: `[2·RANKS columns][shard/2 + shard/16 bytes]`; flag parity selects the
column set so back-to-back calls never overwrite each other (same trick as TRT-LLM). Grid default
(NVLink, from the sweep in §4.5): one `TPB·32`-element chunk per block up to 256 blocks, then 4 chunks
per thread up to 2048; floor 16. On PCIe pass `grid=4…8` (`NV_GRID=8` / `NV_PCIE=1` in the bench).

**Fused one-shot** `oneshot_fused_kernel<RANKS>` — TRT-LLM's `oneShotAllReduceKernel` skeleton: quantize my
chunk once, **push** the 16 B + 2 B to every peer's buffer (column `local_rank`, rotated order, column = whole
tensor), `block_barrier`, then dequantize and sum the RANKS columns of my own buffer into the output. One
quantization pass, so rel_rmse 0.102 instead of 0.140. Grid rule as above (`NV_GRID1` in the bench).

**Split path** (`launch_twoshot_rank`, 5 launches): quantize | counter-barrier | reduce-scatter
(PULL: read every peer's shard slice, rotated order, 16 B loads) | counter-barrier | all-gather
(owner order rotated by rank, 16 B loads). Same kernels' inner loops as the fused kernel; kept for
per-phase timing (`PHASE=1`).

### 2.3 Codec

Hardware E2M1 conversion on every Blackwell (`cvt.rn.satfinite.e2m1x2.f32` encode on sm_100/101/103/120;
`cvt.rn.f16x2.e2m1x2` decode on sm_100/103, PRMT register-LUT decode on sm_120 where the cvt pipe is
slower). These are arch-specific ("a") features: **build with explicit
`-gencode arch=compute_XXXa,code=sm_XXXa`** — the `-arch=sm_XXXa` shorthand does not enable them in
nvcc 13.x (ptxas "not supported on sm_XXX").

### 2.4 What was wrong before (why the first version lost to TRT-FP8 on NVLink)

The first published version kept TRT-LLM's algorithm but not its access pattern, and lost to TRT-FP8
by 17–31 % on B200/GB300. Per-phase timing (B200, 512M, 8 GPUs, µs) located it:

| phase | old NVFP4 | after rank rotation | after rotation + 16 B loads | fused |
|---|---:|---:|---:|---:|
| quantize | 268 | 270 | 271 | — |
| barrier 1 | **546** | 22 | 19 | — |
| reduce-scatter | 501 | 496 | 428 | — |
| barrier 2 | 13 | 10 | 9 | — |
| all-gather | **1331** | 1341 | **454** | — |
| total | 2658 | 2139 | 1180 | **1045** |
| TRT-FP8 (prep + fused) | 1880 | 1880 | 1880 | 1880 |

- **Hot-spotting**: the all-gather indexed `owner = g/shard`, so every rank read owner 0's shard first,
  then owner 1's — seven readers on one GPU's NVLink egress, and a rank skew that the next iteration's
  barrier paid for (546 µs). TRT-LLM rotates the peer order by rank; doing the same removed the skew.
- **Request-bound all-gather**: one 4-byte load per thread per peer gave ~200 GB/s of remote reads,
  while the reduce-scatter (8 independent loads per thread) reached ~530 GB/s; an E4M3-payload twin of
  the same kernel (`myFP8` in the bench output) read twice the bytes in the same time, so it was
  requests, not bandwidth. TRT-LLM uses 16-byte int4 loads; switching to 32 elements per thread
  (16 B packed + 2 B scales) took the all-gather from 1331 to 454 µs.
- Fusing into one kernel with TRT-LLM's `block_barrier` and PUSH reduce-scatter saves the separate
  quantize pass and the two barrier launches: another 7–18 % on NVLink, 1.5–2× on PCIe.

### 2.5 PCIe wire transports (`bench_ar_pcie.cu`)

On a switch-free PCIe host every peer store issued by an SM crosses the CPU root complex and is broken
into small transactions; that is why all in-kernel all-reduces in §4.1/§4.2 sit at ~1 GB/s per rank and
why fewer blocks help. `bench_ar_pcie.cu` keeps the NVFP4 math (quantize → push → local reduce +
requantize → push → dequantize) and swaps only the mechanism that moves bytes:

- `sm`   — the fused kernel of §2.2 (reference).
- `ce`   — `cudaMemcpyPeerAsync` for every slice, fully stream-ordered with cross-device events.
- `rdma` — GPUDirect RDMA through the host's own NICs, following FlashInfer PR #4876's PCIe Ulysses
  transport: each GPU is paired with the NIC on its PCIe bridge (chosen by sysfs path proximity, or
  `NIC_MAP`), one RC QP per (rank, peer) over RoCE v2 (`RDMA_GID`, default 3), buffers registered with
  `nvidia-peermem` (dma-buf fallback), `RDMA_WRITE` for the packed slice and `RDMA_WRITE_WITH_IMM` for
  the scales so the receiver gets a completion, host polls all CQs between phases, and
  `cudaDeviceFlushGPUDirectRDMAWrites` only if the device reports a weak write ordering. Everything is
  one process; no MPI.

Traffic never touches the inter-socket link: GPU → local NIC → fabric → peer's local NIC → peer GPU.

### 2.6 A barrier race in TRT-LLM's `block_barrier` (and how the NVFP4 kernels avoid it)

TRT-LLM's `block_barrier` has threads `tidx < world` publish the flag with `st.release.sys` **without a
preceding `__syncthreads()`**, so the other threads of the block may still have data stores in flight
when the flag becomes visible. On the 8× RTX 6000D (PCIe) this is observable: TRT-LLM's own BF16
one-shot gives rel_rmse 0.0078 / 0.0083 / 0.0094 on three consecutive 8M runs (its two-shot: 0.003960);
with `-DTRT_BARRIER_FIX` (adds the `__syncthreads()`) it is 0.003961 every time at unchanged speed. The
fused NVFP4 kernels inherited the pattern and showed it as grid-dependent error (one-shot 0.102 → 0.114
at 1024 blocks); their barrier now does `__syncthreads()` first, after which every grid gives bit-identical
results. Cost: none measurable. The verbatim TRT-LLM kernels in the bench are left as shipped (the flag
only affects them).

---

## 3. How to run

### 3.1 Build (explicit arch-specific gencode is mandatory)

```bash
# RTX 6000D / RTX PRO 6000 / RTX PRO 5000 (sm_120), 8 GPUs
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -DNRANKS=8 bench_ar.cu -o bench_ar
# B200 (sm_100), 8 GPUs — static binary for nodes without a toolkit
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -DNRANKS=8 -cudart static bench_ar.cu -o bench_ar_b200
# GB300 (sm_103, 4 GPUs/node, aarch64 — compile ON an aarch64 host/container)
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -DNRANKS=4 bench_ar.cu -o bench_ar_gb300        # single node, 4 GPUs
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -DNRANKS=8 bench_ar_mp.cu -o bench_ar_mp -lcuda # 2 nodes x 4 GPUs
```
`NRANKS` is the number of GPUs the binary drives (TRT-LLM kernels are templated on it). `-DNRANKS=8` is the default.

```bash
# PCIe transports (needs rdma-core headers; run as a user that can load nvidia-peermem or has dma-buf)
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -DNRANKS=8 bench_ar_pcie.cu -o bench_ar_pcie -lcuda -libverbs
sudo modprobe nvidia-peermem
AR_TRANSPORT=sm,ce,rdma NV_GRID=1 ./bench_ar_pcie <numel>     # RDMA_GID=<gid index>, NIC_MAP=2,3,0,1,6,7,4,5 to override
```

### 3.2 Run

```bash
./bench_ar <numel>                 # numel % (NRANKS*32) == 0
PHASE=1 ./bench_ar <numel>         # add per-phase timing (rank 0 stream) for every variant
NV_GRID=<blocks> ./bench_ar <numel>      # pin the fused two-shot NVFP4 grid (NV_GRID1 for the fused one-shot; NV_PCIE=1 sets both to 8)
TRT_FP8_GRID=16 ./bench_ar <numel>       # TRT-LLM's verbatim 16-block FP8 dispatch (PCIe design point)
```
Output: `correctness` (rel_rmse of every variant on rank 0), `ONE-SHOT`, `TWO-SHOT` and `FUSED`
lines (`ONE-SHOT-FUSED` for the one-shot kernel) with ratios, then `PHASES …` lines when `PHASE=1`.
`-DTRT_BARRIER_FIX` at build time adds the missing `__syncthreads()` to the verbatim TRT-LLM barrier (§2.5).

### 3.3 Recipes by machine type (exactly what produced the tables)

**A. Machine with a local CUDA toolkit and root (RTX 6000D, 8 GPUs)**
```bash
scp bench_ar.cu root@<host>:~/nvfp4_allreduce/
ssh root@<host> '
  cd ~/nvfp4_allreduce && pkill -9 bench_ar
  nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -DNRANKS=8 bench_ar.cu -o bench_ar
  MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
  nvidia-smi -pm 1; nvidia-smi -lgc $MAX,$MAX
  for N in 524288 8388608; do echo NUMEL=$N; PHASE=1 stdbuf -oL timeout 900 ./bench_ar $N; done
  for N in 32768 131072 524288 8388608; do NV_GRID=4 ./bench_ar $N | grep FUSED; done
  nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1; nvidia-smi -rgc'
```

**B. Slurm node without a toolkit and without sudo (RTX PRO 6000, 8 GPUs)**
```bash
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -DNRANKS=8 -cudart static bench_ar.cu -o bench_ar_static
scp bench_ar_static <slurm-login>:~/nvfp4_allreduce/
ssh <slurm-login> "srun -p <partition> --gres=gpu:8 --gpu-freq=high -t 0:40:0 bash -lc '
   cd ~/nvfp4_allreduce
   for N in 524288 8388608; do echo NUMEL=\$N; PHASE=1 stdbuf -oL timeout 900 ./bench_ar_static \$N; done
   nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1'"
```
`--gpu-freq` is slurm's gres clock lock — it works without sudo (`sudo nvidia-smi -lgc` does not).

**C. Slurm B200 node (8 GPUs, NVLink)**
```bash
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -DNRANKS=8 -cudart static bench_ar.cu -o bench_ar_b200
scp bench_ar_b200 <slurm-login>:~/nvfp4_allreduce/
ssh <slurm-login> "srun -A <account> --qos <qos> -p <partition> --gres=gpu:8 --gpu-freq=1965 -t 0:30:0 bash -lc '
   cd ~/nvfp4_allreduce
   for N in 33554432 134217728 536870912 2147483648; do echo NUMEL=\$N; PHASE=1 stdbuf -oL timeout 900 ./bench_ar_b200 \$N; done
   for N in 33554432 134217728 536870912; do for G in 256 512 1024 2048; do NV_GRID=\$G ./bench_ar_b200 \$N | grep FUSED; done; done
   nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1'"
```

**D. GB300, 8 GPUs = 2 nodes × 4 (aarch64) — multi-process harness, compile inside a CUDA container**
```bash
scp bench_ar_mp.cu <slurm-login>:~/nvfp4_allreduce/
cat > gb300_mp.sbatch <<'EOF'
#!/bin/bash
#SBATCH -A <account> -p <partition> -N 2 --ntasks-per-node=1 --gres=gpu:4 --switches=1 -t 0:45:00
#SBATCH -o nvfp4_mp8_%j.out -e nvfp4_mp8_%j.err
srun --gpu-freq=high \
  --container-image=nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc13 --container-mounts=$HOME/nvfp4_allreduce:/work \
  bash -c '
cd /work
ls /dev/nvidia-caps-imex-channels >/dev/null 2>&1 && echo IMEX_OK || echo NO_IMEX
if [ "$SLURM_PROCID" = 0 ]; then
  /usr/local/cuda/bin/nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -DNRANKS=8 bench_ar_mp.cu -o bench_ar_mp -lcuda
  touch /work/.built_$SLURM_JOB_ID
fi
while [ ! -f /work/.built_$SLURM_JOB_ID ]; do sleep 2; done
for N in 33554432 134217728 536870912 2147483648; do echo NUMEL=$N; stdbuf -oL timeout 600 ./bench_ar_mp $N; done
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1'
EOF
sbatch gb300_mp.sbatch
```
`--switches=1` keeps both nodes in one NVL72 domain on block-topology clusters; otherwise pin
`--nodelist` to two nodes of the same rack. `bench_ar_mp` derives `world = SLURM_NTASKS × visible GPUs`
and requires it to equal `NRANKS`, so 4 nodes × `--gres=gpu:2` also works. Handle exchange goes
through `/work/hs_<jobid>_<numel>/rank<r>.bin`; process 0 checks correctness and prints the table.
The GB300 node has no `nvcc`.

### 3.4 Using the standalone kernel

```cpp
#include "nvfp4_allreduce.cuh"
using namespace nvfp4_ar;
// ---- fused (recommended). Per rank, after cudaSetDevice(rank):
//   comm        : fused_comm_bytes(numel/RANKS, RANKS) bytes, peer-visible, zero-initialised once
//   barrier_in/out: fused_barrier_words(RANKS) uint32 each, peer-visible, zero-initialised once
//   peer_comm / peer_bar_in / peer_bar_out : device arrays[RANKS] of every rank's pointer
//   flag        : uint32, +1 per call (parity selects the comm column set)
launch_fused_rank<8>(input, output, peer_comm, peer_bar_in, peer_bar_out,
                     rank, numel, /*SFScaleVal=*/(6.f*448.f)/amax, flag, stream /*, TPB=256, grid=0 */);
// ---- fused one-shot: comm = oneshot_comm_bytes(numel, RANKS) bytes per rank, one barrier array
launch_oneshot_fused_rank<8>(input, output, peer_comm1, peer_bar1, rank, numel, (6.f*448.f)/amax, flag1, stream);
// ---- split (5 launches), flag consumes flag and flag+1:
launch_twoshot_rank(input, output, payload, reduced, peer_payloads, peer_reduced, barrier_sig,
                    world, rank, numel, (6.f*448.f)/amax, flag64, stream);
```
Launch all ranks before synchronizing any of them — the in-kernel barriers need every rank in
flight (same contract as TRT-LLM). `numel % (world*32) == 0`.

---

## 4. Results

Accuracy (rel_rmse vs FP32, identical on every card): BF16 0.004 · TRT-FP8 0.037 ·
NVFP4 one-shot 0.102 · NVFP4 two-shot split 0.1398 · NVFP4 fused 0.1398–0.1399.
(`myFP8` in the bench output is an internal control — the NVFP4 split path with an E4M3 payload — used
only for the phase analysis in §2.4; it is not a baseline and is not tabulated.)
All two-shot tables list TRT-LLM BF16, TRT-LLM FP8 and the fused NVFP4 kernel; block counts in
parentheses. The 5-launch NVFP4 split path (`launch_twoshot_rank`) is slower than fused everywhere and is
given as a one-line note per table.

### 4.1 PCIe — RTX 6000D, 8 GPUs, sm_120, locked 2347–2355 MHz (µs)

Columns: TRT-LLM kernels at their best grid (blocks in parentheses; native 16-block TRT-FP8 numbers are in the
sweep table), then NVFP4 with the three wire transports of §2.5 — `sm` = in-kernel peer stores (best grid),
`ce` = copy engines, `rdma` = GPUDirect RDMA through the host's NICs. Bold = fastest NVFP4; ratios use it.

**Two-shot**

| numel | TRT-BF16 | TRT-FP8 (blocks) | NVFP4 `sm` (blocks) | NVFP4 `ce` | NVFP4 `rdma` | NVFP4 vs TRT-FP8 | NVFP4 vs TRT-BF16 |
|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 32K | 99 | 215 (4) | **35** (1) | 551 | 228 | **6.1×** | 2.8× |
| 128K | 427 | 441 (1) | **95** (4) | 566 | 216 | **4.6×** | 4.5× |
| 512K | 2164 | 1077 (1) | 371 (2) | 546 | **221** | **4.9×** | 9.8× |
| 8M | 49833 | 14056 (1) | 7969 (1) | 1050 | **387** | **36×** | 129× |
| 32M | — | — | 32969 (1) | 3172 | **1017** | — | — |

**One-shot** (TRT-LLM has no FP8 one-shot; NVFP4 `sm` = the faster of the split and fused in-kernel variants)

| numel | TRT-BF16 | NVFP4 `sm` (variant) | NVFP4 `ce` | NVFP4 `rdma` | NVFP4 vs TRT-BF16 |
|---:|---:|---:|---:|---:|:--:|
| 32K | 569 | 110 (split) | 234 | **114** | **5.0×** |
| 128K | 2350 | 416 (split) | 227 | **127** | **18.5×** |
| 512K | 9866 | 2238 (fused, 1 blk) | 290 | **155** | **64×** |
| 8M | 158265 | 34342 (fused, 1 blk) | 3163 | **833** | **190×** |

Fastest NVFP4 all-reduce per size on this box: 32K two-shot `sm` 35 µs · 128K two-shot `sm` 95 µs · 512K
one-shot `rdma` 155 µs · 8M two-shot `rdma` 387 µs · 32M two-shot `rdma` 1017 µs.

**Grid sweep of the in-kernel variants** (µs; TRT-FP8 via `TRT_FP8_GRID`, NVFP4 fused via `NV_GRID`; TRT-FP8's
native dispatch is 16 blocks, the NVFP4 default rule gives 16 blocks below 8M and 128 at 8M):

| numel | kernel | 1 | 2 | 4 | 8 | 16 | 128 blocks |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32K | TRT-FP8 | 221 | 218 | **215** | 218 | 228 | |
| 32K | NVFP4 fused | **35** | 35 | 38 | 48 | 68 | |
| 128K | TRT-FP8 | **441** | 532 | 660 | 650 | 655 | |
| 128K | NVFP4 fused | 99 | 104 | **95** | 115 | 146 | |
| 512K | TRT-FP8 | **1077** | 1378 | 1786 | 1955 | 2099 | |
| 512K | NVFP4 fused | 397 | **371** | 407 | 464 | 475 | |
| 8M | TRT-FP8 | **14056** | 34089 | 36177 | 36086 | 35964 | 37093 (133) |
| 8M | NVFP4 fused | **7969** | 8150 | 8581 | 10050 | | 14479 |

Notes.
- In-kernel (`sm`) on this box: more than one block per GPU hurts both kernels — TRT-FP8 falls off a cliff
  between 1 and 2 blocks at 8M (2.4×) — most likely per-block `ld.acquire.sys` flag polling and interleaved
  peer writes competing on the PCIe links. TRT-LLM's fixed 16-block dispatch is 2.6× off its own optimum
  here; TRT-BF16 (64 blocks, not tunable without editing the kernel) very likely suffers the same way, so
  its column overstates the NVFP4 advantage. Below ~1M TRT-FP8 is slower than TRT-BF16 even at its best
  grid (preprocess pass), which is why TRT-LLM only enables it from 2M up.
- In-kernel one-shot: the fused kernel pushes 8 small packets per thread, so below ~256K the pull-based
  split version (110 / 416 µs) is faster; above it the 1-block fused kernel wins 2–2.5×.
- `rdma` has a ~200 µs host-orchestration floor (eight event syncs, launches, CQ polling), so below ~256K
  the in-kernel two-shot still wins. From 512K up RDMA is 1.8–32× faster than the best in-kernel variant.
  The one-shot over RDMA (one exchange, one host round-trip) is the fastest NVFP4 variant at 128K–512K;
  the two-shot RDMA overtakes it at 8M because it moves 3.5× fewer bytes.
- `ce` (56 concurrent `cudaMemcpyPeerAsync`) tops out at 9–10 GB/s per rank and never beats `rdma`.
- Accuracy: `ce` and `rdma` are bit-identical; `sm` differs in the 4th digit (reduction order).

### 4.2 PCIe — RTX PRO 6000 Blackwell SE, 8 GPUs, sm_120, locked 2295–2347 MHz (µs)

Same layout as §4.1 (sweep run on a second node at 2295 MHz; the default-grid fused column¹ is from the
first node at 2347 MHz, other columns agree within 2–6 % between the two).

| numel | TRT-BF16 | TRT-FP8 native (blocks) | **TRT-FP8 best** (blocks) | NVFP4 fused default¹ (blocks) | **NVFP4 fused best** (blocks) | NVFP4/BF16 | **NVFP4/TRT-FP8** |
|---:|---:|---:|---:|---:|---:|:--:|:--:|
| 32K | 72 | 136 (16) | **136** (16) | 51 (16) | **29** (1) | 2.5× | **4.8×** |
| 128K | 340 | 396 (16) | **289** (1) | 112 (16) | **80** (1) | 4.3× | **3.6×** |
| 512K | 1465 | 1206 (16) | **733** (1) | 447 (16) | **277** (2) | 5.3× | **2.6×** |
| 8M | 29635 | 20484 (16) | **9897** (1) | 9638 (128) | **4779** (2) | 6.2× | **2.1×** |

Grid sweep (µs):

| numel | kernel | 1 | 2 | 4 | 16 blocks |
|---:|---|---:|---:|---:|---:|
| 32K | TRT-FP8 | 137 | 143 | **136** | 136 |
| 32K | NVFP4 fused | **29** | 29 | 32 | 51¹ |
| 128K | TRT-FP8 | **289** | 309 | 393 | 396 |
| 128K | NVFP4 fused | **80** | 87 | 95 | 112¹ |
| 512K | TRT-FP8 | **733** | 811 | 994 | 1206 |
| 512K | NVFP4 fused | 279 | **277** | 348 | 447¹ |
| 8M | TRT-FP8 | **9897** | 12545 | 18443 | 20484 |
| 8M | NVFP4 fused | 5154 | **4779** | 5395 | 9638¹ (128) |

Same picture as the 6000D: TRT-FP8 at 1 block is 2.1× faster than at its fixed 16 (8M), both kernels want
1–2 blocks, and TRT-FP8 stays slower than TRT-BF16 below ~256K. This box is 1.4× faster than the 6000D at
8M for every kernel. TRT-LLM's BF16 one-shot showed the §2.5 race here too (rel_rmse 0.015 on one 512K run).

One-shot, same layout as §4.1 (fused swept over {1, 2, 4} blocks on this node; the default grid was not run here):

| numel | TRT-BF16 | NVFP4 split | NVFP4 fused best (blocks) | **NVFP4/BF16** (variant) |
|---:|---:|---:|---:|:--:|
| 32K | 359 | **66** | 106 (1) | **5.4×** (split) |
| 128K | 1541 | **239** | 410 (1) | **6.4×** (split) |
| 512K | 6771 | 2415 | **1606** (1) | **4.2×** (fused) |
| 8M | 117766 | 46015 | **24848** (1) | **4.7×** (fused) |

Two-shot split path: 91 / 148 / 571 / 11145 µs.

### 4.3 NVLink — B200, 8 GPUs, sm_100, locked 1965 MHz (µs)

Two-shot. TRT-FP8 with the grid scaled for NVLink (§2.1; 16 below 2M is also its native value).
NVFP4 fused with the default grid rule (§2.2).

| numel | TRT-BF16 | TRT-FP8 (blocks) | **NVFP4 fused** (blocks) | NVFP4/BF16 | **NVFP4/TRT-FP8** |
|---:|---:|---:|---:|:--:|:--:|
| 32K | 29 | 45 (16) | **32** (16) | 0.92× | **1.41×** |
| 128K | 31 | 45 (16) | **34** (16) | 0.92× | **1.34×** |
| 512K | 34 | 44 (16) | **34** (16) | 0.99× | **1.30×** |
| 2M | 54 | 48 (34) | **36** (32) | 1.48× | **1.32×** |
| 8M | 167 | 61 (133) | **43** (128) | 3.9× | **1.41×** |
| 32M | 651 | 176 (529) | **94** (256) | 7.0× | **1.88×** |
| 128M | 2671 | 581 (2048) | **303** (512) | 8.8× | **1.92×** |
| 512M | 10665 | 1861 (2048) | **1037** (2048) | 10.3× | **1.80×** |
| 2G | 44304 | 7216 (2048) | **3776** (2048) | 11.7× | **1.91×** |

One-shot (NVFP4 split = 3 launches; NVFP4 fused = one PUSH kernel; default grid and best grid from the sweep
over {16, 64, 256, 1024, 2048} blocks; ratio uses the best):

| numel | TRT-BF16 | NVFP4 split | NVFP4 fused default (blocks) | **NVFP4 fused best** (blocks) | **NVFP4/BF16** |
|---:|---:|---:|---:|---:|:--:|
| 32K | 18 | 54 | 16.1 (16) | **16.1** (16) | **1.12×** |
| 128K | 18 | 54 | 16.4 (16) | **16.1** (16) | **1.12×** |
| 512K | 32 | 53 | 16.2 (64) | **16.2** (64) | **2.0×** |
| 2M | 91 | 55 | 31 (256) | **31** (256) | **2.9×** |
| 8M | 445 | 79 | 81 (256) | **79** (1024) | **5.6×** |
| 32M | 1771 | 252 | 244 (1024) | **240** (2048) | **7.4×** |
| 128M | 7033 | 911 | 895 (2048) | **895** (2048) | **7.9×** |
| 512M | 28134 | 3515 | 3533 (2048) | **3528** (2048) | **8.0×** |
| 2G | 112747 | 14182 | 14078 (2048) | **14078** (2048) | **8.0×** |

Fused one-shot grid sweep (µs): 512K — 16 blk 23.3 · 64 **16.2** · 256 20.5 · 1024 29.2 · 2048 46.2; 8M — 16 blk 203 ·
64 101 · 256 81 · 1024 **79** · 2048 93; 32M — 64 blk 408 · 256 283 · 1024 245 · 2048 **240**. The one-shot pushes
RANKS copies, so it tolerates more blocks than the two-shot; the default rule is within 3 % of the best everywhere.

Below ~1M elements everything sits on the NVLink latency floor: a one-shot (one barrier) at ~16 µs, a
two-shot (two barriers) at ~30 µs. There the fused NVFP4 one-shot is the fastest kernel of all (1.1× over
TRT-LLM's BF16 one-shot), the fused two-shot ties TRT-BF16 two-shot and beats TRT-FP8 by 1.3–1.4×. From
2M up the fused two-shot is the fastest all-reduce and the fused one-shot the fastest one-shot.
Two-shot split path: 81 / 81 / 83 / 82 / 81 / 116 / 329 / 1160 / 4479 µs (launch gaps below 8M, 5–18 % above).

Phases at 512M (`PHASE=1`, rank 0, µs): NVFP4 split quant 271 · bar 19 · RS 428 · bar 9 · AG 454 (Σ 1180);
NVFP4 fused 1045; TRT-FP8 prep 286 + fused 1594 (Σ 1880); TRT-BF16 10641.

### 4.4 NVLink — GB300 NVL72, 8 GPUs = 2 nodes × 4, sm_103, 2070 MHz (µs)

`bench_ar_mp` (fabric handles + IMEX), same layout as §4.3. One job on one NVL72 rack; earlier runs on
two other racks agreed within 1–3 % at every size.

| numel | TRT-BF16 | TRT-FP8 (blocks) | **NVFP4 fused** (blocks) | NVFP4/BF16 | **NVFP4/TRT-FP8** |
|---:|---:|---:|---:|:--:|:--:|
| 32K | 28 | 42 (16) | **33** (16) | 0.86× | **1.26×** |
| 128K | 31 | 43 (16) | **34** (16) | 0.91× | **1.27×** |
| 512K | 34 | 42 (16) | **35** (16) | 0.97× | **1.21×** |
| 2M | 55 | 45 (34) | **37** (32) | 1.48× | **1.20×** |
| 8M | 172 | 58 (133) | **44** (128) | 3.9× | **1.32×** |
| 32M | 662 | 172 (529) | **99** (256) | 6.7× | **1.74×** |
| 128M | 2725 | 566 (2048) | **318** (512) | 8.6× | **1.78×** |
| 512M | 10883 | 1810 (2048) | **1086** (2048) | 10.0× | **1.67×** |
| 2G | 44365 | 7025 (2048) | **3974** (2048) | 11.2× | **1.77×** |

One-shot (NVFP4 split = 3 launches; NVFP4 fused = one PUSH kernel; default grid and best grid from the sweep
over {16, 64, 256, 1024, 2048} blocks; ratio uses the best):

| numel | TRT-BF16 | NVFP4 split | NVFP4 fused default (blocks) | **NVFP4 fused best** (blocks) | **NVFP4/BF16** |
|---:|---:|---:|---:|---:|:--:|
| 32K | 12 | 30 | 12.6 (16) | **12.8** (16) | **0.94×** |
| 128K | 14 | 31 | 12.5 (16) | **12.4** (16) | **1.14×** |
| 512K | 30 | 31 | 15.1 (64) | **14.7** (64) | **2.1×** |
| 2M | 89 | 38 | 30.4 (256) | **30.2** (256) | **2.9×** |
| 8M | 432 | 79 | 80 (256) | **79** (1024) | **5.5×** |
| 32M | 1721 | 248 | 243 (1024) | **239** (2048) | **7.2×** |
| 128M | 6859 | 902 | 890 (2048) | **890** (2048) | **7.7×** |
| 512M | 27455 | 3505 | 3509 (2048) | **3509** (2048) | **7.8×** |
| 2G | 109829 | 13891 | 13975 (2048) | **13975** (2048) | **7.9×** |

Fused one-shot grid sweep (µs): 512K — 16 blk 21.6 · 64 **14.7** · 256 19.7 · 1024 29.0 · 2048 46.1; 8M — 16 blk 200 ·
64 98 · 256 80 · 1024 **79** · 2048 93; 32M — 64 blk 395 · 256 279 · 1024 243 · 2048 **239**; 128M — 256 blk 1062 ·
1024 914 · 2048 **890**; 512M — 256 blk 4192 · 1024 3605 · 2048 **3509** (2G not swept; 2048 is the default and the
trend from 32M up). Same shape as B200: the default rule is within 3 % of the best everywhere.

The fused one-shot ties TRT-LLM's BF16 one-shot on the ~12 µs latency floor and wins from 128K up; it is
the fastest NVFP4 variant below ~1M (one barrier instead of two), the fused two-shot above ~8M.

Cross-node (2 × 4) NVLink tracks single-node B200 within a few percent at every size. Two-shot split path:
42 / 42 / 42 / 46 / 60 / 113 / 324 / 1142 / 4374 µs. The 4-GPU single-node GB300 run of the first version
(NVFP4/TRT-FP8 1.13 … 0.83×) is superseded.

### 4.5 Fused-kernel grid sweep (B200, 8 GPUs, µs)

| numel | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 blocks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 32 (4 blk: 30) | | | | | | | |
| 512K | **34** | | 34 | | 39 | | | |
| 2M | 36 | **36** | 37 | 37 | 41 | 45 | | |
| 8M | 166 | 94 | 59 | **43** | 48 | 61 | | |
| 32M | | | 184 | 119 | **95** | 107 | 132 | 193 |
| 128M | | | | | **292** | 303 | 323 | 360 |
| 512M | | | | | 1073 | 1065 | 1073 | **1032** |

Up to 32M the best grid is "one chunk per block" (≈ shard / 8192) capped at 256 — more blocks add
barrier traffic (blocks × ranks flag stores), fewer leave bandwidth idle; from 512M the full 2048 wins.
That is the default rule; it is within 4 % of the best at every size above. On PCIe the optimum is
4 blocks regardless of size (§4.1, §4.2).

### 4.6 PCIe wire transports — what the NIC path actually delivers (RTX 6000D)

Host: 2 sockets, GPUs 0–3 / 4–7 on different NUMA nodes, one 400G RoCE port per GPU on its PCIe bridge.
Raw link probes: `ib_write_bw` GPU0→GPU4 (cross-NUMA, GPUDirect) 45.5 GB/s; `cudaMemcpyPeerAsync` GPU0→GPU4
50.5 GB/s; the in-kernel all-reduces of §4.1 move ~1 GB/s per rank. Achieved all-reduce wire bandwidth per
rank (wire bytes: two-shot 2 × 7/8 × 0.5625 × numel, one-shot 7 × 0.5625 × numel; times from §4.1):

| numel | two-shot `sm` | two-shot `ce` | **two-shot `rdma`** | one-shot `rdma` |
|---:|---:|---:|---:|---:|
| 32K | 0.9 GB/s | 0.1 | 0.1 | 1.1 |
| 128K | 1.3 | 0.2 | 0.6 | 4.1 |
| 512K | 1.3 | 0.9 | 2.3 | 13.4 |
| 8M | 1.0 | 7.9 | **21.4** | **39.7** |
| 32M | 1.0 | 10.4 | **32.5** | — |

RDMA two-shot phase breakdown at 8M (host clock, µs): quantize+sync 62 · RS writes+CQ 122 · reduce+sync 45 ·
AG writes+CQ 122 · dequantize+sync 33 — the two write phases run at ~40 GB/s, everything else is the
~200 µs host-orchestration floor. One-shot at 8M: quantize+sync 58 · writes+CQ 736 · reduce+sync 37.
Traffic never crosses the inter-socket link (GPU → local NIC → fabric → peer NIC → peer GPU), so NUMA
placement stops mattering; `nvidia-peermem` was loaded for MR registration (dma-buf is the fallback).

### 4.7 Codec ablation (first version; what the codec did and did not change)

| card | config | one-shot | two-shot |
|---|---|---:|---:|
| B200 512M | v0 local-memory LUT decode + branchy encode | 19937 µs | 5996 µs |
| B200 512M | v1 PRMT decode + hw encode | 3862 | 2671 |
| B200 512M | v2 hw decode + hw encode | 3846 | 2672 |
| 6000D 512K | PRMT decode | 3893 | 1111 |
| 6000D 512K | hw decode (sm_120) | 5342 (**+37 %**) | 1258 (+13 %) |

A runtime-indexed float table lands in local memory and made the kernel compute-bound; any register-only
decode (PRMT or hardware cvt) fixes that, and after it the access pattern (§2.4) is what matters.

---

## 5. Conclusions

1. **vs TRT-LLM BF16 custom AR: NVFP4 fused wins 3.9–11.8× on NVLink from 8M up and 2.3–6.3× on PCIe**
   (two-shot, 8 GPUs, PCIe with the best grid); below ~1M on NVLink both sit on the ~30 µs latency floor and tie.
   Caveat: TRT-BF16's 64-block grid is not tunable and likely suffers the same PCIe multi-block penalty as TRT-FP8 (§4.1).
2. **vs TRT-LLM's real FP8 kernel (fair grid): NVFP4 fused wins at every size on every card** — B200 1.29–1.91× (32K–2G),
   GB300 2×4 1.16–1.78× (32K–2G), RTX 6000D 1.8–6.1× and RTX PRO 6000 2.1–4.8× with both kernels at their best PCIe grid (TRT-FP8 at 1 block is 2.1–2.6× faster than its own fixed 16 at 8M). The byte ratio
   is 1.89×; at ≥128M on NVLink the kernel realises 1.8–1.9× of it.
3. **Follow TRT-LLM's access pattern, not just its algorithm.** The first version kept the two-shot
   algorithm but used an unrotated owner order and 4-byte loads and lost 17–31 % to TRT-FP8 on
   NVLink; per-phase timing showed the all-gather at 200 GB/s and a 546 µs barrier skew. Rank-rotated
   peer order and 16-byte accesses — both taken from `twoShotAllReduceKernel` — turned that into a
   1.5–1.75× win; fusing into one kernel with TRT-LLM's `block_barrier` added another 7–18 % (NVLink)
   and 1.5–2× (PCIe).
4. Only the verbatim TRT-LLM FP8 kernel is a valid baseline: a per-16-scale FP8 re-implementation with
   the NVFP4 split structure (`myFP8`, kept in the bench as a control) is 1.1–2.2× slower than TRT-FP8 and
   would have overstated NVFP4 by the same factor.
5. Accuracy cost: rel_rmse 0.14 (NVFP4 two-shot, two quantization passes) / 0.10 (one-shot) vs 0.037 (FP8) vs 0.004 (BF16).
7. TRT-LLM's `block_barrier` publishes its flag without a preceding `__syncthreads()`; on PCIe with many blocks this is a real race (its BF16 one-shot is non-deterministic there). The NVFP4 kernels add the `__syncthreads()`; `-DTRT_BARRIER_FIX` shows the fix on the verbatim kernels (§2.5).
6. The E2M1 hardware `cvt` exists on every Blackwell including sm_120; build with explicit `-gencode …a`.
8. **On PCIe hosts the wire, not the format, is the bottleneck.** SM stores across the root complex give ~1 GB/s per rank for every in-kernel all-reduce (ours and TRT-LLM's). Moving the NVFP4 slices with GPUDirect RDMA through the host's own NICs (FlashInfer PR #4876's recipe) reaches 32 GB/s per rank: 8M in 387 µs vs 7969 µs in-kernel, 14056 µs TRT-FP8 and 49559 µs TRT-BF16 (§4.6). The ~200 µs host-orchestration floor leaves the in-kernel kernel the winner below ~256K.
