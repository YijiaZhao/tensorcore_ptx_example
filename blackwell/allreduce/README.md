# NVFP4-compressed all-reduce vs TRT-LLM custom all-reduce (Blackwell)

An all-reduce that moves NVFP4 (E2M1 + per-16 UE4M3 block scale, 0.5625 B/elt) over the wire
instead of BF16 (2 B/elt) or FP8 (TRT-LLM low-precision AR, ~1.06 B/elt), built on the exact
kernel structure of TRT-LLM's `twoShotAllReduceKernel`, and benchmarked against **TRT-LLM's own
kernels compiled verbatim into the same binary**:

- `TRT-BF16` — TRT-LLM `customAllReduceKernels.cu` one-shot / two-shot (PUSH_MODE, `block_barrier`).
- `TRT-FP8`  — TRT-LLM `customLowPrecisionAllReduceKernels.cu` two-shot (preprocess + fused kernel).
- `myFP8`    — a per-16-scale FP8 re-implementation with the *same* structure as the NVFP4 split path
  (structural twin, **not** a baseline — TRT-FP8 beats it 1.1–2.2×).
- `NVFP4`    — this work: split (5 launches) and **fused** (one kernel, TRT-LLM structure) variants.

Files:

| file | what |
|---|---|
| `nvfp4_allreduce.cuh` | standalone header: `launch_fused_rank<RANKS>()` (recommended) and `launch_twoshot_rank()` (split) |
| `bench_ar.cu` | single-process harness, 8 GPUs in one node; all baselines + NVFP4; `PHASE=1` per-phase timing |
| `bench_ar_mp.cu` | multi-process harness (one process per node, fabric handles + IMEX) for 2-node GB300 |

Headline (two-shot, 8 GPUs, clocks locked, µs; **fused NVFP4 vs TRT-LLM FP8**):

| GPU | interconnect | 32M / 512K | 128M / 8M | 512M | 2G |
|---|---|---|---|---|---|
| B200 ×8 | NVLink | **1.49×** | **1.91×** | **1.80×** | **1.91×** |
| GB300 2×4 (NVL72) | NVLink cross-node | **1.37×** | **1.78×** | **1.67×** | **1.78×** |
| RTX 6000D ×8 (PCIe) | PCIe | **4.48×** (512K) | **2.72×** (8M) | — | — |
| RTX PRO 6000 ×8 (PCIe) | PCIe | see §4.2 | see §4.2 | — | — |

vs TRT-LLM BF16 custom AR the fused NVFP4 kernel is 5.5–11.8× faster on NVLink and 3.7–4.6× on PCIe.
Accuracy cost: rel_rmse 0.14 (two quantization passes) vs 0.037 (FP8) vs 0.004 (BF16).

---

## 1. Test environment

| GPU | sm | GPUs | interconnect | build toolchain | clock lock (verified by `nvidia-smi --query-gpu=clocks.sm` after each run) |
|---|---|---|---|---|---|
| RTX 6000D (cc 12.0) | sm_120 | 8 | PCIe | CUDA 13.3 nvcc, local | `nvidia-smi -lgc <max>` → 2347 MHz |
| RTX PRO 6000 Blackwell Server Edition | sm_120 | 8 | PCIe | static binary (`-cudart static`) built with CUDA 13.3 on an x86 host; no toolkit on the node | slurm `--gpu-freq=high` → 2422 MHz |
| B200 (cc 10.0) | sm_100 | 8 | NVLink | static binary built with CUDA 13.3 on an x86 host | slurm `--gpu-freq=1965` → 1965 MHz |
| GB300 NVL72 (cc 10.3, aarch64) | sm_103 | 8 = **2 nodes × 4** | NVLink (intra-rack, cross-node) | built inside `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` (CUDA 13.1, aarch64); no toolkit on the node | `--gpu-freq=high` requested (slurm reported `control_disabled`); read back 2070 MHz = GB300 max SM clock |

- Timing: CUDA events on rank 0; 20 warm-up + 100 timed iterations (5 + 20 for ≥128M elements). µs per all-reduce.
- Accuracy: rel_rmse vs an FP32 host reference of the same random inputs (fixed seed); NVFP4 two-shot is bit-reproducible run to run.
- Harness `bench_ar.cu`: one process, `cudaDeviceEnablePeerAccess` + raw peer pointers (no NCCL/MPI).
  `bench_ar_mp.cu`: one process per node, buffers from `cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC)`, handles
  exchanged through files on the shared `$HOME`, remote ranks mapped with `cuMemImportFromShareableHandle`
  + `cuMemMap`; needs IMEX (`/dev/nvidia-caps-imex-channels`) and both nodes in one NVL72 domain.
- GB300 8-GPU numbers were reproduced on two different NVL72 racks (different sites) within 1 %.
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
column set so back-to-back calls never overwrite each other (same trick as TRT-LLM). Grid default:
~4 chunks of `TPB·32` elements per thread, clamped to [16, 2048] (see the sweep in §4.5).

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
  while the reduce-scatter (8 independent loads per thread) reached ~530 GB/s; myFP8's all-gather read
  twice the bytes in the same time, so it was requests, not bandwidth. TRT-LLM uses 16-byte int4
  loads; switching to 32 elements per thread (16 B packed + 2 B scales) took the all-gather from 1331
  to 454 µs.
- Fusing into one kernel with TRT-LLM's `block_barrier` and PUSH reduce-scatter saves the separate
  quantize pass and the two barrier launches: another 7–18 % on NVLink, 1.5–2× on PCIe.

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

### 3.2 Run

```bash
./bench_ar <numel>                 # numel % (NRANKS*32) == 0
PHASE=1 ./bench_ar <numel>         # add per-phase timing (rank 0 stream) for every variant
NV_GRID=<blocks> ./bench_ar <numel>      # pin the fused NVFP4 grid (default: ~4 chunks/thread, [16,2048])
TRT_FP8_GRID=16 ./bench_ar <numel>       # TRT-LLM's verbatim 16-block FP8 dispatch (PCIe design point)
```
Output: `correctness` (rel_rmse of every variant on rank 0), `ONE-SHOT`, `TWO-SHOT` and `FUSED`
lines with ratios, then `PHASES …` lines when `PHASE=1`.

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
  for G in 8 32 128; do NV_GRID=$G ./bench_ar 8388608 | grep FUSED; done
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
// ---- split (5 launches), flag consumes flag and flag+1:
launch_twoshot_rank(input, output, payload, reduced, peer_payloads, peer_reduced, barrier_sig,
                    world, rank, numel, (6.f*448.f)/amax, flag64, stream);
```
Launch all ranks before synchronizing any of them — the in-kernel barriers need every rank in
flight (same contract as TRT-LLM). `numel % (world*32) == 0`.

---

## 4. Results

Accuracy (rel_rmse vs FP32, identical on every card): BF16 0.004 · TRT-FP8 0.037 · myFP8 0.036 ·
NVFP4 one-shot 0.102 · NVFP4 two-shot split 0.1398 · NVFP4 fused 0.1398–0.1399.
TRT-FP8 grid is the scaled one (§2.1) unless marked `g16`; the fused NVFP4 grid is the default rule
unless marked.

### 4.1 PCIe — RTX 6000D, 8 GPUs, sm_120, locked 2347 MHz (µs)

| numel | shape | TRT-BF16 | TRT-FP8 | myFP8 | NVFP4 split | **NVFP4 fused** | fused/BF16 | **fused/TRT-FP8** | split/TRT-FP8 |
|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 512K | two-shot | 2147 | 2097 (g16) | 1884 | 900 | **467** (g16) | 4.6× | **4.48×** | 2.33× |
| 8M | two-shot | 49800 | 36916 (g133) | 23265 | 20264 | **13561** (g32) | 3.7× | **2.72×** | 1.82× |
| 8M | two-shot, `NV_GRID=8` | | | | | **9937** | 5.0× | **3.73×** | |
| 512K | one-shot | 9848 | — | 10319 | **3772** | | 2.6× | 2.7× vs myFP8 | |
| 8M | one-shot | 158205 | — | 166730 | **92070** | | 1.7× | 1.8× vs myFP8 | |

Fused grid on PCIe (8M): 8 blocks 9937 · 32 blocks 13600 · 128 blocks 14479 µs — PCIe wants few
blocks, exactly TRT-LLM's PCIe design point (16). Split-path phases at 8M: RS 10289 / AG 9446 µs; the
barriers absorb 300–600 µs of rank skew on PCIe.

### 4.2 PCIe — RTX PRO 6000 Blackwell SE, 8 GPUs, sm_120, locked 2422 MHz (µs)

_(pending — being re-run with the final kernels; the previous-version numbers were NVFP4 split
1249 / 25872 µs vs TRT-FP8 1881 / 30411 at 512K / 8M, i.e. 1.51× / 1.18×.)_

### 4.3 NVLink — B200, 8 GPUs, sm_100, locked 1965 MHz (µs), two-shot

| numel | TRT-BF16 | TRT-FP8 (grid) | myFP8 | NVFP4 split | **NVFP4 fused** (grid) | fused/BF16 | **fused/TRT-FP8** | split/TRT-FP8 |
|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 32M | 652 | 176 (529) | 196 | 117 | **119** (128) | 5.5× | **1.49×** | 1.50× |
| 128M | 2668 | 580 (2048) | 648 | 332 | **304** (512) | 8.8× | **1.91×** | 1.75× |
| 512M | 10652 | 1863 (2048) | 2507 | 1160 | **1032** (2048) | 10.3× | **1.80×** | 1.61× |
| 2G | 44417 | 7224 (2048) | 9854 | 4462 | **3780** (2048) | 11.8× | **1.91×** | 1.62× |

One-shot, B200 (BF16 / myFP8 / **NVFP4**): 32M 1767 / 456 / **281**; 128M 7025 / 1682 / **1001**;
512M 28072 / 6596 / **3852**; 2G 112445 / 26279 / **15323** — NVFP4/myFP8 = 1.62–1.71× at every size
(byte ratio 1.89×; TRT-LLM has no FP8 one-shot).

Phases at 512M (`PHASE=1`, rank 0, µs): NVFP4 split quant 271 · bar 19 · RS 428 · bar 9 · AG 454 (Σ 1180);
NVFP4 fused 1045; TRT-FP8 prep 286 + fused 1594 (Σ 1880); TRT-BF16 10641.

### 4.4 NVLink — GB300 NVL72, 8 GPUs = 2 nodes × 4, sm_103, 2070 MHz (µs), two-shot

`bench_ar_mp` (fabric handles + IMEX). 2G row from the same-configuration run on the first rack
(grid 2048 in both grid rules); the other rows from the second rack with the final grid rule.

| numel | TRT-BF16 | TRT-FP8 (grid) | myFP8 | NVFP4 split | **NVFP4 fused** (grid) | fused/BF16 | **fused/TRT-FP8** | split/TRT-FP8 |
|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 32M | 664 | 172 (529) | 193 | 113 | **125** (128) | 5.3× | **1.37×** | 1.52× |
| 128M | 2736 | 566 (2048) | 632 | 323 | **318** (512) | 8.6× | **1.78×** | 1.76× |
| 512M | 10937 | 1809 (2048) | 2431 | 1141 | **1083** (2048) | 10.1× | **1.67×** | 1.59× |
| 2G | 44755 | 7035 (2048) | 9485 | 4374 | **3956** (2048) | 11.3× | **1.78×** | 1.61× |

One-shot, GB300 8 GPUs (BF16 / myFP8 / **NVFP4**): 32M 1700 / 465 / **273**; 128M 6774 / 1692 / **987**;
512M 27090 / 6584 / **3836**; 2G 109675 / 26177 / **15231**.

Cross-node NVLink numbers track single-node B200 within a few percent at every size; the 4-GPU
single-node GB300 run of the first version (NVFP4/TRT-FP8 1.13 … 0.83×) is superseded.

### 4.5 Fused-kernel grid sweep (B200, 8 GPUs, µs)

| numel | 64 | 128 | 256 | 512 | 1024 | 2048 blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 32M | 184 | 119 | **95** | 107 | 132 | 193 |
| 128M | | | **292** | 303 | 323 | 360 |
| 512M | | | 1073 | 1065 | 1073 | **1032** |

Small messages want ~256 blocks (barrier traffic scales with blocks × ranks); ≥512M wants the full
2048. The default rule (~4 chunks per thread, clamped to [16, 2048]) lands within 4 % of the best at
128M/512M/2G and 25 % off at 32M; pass `grid` (header) / `NV_GRID` (bench) to pin it. On PCIe use 8–16.

### 4.6 Codec ablation (first version; what the codec did and did not change)

| card | config | one-shot | two-shot |
|---|---|---:|---:|
| B200 512M | v0 local-memory LUT decode + branchy encode | 19937 µs (0.33× vs myFP8) | 5996 µs (0.68×) |
| B200 512M | v1 PRMT decode + hw encode | 3862 (1.71×) | 2671 (1.50×) |
| B200 512M | v2 hw decode + hw encode | 3846 | 2672 |
| 6000D 512K | PRMT decode | 3893 | 1111 |
| 6000D 512K | hw decode (sm_120) | 5342 (**+37 %**) | 1258 (+13 %) |

A runtime-indexed float table lands in local memory and made the kernel compute-bound; any register-only
decode (PRMT or hardware cvt) fixes that, and after it the access pattern (§2.4) is what matters.

---

## 5. Conclusions

1. **vs TRT-LLM BF16 custom AR: NVFP4 fused wins 5.5–11.8× on NVLink and 3.7–4.6× on PCIe** (two-shot,
   8 GPUs); 6–7× one-shot on NVLink.
2. **vs TRT-LLM's real FP8 kernel (fair grid): NVFP4 fused wins everywhere** — B200 1.49–1.91×,
   GB300 2×4 1.37–1.78×, RTX 6000D 2.7–4.5× (up to 3.7× at 8M with an 8-block grid). The byte ratio
   is 1.89×; at ≥128M on NVLink the kernel realises 1.8–1.9× of it.
3. **Follow TRT-LLM's access pattern, not just its algorithm.** The first version kept the two-shot
   algorithm but used an unrotated owner order and 4-byte loads and lost 17–31 % to TRT-FP8 on
   NVLink; per-phase timing showed the all-gather at 200 GB/s and a 546 µs barrier skew. Rank-rotated
   peer order and 16-byte accesses — both taken from `twoShotAllReduceKernel` — turned that into a
   1.5–1.75× win; fusing into one kernel with TRT-LLM's `block_barrier` added another 7–18 % (NVLink)
   and 1.5–2× (PCIe).
4. Only the verbatim TRT-LLM FP8 kernel is a valid baseline: the per-16-scale re-implementation
   (myFP8) is 1.1–2.2× slower than it and would have overstated NVFP4 by the same factor.
5. Accuracy cost: rel_rmse 0.14 (NVFP4, two quantization passes) vs 0.037 (FP8) vs 0.004 (BF16).
6. The E2M1 hardware `cvt` exists on every Blackwell including sm_120; build with explicit `-gencode …a`.
