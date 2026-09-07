# NVFP4 All-Reduce on Blackwell — vs TRT-LLM custom All-Reduce (BF16 / FP8)

An NVFP4 (4-bit E2M1) compressed all-reduce implemented on **TRT-LLM's own custom-AR algorithm**
(one-shot / two-shot reduce-scatter + all-gather), benchmarked in one identical single-process
P2P harness against the **verbatim TRT-LLM kernels**: its BF16 custom AR and its FP8
low-precision AR. Five Blackwell GPUs (sm_100 / sm_103 / sm_120), PCIe and NVLink, clocks locked.

The only thing that differs between columns is the wire payload:
**BF16 2.0 B/elt · FP8 ~1.06 B/elt · NVFP4 0.5625 B/elt** (E2M1 pairs + per-16 UE4M3 block scale + one FP32 tensor scale).

```
blackwell/allreduce/
├── nvfp4_allreduce.cuh   reusable NVFP4 two-shot all-reduce (kernels + host launcher), self-contained
├── bench_ar.cu           the benchmark: TRT-LLM BF16 (verbatim) / TRT-LLM FP8 (verbatim) / myFP8 / NVFP4
└── README.md             this file
```

---

## 1. Test environment

| machine | GPU | sm | GPUs used | interconnect | build toolchain | clock lock (verified) | how reached |
|---|---|---|---|---|---|---|---|
| `root@10.6.142.1` | RTX 6000D (cc 12.0) | sm_120 | 8 | PCIe | CUDA 13.3 nvcc, local | `nvidia-smi -lgc <max>` → ~2.35 GHz | direct root ssh |
| computelab-aus-01 slurm, partition `rtx-pro-6000-blackwell-server-edition@cr+mp/genoa2d24g2l/8gpu-256cpu-2304gb` | RTX PRO 6000 Blackwell Server Edition | sm_120 | 8 | PCIe | no toolkit on node → static binary built on 142.1 (CUDA 13.3, `-cudart static`) | `srun --gpu-freq=high` → **2422 MHz** | `ssh computelab-aus-01`, `srun --gres=gpu:8` |
| nsc-svg-slurm-1 (AI Hub / MARS, Norway), partition `batch` | B200 (cc 10.0) | sm_100 | 8 | NVLink | static binary built on 142.1 (CUDA 13.3) | `srun --gpu-freq=1965` → **1965 MHz** | `ssh -J computelab-sc-01 kimiz@nsc-svg-slurm-1-login-02.nvidia.com`, account `coreai_comparch_inferencex`, qos `normal` |
| dlcluster, partition `gb300nvl72_preprod` | GB300 (cc 10.3, aarch64) | sm_103 | **4** (4 GPU/node) | NVLink | no toolkit on node → built inside pyxis container `nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc13` (CUDA 13.1, aarch64) | `srun --gpu-freq=high` → **2070 MHz** | `ssh dlcluster`, account `blackwell`, `--gres=gpu:GB300:4` |
| `root@10.6.142.14` (early data only) | RTX PRO 5000 Blackwell 72GB | sm_120 | 8 | PCIe, 2 NUMA (0–3 / 4–7, cross = SYS) | CUDA 13.0 nvcc, local | `-lgc` → ~2.6 GHz (power-capped, not pinned) | direct root ssh — **shared box, do not use for benchmarks** |

- Harness: single process, `cudaDeviceEnablePeerAccess` + raw peer pointers (no NCCL, no MPI).
- Timing: CUDA events on rank 0; 20 warm-up + 100 timed iterations (5 + 20 for ≥128M elements). Numbers are µs per all-reduce.
- Clocks were read back with `nvidia-smi --query-gpu=clocks.sm` after every run; the MHz above are those readings.
- Accuracy: rel_rmse vs an FP32 host reference of the same random inputs (fixed seed).
- Driver versions were not recorded.
- GB300 nodes have 4 GPUs; an 8-GPU GB300 run would need a multi-process (IMEX/fabric-handle) harness, which this single-process code is not.

## 2. What is being compared

**TRT-LLM BF16 custom AR (verbatim).** `oneShotAllReduceKernel` / `twoShotAllReduceKernel` from
`cpp/tensorrt_llm/kernels/customAllReduceKernels.cu`: PUSH_MODE, fused flag `block_barrier`
(one launch per all-reduce), 128-bit `add128b`.

**TRT-LLM FP8 low-precision AR (verbatim).** `lowPrecisionPreprocessKernel` +
`lowPrecisionTwoShotAllReduceKernel` (first/second stage) from
`cpp/tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.cu`.
Per-warp FP32 scale carried in-stream (31 lanes × 16 fp8 = 496 data elements + 1 scale slot per
512), dequant → FP32 reduce → requant once in place → all-gather + dequant, PULL mode, one
fused kernel after a separate quantize launch. TRT-LLM's dispatch hard-codes **grid = 16 blocks ×
512 threads** and only enables this kernel on PCIe machines with ≤4 ranks (8 ranks use a
NUMA-hierarchical variant that is PCIe-2-NUMA specific and was not ported). Because 16 blocks
cannot saturate NVLink, the benchmark also reports the same kernel with the grid scaled to one
block per 7936-element round (`TRT_FP8_GRID` env var pins any grid; `=16` is the verbatim dispatch).

**myFP8.** A per-16 UE4M3 block-scale FP8 with the NVFP4 layout. Kept only to prove the FP8
reference is not weak; TRT-FP8 matches it on PCIe and beats it 1.6–2.3× on NVLink.

**NVFP4 (this work).** TRT-LLM's two-shot (and one-shot) structure with an NVFP4 payload:
quantize (per-16 UE4M3 block scale + FP32 tensor scale) → barrier → reduce-scatter (dequant → FP32
accumulate → requantize once into a separate reduced buffer) → barrier → all-gather (dequant → BF16).
Five launches per all-reduce (this is the structural gap to TRT-FP8's fused kernel — see §5).

### NVFP4 codec (decides the result)

| version | decode | encode | per 8 elts | outcome |
|---|---|---|---|---|
| v0 | `lut[n&7]` — 8-entry float table indexed at runtime → local memory, one load per element | 7-way `if/else` | ~50–60 instr + 8 loads | compute-bound; **lost to FP8 and even BF16 on NVLink** |
| v1 | PRMT: 2× `__byte_perm` on a register LUT → e4m3 bytes → hardware fp8→f16 cvt (same trick as `hopper/prmt_decode_mxfp4_to_e4m3`; E2M1 element format is identical for MXFP4/NVFP4) | branchless compare-sum | ~18 instr, no memory | **fixed it**: B200 512M one-shot 19937 → 3862 µs |
| v2 | hardware `cvt.rn.f16x2.e2m1x2` | hardware `cvt.rn.satfinite.e2m1x2.f32` (TRT-LLM `fp32_vec_to_e2m1` operand order) | ~8 instr | ≈ v1 (±1–2 %) on sm_100/sm_103; **13–37 % slower than v1 on sm_120** |

Shipped selection (compile-time guard): decode = hardware on sm_100/sm_103, PRMT on sm_120;
encode = hardware on all Blackwell. Everything else is one source for all three architectures.

**Arch gotcha.** The E2M1 `cvt` instructions are arch-specific ("a") features available on
**all** Blackwell (sm_100, sm_101, sm_103, sm_120 — `cuda_fp4.hpp` guard
`SM100_ALL || SM101_ALL || SM120_ALL`, +SM103 in CUDA 13.x). The `-arch=sm_XXXa` shorthand does
**not** enable them in nvcc 13.x (PTX lands in `compute_XXX`; ptxas: *"Feature 'cvt.e2m1x2.f32'
not supported on .target 'sm_XXX'"*). Always build with the explicit form shown below.

---

## 3. How to run

### 3.1 Build matrix (explicit arch-specific gencode is mandatory)

```bash
# RTX PRO 5000 / 6000 / 6000D  (sm_120, 8 GPUs)
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 bench_ar.cu -o bench_ar

# B200 (sm_100, 8 GPUs) — static cudart so the binary runs on nodes without a toolkit
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -cudart static bench_ar.cu -o bench_ar_b200

# GB300 (sm_103, 4 GPUs/node, aarch64 — must be compiled ON an aarch64 host/container)
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -DNRANKS=4 bench_ar.cu -o bench_ar_gb300
```

`NRANKS` (default 8) must equal the number of visible GPUs. A cross-compiled sm_100a static binary
built on an x86 CUDA 13.3 box ran fine on the B200 nodes.

### 3.2 Run

```bash
./bench_ar <numel>                 # elements per rank; numel % (NRANKS*16) == 0
TRT_FP8_GRID=16 ./bench_ar <numel> # reproduce TRT-LLM's verbatim FP8 dispatch (default: grid scaled)
```

Sizes used: PCIe `524288 8388608` (small/medium, where TRT-LLM uses custom AR); NVLink
`33554432 134217728 536870912 2147483648` (32M–2G).

Per-rank device memory ≈ 43 × numel bytes at 8 ranks (the TRT one-shot push buffer
`2×world×numel×2 B` dominates): 2G elements → ~86 GB/GPU (fits B200 180 GB); 4G would not fit B200.
Host side allocates `world × numel` floats for the reference (2G × 8 → 64 GB RAM).

Always wrap runs: `stdbuf -oL timeout 2400 ./bench_ar N` — a leftover `bench_ar` from a killed run
holds the GPUs and makes the next run's flag barrier spin forever. Before re-running on a root box:
`pkill -9 bench_ar; nvidia-smi -rgc`.

Output per size: rel_rmse for every path, then

```
  ONE-SHOT  BF16=…  FP8=…  NVFP4=…  | NVFP4/BF16=…x  NVFP4/FP8=…x
  (TRT-FP8 grid=N blocks x 512 thr)
  TWO-SHOT  BF16=…  myFP8=…  TRT-FP8=…  NVFP4=…  | NVFP4/BF16=…x  NVFP4/myFP8=…x  NVFP4/TRT-FP8=…x  myFP8/TRT-FP8=…x
```

### 3.3 Per-environment recipes (exactly what produced the tables)

**A. Direct root box (RTX 6000D, 10.6.142.1)**
```bash
scp bench_ar.cu root@10.6.142.1:~/nvfp4_allreduce/
ssh root@10.6.142.1 '
  cd ~/nvfp4_allreduce && pkill -9 bench_ar
  /usr/local/cuda-13.3/bin/nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 bench_ar.cu -o bench_ar
  MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
  nvidia-smi -pm 1; nvidia-smi -lgc $MAX,$MAX
  for N in 524288 8388608; do echo NUMEL=$N; stdbuf -oL timeout 600 ./bench_ar $N; done
  nvidia-smi -rgc'
```

**B. computelab slurm (RTX PRO 6000, 8 GPUs) — no toolkit on the node, no sudo**
```bash
# build a static binary on any CUDA 13.x box, ship it
nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -cudart static bench_ar.cu -o bench_ar_static
scp bench_ar_static computelab-aus-01:~/nvfp4_allreduce/
ssh computelab-aus-01 "srun -p 'rtx-pro-6000-blackwell-server-edition@cr+mp/genoa2d24g2l/8gpu-256cpu-2304gb' \
   --gres=gpu:8 --gpu-freq=high -t 0:40:0 bash -lc '
   cd ~/nvfp4_allreduce
   for N in 524288 8388608; do echo NUMEL=\$N; stdbuf -oL timeout 600 ./bench_ar_static \$N; done
   nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1'"
```
`--gpu-freq` is slurm's gres clock lock — it works without sudo (`sudo nvidia-smi -lgc` fails here).

**C. B200 on nsc-svg-slurm-1 (AI Hub), 8 GPUs, NVLink**
```bash
nvcc -gencode arch=compute_100a,code=sm_100a -O3 -std=c++17 -cudart static bench_ar.cu -o bench_ar_b200
scp -o "ProxyJump computelab-sc-01" bench_ar_b200 kimiz@nsc-svg-slurm-1-login-02.nvidia.com:~/nvfp4_allreduce/
ssh -J computelab-sc-01 kimiz@nsc-svg-slurm-1-login-02.nvidia.com "srun -A coreai_comparch_inferencex --qos normal \
   -p batch --gres=gpu:8 --gpu-freq=1965 -t 1:30:0 bash -lc '
   cd ~/nvfp4_allreduce
   for N in 33554432 134217728 536870912 2147483648; do echo NUMEL=\$N; stdbuf -oL timeout 2400 ./bench_ar_b200 \$N; done
   nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1'"
```

**D. GB300 on dlcluster, 4 GPUs/node, aarch64 — compile inside a CUDA container**
```bash
scp bench_ar.cu dlcluster:~/nvfp4_allreduce/
ssh dlcluster "srun -A blackwell -p gb300nvl72_preprod --gres=gpu:GB300:4 --gpu-freq=high -t 1:30:0 \
   --container-image=nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc13 --container-mounts=\$HOME/nvfp4_allreduce:/work \
   bash -lc '
   cd /work
   /usr/local/cuda/bin/nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -DNRANKS=4 bench_ar.cu -o bench_ar_gb300
   for N in 33554432 134217728 536870912 2147483648; do echo NUMEL=\$N; stdbuf -oL timeout 2400 ./bench_ar_gb300 \$N; done
   nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1'"
```
The GB300 node has no `nvcc`; `--gpu-freq=high` set at the host level is honoured inside the
container (`nvidia-smi -lgc` inside the container is refused).

### 3.4 Using the standalone kernel

```cpp
#include "nvfp4_allreduce.cuh"
using namespace nvfp4_ar;
// per rank, after cudaSetDevice(rank); payload/reduced must be peer-visible; the *_ptr arrays are
// device arrays[world] of every rank's pointer; `flag` must increase monotonically per call.
launch_twoshot_rank(input, output, payload, reduced, peer_payloads, peer_reduced, barrier_sig,
                    world, rank, numel, /*SFScaleVal=*/(6.f*448.f)/amax, flag, stream);
```
Launch all `world` ranks before synchronizing — the flag barrier needs every rank in flight.

---

## 4. Results

Accuracy (rel_rmse vs FP32, identical on every card): BF16 0.004 · TRT-FP8 0.037 · myFP8 0.036 ·
NVFP4 one-shot 0.102 · NVFP4 two-shot 0.140. NVFP4 two-shot is bit-reproducible across runs.

### 4.1 PCIe — RTX 6000D, 8 GPUs, sm_120, locked ~2.35 GHz (µs)

| numel | shape | BF16 | TRT-FP8 | myFP8 | **NVFP4** | NVFP4/BF16 | **NVFP4/TRT-FP8** |
|---:|---|---:|---:|---:|---:|:--:|:--:|
| 512K | two-shot | 2143 | 2112 (g16) | 1954 | **1126** | 1.90× | **1.88×** |
| 8M | two-shot | 49556 | 36925 (g133) / 35961 (g16) | 36170 | **17935** | 2.76× | **2.06×** |
| 512K | one-shot | 9800 | — | 10386 | **3839** | 2.55× | 2.71× vs myFP8 |
| 8M | one-shot | 158192 | — | 167202 | **85103** | 1.86× | 1.96× vs myFP8 |

### 4.2 PCIe — RTX PRO 6000 Blackwell SE, 8 GPUs, sm_120, locked 2422 MHz (µs)

| numel | shape | BF16 | TRT-FP8 | myFP8 | **NVFP4** | NVFP4/BF16 | **NVFP4/TRT-FP8** |
|---:|---|---:|---:|---:|---:|:--:|:--:|
| 512K | two-shot | 2455 | 1881 (g16) | 2039 | **1249** | 1.97× | **1.51×** |
| 8M | two-shot | 44325 | 30411 (g133) / 29128 (g16) | 52306 | **25872** | 1.71× | **1.18×** |
| 512K | one-shot | 7105 | — | 11075 | **6075** | 1.17× | 1.82× vs myFP8 |
| 8M | one-shot | 116627 | — | 182587 | **83255** | 1.40× | 2.19× vs myFP8 |

On PCIe the 16-block TRT-FP8 grid already saturates the link; scaling it changes nothing
(≤4 %). TRT-FP8 beats myFP8 1.7× at 8M on this card — the reason a re-implemented FP8 must not be
the baseline.

### 4.3 NVLink — B200, 8 GPUs, sm_100, locked 1965 MHz (µs), two-shot

| numel | BF16 | TRT-FP8 grid 16 (verbatim) | **TRT-FP8 grid scaled** | myFP8 | NVFP4 | NVFP4/BF16 | NVFP4/TRT-FP8₁₆ | **NVFP4/TRT-FP8ₛ** |
|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 32M | 649 | 593 | **176** (g=529) | 278 | 211 | 3.08× | 2.80× | **0.83×** |
| 128M | 2655 | 2250 | **579** (g=2048) | 1021 | 694 | 3.82× | 3.28× | **0.83×** |
| 512M | 10601 | 8824 | **1859** (g=2048) | 3987 | 2671 | 3.97× | 3.31× | **0.70×** |
| 2G | 43707 | 35050 | **7215** (g=2048) | 16233 | 10479 | 4.17× | 3.34× | **0.69×** |

One-shot, B200: 32M BF16 1742 / myFP8 456 / **NVFP4 280** (6.2× / 1.63×); 512M 27737 / 6593 / **3846**
(7.2× / 1.71× ≈ byte ratio 1.89×); 2G 111023 / 26270 / **15310**. (TRT-LLM has no FP8 one-shot.)

### 4.4 NVLink — GB300, 4 GPUs/node, sm_103, locked 2070 MHz (µs), two-shot

| numel | BF16 | TRT-FP8 grid 16 (verbatim) | **TRT-FP8 grid scaled** | myFP8 | NVFP4 | NVFP4/BF16 | NVFP4/TRT-FP8₁₆ | **NVFP4/TRT-FP8ₛ** |
|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 32M | 591 | 684 | **186** (g=1058) | 207 | **164** | 3.60× | 4.14× | **1.13×** |
| 128M | 2480 | 2664 | **546** (g=2048) | 705 | **524** | 4.73× | 5.07× | **1.04×** |
| 512M | 9851 | 10499 | **1707** (g=2048) | 2648 | 1944 | 5.07× | 5.39× | **0.88×** |
| 2G | 40136 | 41846 | **6359** (g=2048) | 10419 | 7663 | 5.24× | 5.46× | **0.83×** |

One-shot, GB300: 32M BF16 810 / myFP8 210 / **NVFP4 152**; 512M 12876 / 2966 / **1980**; 2G 51588 / 11809 / **7923**.

**Reading the NVLink tables.** With its verbatim 16-block dispatch TRT-FP8 is occupancy-starved
on NVLink (16 × 512 threads cannot drive ~900 GB/s) and runs at BF16 speed — off-design, so the
`NVFP4/TRT-FP8₁₆` column (2.8–5.5×) must not be quoted. At fair occupancy TRT-LLM's FP8 kernel
**beats NVFP4 on B200 at every size (NVFP4 17–31 % slower)** and roughly ties on GB300
(+13 % … −17 %).

### 4.5 Codec ablation (what actually moved the numbers)

| card | config | one-shot | two-shot |
|---|---|---:|---:|
| B200 512M | v0 local-memory LUT decode + branchy encode | 19937 µs (0.33× vs myFP8) | 5996 µs (0.68×) |
| B200 512M | v1 PRMT decode + hw encode | 3862 (1.71×) | 2671 (1.50×) |
| B200 512M | v2 hw decode + hw encode | 3846 | 2672 |
| GB300 2G | PRMT decode + software encode | — | 9059 |
| GB300 2G | PRMT decode + hw encode | — | 7798 |
| GB300 2G | hw decode + hw encode | — | 7662 |
| 6000D 512K | PRMT decode | 3893 | 1111 |
| 6000D 512K | hw decode (sm_120) | 5342 (**+37 %**) | 1258 (+13 %) |
| 6000D 8M | PRMT decode / hw decode | 92471 / 85218 | 15645 / 18057 |

Removing the per-element local-memory lookup (v0 → v1) is the whole story; hardware vs PRMT
decode is a wash on sm_100/103 and a loss on sm_120. Removing `asm volatile` changed nothing.

### 4.6 Early data — RTX PRO 5000, 8 GPUs, sm_120, locked ~2.6 GHz, **v0 codec**, BF16 / myFP8 only (µs)

| numel | one-shot BF16 / myFP8 / NVFP4 | two-shot BF16 / myFP8 / NVFP4 |
|---:|---|---|
| 32K | 591 / 250 / **171** | 108 / 125 / **95** |
| 512K | 9860 / 9713 / **4039** | 2133 / 1906 / **842** |
| 2M | 39341 / 41866 / **21316** | 12011 / 7949 / **3555** |
| 8M | 156858 / 156287 / **94558** | 51152 / 27689 / **12683** |

Included for completeness; superseded by §4.1–4.2 (fast codec, TRT-FP8 baseline). The box is shared and was not re-run.

---

## 5. Conclusions

1. **vs TRT-LLM BF16 custom AR: NVFP4 wins everywhere** — 1.7–2.8× on PCIe, 3–5× on NVLink (two-shot, large messages).
2. **vs TRT-LLM's real FP8 kernel** — **wins 1.2–2.1× on PCIe**, **loses on NVLink**: B200 −17…−31 % at all sizes 32M–2G, GB300 +13 % … −17 %. Only the verbatim TRT-LLM FP8 kernel is a valid baseline; a per-16-scale re-implementation ran 1.6–2.3× slower than it on NVLink and would have made NVFP4 look 1.5× better than it is.
3. **Why NVFP4 loses on NVLink despite moving 1.9× fewer bytes than FP8:** kernel structure, not format. This NVFP4 path is five launches (quantize, barrier, reduce-scatter, barrier, all-gather) with two separate barrier kernels and per-16 scales read on both legs; TRT-FP8 is one fused kernel with an in-kernel barrier, 16-element int4 loads and a per-496 in-stream scale. PCIe (~0.7 GB/s effective P2P) hides all of that; NVLink does not.
4. **The codec is a precondition, not the differentiator.** A runtime-indexed table in local memory made NVFP4 compute-bound and lose to BF16 on NVLink; PRMT or hardware `cvt` (any register-only decode) restores the bandwidth-bound regime. Hardware E2M1 `cvt` exists on every Blackwell including sm_120 — build with explicit `-gencode …a`.
5. Accuracy cost: rel_rmse 0.14 (NVFP4 two-shot) vs 0.037 (FP8).
6. **Next step with a real chance to flip NVLink:** a fused single-kernel NVFP4 with TRT-LLM's structure (in-kernel barriers, one launch, 16-element loads, coarser in-stream scale). The byte budget favours NVFP4 1.9× over FP8; the deficit is implementation.
