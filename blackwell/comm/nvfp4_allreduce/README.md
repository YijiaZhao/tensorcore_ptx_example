# NVFP4-compressed All-Reduce vs TRT-LLM custom All-Reduce

An NVFP4 (4-bit) compressed all-reduce built on **TRT-LLM's own custom-AR algorithm**
(one-shot / two-shot, reduce-scatter + all-gather), benchmarked head-to-head against
the **verbatim TRT-LLM BF16 kernels** in one identical single-process P2P harness.

Same algorithm, same harness, same box — the **only** variable is the wire payload:
NVFP4 (~0.5625 B/elt) vs BF16 (2 B/elt), a 3.56x byte reduction. This isolates exactly
what low-precision compression buys on top of TRT-LLM's transport.

## Files

| file | what |
|---|---|
| `nvfp4_allreduce.cuh` | standalone, reusable NVFP4 **two-shot** all-reduce (quantize / reduce-scatter / all-gather / barrier kernels + host launcher). Drop-in; portable sm_100a **and** sm_120a. |
| `bench_ar.cu` | the benchmark: TRT-LLM's verbatim `oneShot`/`twoShot` PUSH_MODE BF16 kernels (fused `block_barrier` + `add128b`) **vs** NVFP4 one-shot/two-shot, in one harness. |

## What "TRT-LLM custom AR" means here

The BF16 baseline is **not** a hand-rolled naive reducer. It is the literal algorithm
ported from `TensorRT-LLM/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu`:

- **PUSH_MODE**: each rank writes its chunk into every peer's buffer column `[local_rank]`,
  then reads its own buffer's columns and reduces locally.
- **fused `block_barrier`**: flag-based (`st.release.sys` / `ld.acquire.sys`) with per-block
  ping-pong offsets — **one kernel launch per all-reduce**, no separate barrier kernel,
  no `cudaDeviceSynchronize` between phases.
- **`add128b`** vectorized 128-bit reduce over `PackedBFloat16` (`__nv_bfloat162[4]`).

The NVFP4 path reuses the identical two-shot structure; its reduce dequantizes each peer's
E2M1 block to FP32, accumulates, and **re-quantizes the reduced shard exactly once**
(4-bit cannot survive per-hop requant), writing to a separate reduced buffer.

## Hardware

- **8× NVIDIA RTX PRO 5000 Blackwell (72 GB)** — compute capability **12.0** (`sm_120`), 73415 MiB each
- **PCIe**, no NVLink. **Two NUMA nodes**: GPU0–3 (NUMA 0) / GPU4–7 (NUMA 1).
  Intra-NUMA links are `PXB`/`NODE`; every cross-NUMA hop is `SYS` (over the CPU
  interconnect) — the all-to-all in one/two-shot pays this on the 0↔1 crossings.
- CUDA 13.0, `nvcc -arch=sm_120a -O3 -std=c++17`
- Single process, 8 GPUs via `cudaDeviceEnablePeerAccess` + raw peer pointers.
- Timing: CUDA events on rank 0, 20 warmup + 100 timed iters, µs per all-reduce.

## Results — NVFP4 vs TRT-LLM BF16 custom AR (same harness)

`numel` = elements per rank (BF16 tensor = `numel`×2 B). **Lower µs is better.**

### TWO-SHOT — reduce-scatter + all-gather, O(N) traffic (the production config)

| numel | BF16 tensor | TRT-LLM BF16 (µs) | NVFP4 (µs) | **NVFP4 speedup** |
|---:|---:|---:|---:|:--:|
| 32 K   | 64 KiB | 106.0    | 94.1    | **1.13×** |
| 512 K  | 1 MiB  | 2 178    | 857     | **2.54×** |
| 2 M    | 4 MiB  | 12 107   | 3 586   | **3.38×** |
| 8 M    | 16 MiB | 51 773   | 12 789  | **4.05×** |
| 33 M   | 64 MiB | 212 437  | 59 630  | **3.56×** |

### ONE-SHOT — all-read-all, O(N²) traffic

| numel | BF16 tensor | TRT-LLM BF16 (µs) | NVFP4 (µs) | **NVFP4 speedup** |
|---:|---:|---:|---:|:--:|
| 32 K   | 64 KiB | 595      | 180     | **3.30×** |
| 512 K  | 1 MiB  | 10 010   | 3 971   | **2.52×** |
| 2 M    | 4 MiB  | 39 650   | 21 635  | **1.83×** |
| 8 M    | 16 MiB | 158 053  | 95 704  | **1.65×** |
| 33 M   | 64 MiB | 631 768  | 392 606 | **1.61×** |

### Accuracy (rel_rmse vs FP32 reference)

| path | rel_rmse |
|---|---|
| TRT-LLM BF16 one-shot / two-shot | 0.004 – 0.017 (BF16 accumulation only) |
| NVFP4 one-shot | ~0.102 |
| NVFP4 two-shot | ~0.140 (input quant + one requant of the reduced sum) |

NVFP4 two-shot is deterministic (identical rel_rmse across repeated runs — no data race).

## How to read this

1. **Two-shot large messages beat the 3.56× payload ratio (up to 4.05×)**: NVFP4 moves
   compressed data on **both** legs (reduce-scatter push *and* all-gather pull) while the
   BF16 baseline moves full BF16 on both. The extra saving comes from the second leg.
2. **One-shot large messages win less (down to 1.61×)**: NVFP4 one-shot must dequantize
   *every* peer's payload, so it becomes **compute-bound** on the E2M1→FP32 decode, eroding
   the byte advantage. Quantization itself is cheap (`quant_only` ≈ 15–41 µs); the cost is
   the per-element decode inside the reduce.
3. **Absolute µs are inflated by PCIe cross-NUMA + a single-process, per-phase-launch
   harness** (TRT-LLM's kernels are built for NVLink). Both paths pay this equally, so the
   *relative* comparison is fair; on NVLink both would be far faster and the gap would still
   track the 3.56× compression.

**Best config: NVFP4 two-shot** — O(N) traffic *and* compression, beating the verbatim
TRT-LLM BF16 custom AR by **1.1–4.1×** at the cost of rel_rmse ≈ 0.14.

## Build & run

```bash
nvcc -arch=sm_120a -O3 -std=c++17 bench_ar.cu -o bench_ar   # or -arch=sm_100a on B200
for N in 32768 524288 2097152 8388608 33554432; do ./bench_ar $N; done
```

Requires 8 P2P-capable GPUs (the TRT-LLM kernels are templated on `RANKS=8`).

## Reusing the kernel

`nvfp4_allreduce.cuh` is self-contained. Per rank (after `cudaSetDevice(rank)`):

```cpp
#include "nvfp4_allreduce.cuh"
using namespace nvfp4_ar;
// payload/reduced buffers must be peer-visible (P2P pointers or cudaIpc handles);
// peer_payloads/peer_reduced/barrier_sig are device arrays[world] of every rank's pointer.
launch_twoshot_rank(input, output, payload, reduced,
                    peer_payloads, peer_reduced, barrier_sig,
                    world, rank, numel, SFScaleVal, /*flag=*/monotonic, stream);
```

`SFScaleVal = (E2M1_MAX * 448) / amax`. `numel % (world*16) == 0`. Launch all `world` ranks
before syncing (the flag barrier needs every rank in flight — TRT-LLM's execution model).
E2M1 is encoded manually (no `cvt.e2m1x2` PTX), so the same source runs on sm_100a and sm_120a.
