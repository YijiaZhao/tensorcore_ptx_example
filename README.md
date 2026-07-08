# Tensor Core PTX 最小示例集

每个 `.cu` 都是**单条 tensor core 指令的最小可跑验证**：全 1.0/1 输入 → 期望输出 = K（或 K×scale），
一眼看出指令是否真的按预期执行。附带 SASS 对照，区分「原生硬件」和「编译器模拟」。

```
目录 = 架构 / 指令族_精度_适用sm[_模拟标记] / 文件.cu
ada/       sm_89        (L40S / RTX 40)
hopper/    sm_90a       (H100 / H20)
blackwell/ sm_100a+120a (B200 / RTX 6000D)
```

---

## 1. 三代指令族总览

每代卡只有一个「正主路径」(★)：新精度硬件只往正主上加，非正主保功能不保速度。

```
                 Ampere     Ada        Hopper      Blackwell大卡   Blackwell消费级
                 sm_80      sm_89      sm_90a      sm_100a/103a    sm_120a
                 (A100)     (L40S)     (H100/H20)  (B100/B200/300) (RTX 6000D)
───────────────────────────────────────────────────────────────────────────────
mma.sync         ★ 正主     ★ 正主      兼容层       兼容层          ★ 正主
(同步/寄存器)
wgmma              —          —       ★ 正主         — (已移除)       —
(异步/smem→寄存器)
tcgen05            —          —          —         ★ 正主            —
(异步/smem→TMEM)
```

三族执行模型对比：

```
mma.sync   : 32线程(1 warp)同步发射, A/B/C/D 全在寄存器, tile 16×8×K
wgmma      : 128线程(1 warpgroup)发射, A/B 走 smem descriptor, D 在寄存器,
             异步 (fence → mma_async → commit_group → wait_group), tile 64×N×K
tcgen05    : 1线程发射(elect_one), A/B 走 smem descriptor, D 在 TMEM,
             异步 (mbarrier + tcgen05.commit), tile 128×N×K (cta_group::2 → 256)
单条指令 K 三族相同 = 32字节 (fp16/bf16→16, tf32→8, fp8/int8→32, fp4→64)
长 K = 沿 K 发多条, 累加器原位累加 (首条 p=0 清零, 后续 p=1)
```

---

## 2. 精度支持矩阵（本仓库全部实测）

✅ = 原生单条指令   🔶 = 能跑但编译器模拟   ❌ = 指令不存在

### mma.sync

| 精度 | Ada sm_89 | Hopper sm_90 | B200 sm_100 | 6000D sm_120 | 备注 |
|---|---|---|---|---|---|
| fp16  | ✅ HMMA.16816 | ✅ | ✅ | ✅ | sm_70 起全代原生 |
| bf16  | ✅ HMMA.16816.BF16 | ✅ | ✅ | ✅ | sm_80 起 |
| tf32  | ✅ HMMA.1688.TF32 | ✅ | ✅ | ✅ | sm_80 起 |
| int8  | ✅ IMMA.16832.S8 | ✅ | ✅ | ✅ | |
| fp8   | ✅ QMMA.16832.E4M3 **首发** | 🔶 fp16模拟 | 🔶 fp16模拟 | ✅ QMMA | 模拟 = F2FP.F16.E4M3 上转 + 2×HMMA, 吞吐=fp16 |
| int4  | ✅ IMMA.16864.S4 **末代** | 🔶 int8模拟 | 🔶 int8模拟 | 🔶 int8模拟 | 模拟 = 36×IMAD 解包 + 2×IMMA.S8, sm_90 起硬件全线移除 |
| fp4 (block scale) | ❌ | ❌ | ❌ | ✅ **专属** | kind::mxf4nvf4.block_scale |
| mxfp8 (block scale) | ❌ | ❌ | ❌ | ✅ QMMA.SF **专属** | kind::mxf8f6f4.block_scale |

### wgmma（Hopper 专属，Blackwell 已移除）

| 精度 | sm_90a | SASS |
|---|---|---|
| fp16 / bf16 | ✅ | HGMMA |
| tf32 | ✅ | HGMMA.TF32 |
| fp8 e4m3 | ✅ | **QGMMA.64 — Hopper 唯一原生 fp8 路径** |
| int8 | ✅ | IGMMA |
| fp4 / int4 / block scale | ❌ | — |

### tcgen05（Blackwell 大卡专属：sm_100a / sm_103a）

| 精度 | kind | 备注 |
|---|---|---|
| fp16 / bf16 | `kind::f16` | InstrDescriptor a/b_format 位选 0=F16 / 1=BF16 |
| tf32 | `kind::tf32` | |
| fp8 (裸) | `kind::f8f6f4` | 无 scale |
| int8 | `kind::i8` | s32 累加 |
| mxfp8 | `kind::mxf8f6f4.block_scale.scale_vec::1X` | 每32元素 1×ue8m0, scale 存 TMEM, **硬件相乘** |
| mxfp4 | `kind::mxf4.block_scale.scale_vec::2X` | 每32元素 1×ue8m0 |
| nvfp4 | `kind::mxf4nvf4.block_scale.scale_vec::4X` | 每16元素 1×ue4m3 |
| int4 | ❌ 无 `kind::i4` | 4-bit 赛道全给了 FP4 |

---

## 3. 两条硬件演进主线（踩坑重点）

### fp8：挂在谁身上取决于谁是正主

```
sm_89 (Ada)     mma.sync ✅ QMMA          ← fp8 首发, 挂正主 mma.sync
sm_90 (Hopper)  mma.sync 🔶 / wgmma ✅    ← 正主换 wgmma, mma.sync 变 fp16 模拟
sm_100 (B200)   mma.sync 🔶 / tcgen05 ✅  ← 正主换 tcgen05, 同上
sm_120 (6000D)  mma.sync ✅ QMMA          ← 消费级没有异步族, mma.sync 回归正主
```

### int4：sm_90 起硬件绝版，天花板 = int8

```
sm_80/89        IMMA.16864.S4 原生, 2× int8 吞吐
sm_90/100/120   同一条 PTX → 36×IMAD 解包 + 2×IMMA.16832.S8 (借 int8 单元)
                → 上限 = int8 速率, 实际还倒贴解包开销
                → int4 量化模型请走 w4a8/w4a16 软件解包, 或改 nvfp4
```

### block scale：Blackwell 独占的硬件能力

| | 谁有 | scale 格式 | 粒度 |
|---|---|---|---|
| mxfp8 | tcgen05(sm_100) + mma.sync(sm_120) | ue8m0（纯 2 的幂）| 1/32 |
| mxfp4 | 同上 | ue8m0 | 1/32 |
| nvfp4 | 同上 | ue4m3（可非 2 幂）| 1/16 |

Ada/Hopper 的 fp8 scale 只能软件做（TE per-tensor epilogue 乘 / DeepGEMM 寄存器 promotion）。
**格式对不上硬件规格的 scale（如 DeepSeek 的 fp32 1×128）在 Blackwell 上也仍是软件。**

---

## 4. 目录导航

```
ada/                                sm_89, 只有 mma.sync (六精度全原生的唯一架构)
  mma.sync_{fp16,bf16,tf32,fp8,int8,int4}_sm89/

hopper/                             sm_90a
  mma.sync_{fp16,bf16,tf32,int8}_sm90/
  mma.sync_fp8_sm90_fp16emu/        ← 模拟, 原生请用 wgmma
  mma.sync_int4_sm90_int8emu/
  wgmma_{fp16,bf16,tf32,fp8,int8}_sm90/

blackwell/                          sm_100a (tcgen05) + sm_120a (mma.sync 扩展)
  mma.sync_{fp16,bf16,tf32,int8}_sm100_sm120/
  mma.sync_fp8_sm100_sm120_sm100fp16emu/   (含 mxfp8 block scale, sm120 only)
  mma.sync_int4_sm100_sm120_int8emu/
  mma.sync_fp4_sm120/               nvfp4(block16 ue4m3) + mxfp4(block32 ue8m0)
  tcgen05_{fp16,bf16,tf32,fp8,int8}_sm100/ (fp8 目录含 mxfp8 block scale)
  tcgen05_fp4_sm100/                nvfp4/mxfp4 + maxtile + cta_group::2 + warp数实验
  warp_specialization/              producer/consumer 流水线 (named barrier / mbarrier / setmaxnreg)
  comm/                             TMA / P2P / multimem 通信最小例
  ptx_inline_asm_shl_demo.cu        PTX 内联汇编入门 (与 tensor core 无关)
```

## 5. 编译与验证

```bash
# 每个文件头都有对应架构的编译行, 例:
nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -o x tcgen05_fp8.cu   # B200
nvcc -gencode arch=compute_90a,code=sm_90a  -std=c++17 -o x wgmma_fp8.cu      # H100/H20
nvcc -gencode arch=compute_120a,code=sm_120a -o x mma_mxfp8_block32_ue8m0_sm120only.cu  # 6000D
```

验证状态（2026-07，CUDA 13.0）：

| 目录 | 真机运行 | 编译+SASS |
|---|---|---|
| blackwell tcgen05/mma.sync 全部精度 | ✅ B200 | ✅ |
| hopper wgmma + mma.sync 全部 | ✅ H20 | ✅ |
| ada 全部 | 待 Ada 卡 | ✅ sm_89 全原生单指令 |
| blackwell sm_120 专属 (fp4/mxfp8) | 待 6000D | ✅ QMMA.SF 等 |

SASS 检查方法: `cuobjdump -sass x.cubin | grep -E "MMA|GMMA|F2FP|IMAD.SHL"`
（看到 F2FP/IMAD.SHL 前奏 = 模拟路径; 单条 *MMA = 原生）
