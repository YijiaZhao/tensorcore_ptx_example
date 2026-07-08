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

## 5. 怎么编译、怎么跑

### 环境：统一 docker 镜像（宿主机零依赖）

所有实测用同一个镜像（自带 nvcc，CUDA 13.0，支持 sm_89~sm_121a）：

```
nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu22.04
```

### 每个 .cu 的通用跑法

每个文件都是自包含单文件，无任何外部依赖，三步：**编译 → 运行 → 看 Result/PASS 行**。
编译指令写在每个 .cu 的头注释里，只有 `-gencode` 按架构换：

| 架构 | -gencode | 测试卡 |
|---|---|---|
| ada (sm_89) | `arch=compute_89,code=sm_89` | L40 |
| hopper (sm_90a) | `arch=compute_90a,code=sm_90a` | H20 |
| blackwell tcgen05 (sm_100a) | `arch=compute_100a,code=sm_100a` | B200 |
| blackwell mma.sync 扩展 (sm_120a) | `arch=compute_120a,code=sm_120a` | RTX PRO 6000 |

```bash
# 例: 在 B200 上跑一个 tcgen05 例子
nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -o /tmp/t tcgen05_fp8.cu && /tmp/t
# 输出: Result: 128/128 correct (=32.0)  ← 全对即硬件行为符合预期

# 批量跑一个目录 (架构对应换 -gencode):
for f in */*.cu; do b=$(basename $f .cu)
  nvcc -gencode arch=compute_120a,code=sm_120a -std=c++17 -o /tmp/$b $f && /tmp/$b | grep -E "Result|PASS"
done
```

注: `tcgen05_*` 需要 `-std=c++17`；`mma_bf16_sass_compare_sm89_sm120.cu` 是纯 SASS 观察 stub（无 main），
只能 `-cubin` 编译后 `cuobjdump -sass` 看，不能运行。

### 宿主机没有 CUDA 工具链时（docker 跑法，零依赖）

```bash
docker run --rm --gpus all -v <本仓库路径>:/work \
  nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu22.04 \
  bash -c 'cd /work/ada && for f in */*.cu; do b=$(basename $f .cu);
           nvcc -gencode arch=compute_89,code=sm_89 -std=c++17 -o /tmp/$b $f && /tmp/$b | grep -E "Result|PASS"; done'
```

### 随机数据 + CPU 参考验证（内嵌于每个 .cu + `random_cpu_ref/`）

全 1.0 测试对布局/映射错误不敏感（输入对称，怎么排结果都一样）。因此**每个含 TC 指令的 .cu
都内嵌两阶段验证**，跑任何示例都会依次执行：

1. **阶段1（全 1.0）**：指令语义 smoke——格式解码、K 长度、累加、scale 生效
2. **阶段2（随机 vs CPU）**：同一条指令，A/B 每元素独立随机（block scale 的 scale 也随机），
   CPU 三重循环算参考，逐元素 `==` 精确比对——fragment 映射 / descriptor 布局 /
   读回映射 / 完成同步，全 1.0 兜不住的错误全在这层现形。
   tcgen05 类的 D 读回布局用 base-4 one-hot 探针真机实测（含 cta_group::2 双 CTA 256 槽位），
   不依赖任何文档假设；一致性探针自动过滤 TMEM 跨 kernel 残留。
   退出码 = 阶段2 结果（0=全过），可直接接 CI。

`random_cpu_ref/` 提供公共件与独立版验证器：

- **`verify_random.h`**（可复用）：各格式编码表、确定性随机、CPU 参考 GEMM、精确比对。
  取值集刻意限制在 {0, ±0.5, ±1, ±1.5, ±2} —— 任意乘加顺序在 fp32 里精确，
  所以 CPU float 和 tensor core 结果可以逐 bit `==` 比对，无容差。
- **`mma_random_cpu_ref_all_arch.cu`**：六精度 mma.sync，fragment 映射按 PTX 布局装载，四架构通用
- **`wgmma_random_cpu_ref_sm90.cu`**：五精度 wgmma，smem 按 core-matrix 布局排
- **`tcgen05_random_cpu_ref_sm100.cu`**：五精度 tcgen05，D 读回布局用 one-hot 探针实测（不做假设），
  一致性探针自动过滤 TMEM 跨 kernel 残留槽位

跑法（-gencode 按第 5 节开头的架构表换）：

```bash
cd random_cpu_ref
nvcc -gencode arch=compute_120a,code=sm_120a -std=c++17 -o t mma_random_cpu_ref_all_arch.cu
./t          # 默认 seed=1, 每精度3轮; 期望每行 "PASS (128/128 exact)", 最后 "全部PASS"
./t 42       # 换个 seed 复跑 (确定性随机, 同seed结果可复现)
# wgmma/tcgen05 版同理; tcgen05 版会先打印 "D布局探针: OK, 64/128 槽位一致有效"
# 退出码: 0=全过, 非0=有FAIL (可以直接进CI)
```

这套测试实际抓出过两个"全 1.0 全对、随机数据全错"的真问题（已修，教训写在文件头）：
1. **GMMA/UMMA smem descriptor 的 no-swizzle 布局不是线性行主**，是 8行×16B core-matrix 分块，
   K 方向 core-matrix 间距(LBO)=128B、M 方向(SBO)=256B —— 参数扫描实测钉死
2. **`tcgen05.ld` 后必须 `tcgen05.wait::ld`**，否则读寄存器是竞态；且 TMEM dealloc/realloc
   不清零，会读到上一个 kernel 的残留数据

### SASS 对照(验证原生/模拟)

```bash
nvcc -gencode arch=compute_XXa,code=sm_XXa -cubin -o x.cubin x.cu
cuobjdump -sass x.cubin | grep -oE "(H|Q|I)G?MMA[A-Z0-9._]*|F2FP[A-Z0-9._]*|IMAD.SHL" | sort | uniq -c
# 单条 *MMA = 原生; 出现 F2FP(fp8上转) 或大量 IMAD.SHL(int4解包) 前奏 = 模拟
```

验证状态（2026-07，CUDA 13.0）：

| 目录 | 真机运行 | 编译+SASS |
|---|---|---|
| blackwell tcgen05 (sm_100) | ✅ B200 | ✅ |
| blackwell mma.sync (sm_120, 含 fp4/mxfp8/原生fp8) | ✅ RTX PRO 6000 Blackwell | ✅ QMMA.SF 等 |
| hopper wgmma + mma.sync 全部 | ✅ H20 | ✅ |
| ada 全部 (原生 fp8 QMMA + 末代 int4 IMMA.S4) | ✅ L40 | ✅ 全原生单指令 |
| comm/ P2P + TMA | ✅ GB200 ×4 | — |
| comm/ multimem (需 NVSwitch multicast) | ✅ B200 ×8 (NVL4 小机型不支持, check 工具可探测) | — |

SASS 检查方法: `cuobjdump -sass x.cubin | grep -E "MMA|GMMA|F2FP|IMAD.SHL"`
（看到 F2FP/IMAD.SHL 前奏 = 模拟路径; 单条 *MMA = 原生）
