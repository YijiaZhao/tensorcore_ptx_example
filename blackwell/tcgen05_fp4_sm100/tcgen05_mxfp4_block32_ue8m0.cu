/**
 * tcgen05.mma 最小单元 — MXFP4 block32 (sm_100, B100/B200)
 *
 * 指令: tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
 *       [tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p
 *
 * MXFP4: e2m1 data + ue8m0 scale, 每32个FP4共享1个scale
 * Tile:  M=128, N=8, K=64, 128线程 (4 warp, cta_group::1)
 * FMA:   128 × 8 × 64 = 65,536
 *
 * 数据流:
 *   1. tcgen05.alloc  → 分配TMEM (累加器 + scale)
 *   2. 填充smem        → A[128×64] FP4, B[8×64] FP4
 *   3. tcgen05.st     → 写ue8m0 scale到TMEM
 *   4. make_desc()    → 构造smem descriptor (uint64)
 *   5. tcgen05.mma    → 1个线程发射 (elect_one)
 *   6. tcgen05.ld     → 从TMEM读结果
 *   7. tcgen05.dealloc → 释放TMEM
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 \
 *            -o tcgen05_mxfp4_block32_ue8m0 tcgen05_mxfp4_block32_ue8m0.cu
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

__device__ uint32_t smem_u32(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// SmemDescriptor: [0:14) addr>>4, [16:30) LBO>>4, [32:46) SBO>>4, [46:48) version=1
// ⚠️ no-swizzle 布局是 8行×16B core-matrix 分块(不是线性行主): K方向间距LBO=128B, M/N方向SBO=256B
//    本例全1.0输入对排布不敏感; 喂真实数据时 smem 必须按 core-matrix 排, 见 random_cpu_ref/
__device__ uint64_t make_desc(void const* smem_ptr, int stride_bytes) {
    (void)stride_bytes;
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)8  << 16;   // LBO = 128B >> 4
    desc |= (uint64_t)16 << 32;   // SBO = 256B >> 4
    desc |= (uint64_t)(1) << 46;
    return desc;
}

// InstrDescriptorBlockScaled for MXFP4: a/b=E2M1, scale=UE8M0, M=128, N=8
__device__ uint32_t make_idesc(uint32_t tsfa, uint32_t tsfb) {
    uint32_t d = 0;
    d |= (1 << 7);                    // a_format = E2M1
    d |= (1 << 10);                   // b_format = E2M1
    d |= (1 << 17);                   // n_dim = N/8 = 1
    d |= (1 << 23);                   // scale_format = 1 (UE8M0)
    d |= (8 << 24);                   // m_dim = M/16 = 8
    d |= ((tsfa >> 30) & 3) << 29;    // a_sf_id
    d |= ((tsfb >> 30) & 3) << 4;     // b_sf_id
    return d;
}

constexpr int M = 128, N = 8, K = 64;
constexpr int THREADS = 128;
constexpr int A_BYTES = M * K / 2;  // 4096
constexpr int B_BYTES = N * K / 2;  // 256

// (原阶段1全1.0测试已移除: 随机vs CPU为严格超集 — PHASE1_STRIPPED)

// ============================================================
// 验证: 随机数据 vs CPU 参考 (逐bit精确) — host驱动统一在 verify_random.h::vr_verify()
//   本文件只保留: 被测kernel + launch适配器
// ============================================================
#include "../../random_cpu_ref/verify_random.h"

// ======================= 被测 kernel =======================
// tcgen05 执行模型 (Blackwell 大卡 sm_100a 专属):
//   - A/B 走 smem descriptor(同 wgmma); 累加器 D 不在寄存器, 住在 TMEM
//     (Tensor Memory, 每 SM 256KB 专用, 用 tcgen05.alloc 按列分配)
//   - 发射只要 1 个线程(对比 mma.sync 的32线程/wgmma 的128线程), 异步执行
//   - 结果用 tcgen05.ld 从 TMEM 搬回寄存器, 必须跟 wait::ld 等完成
__global__ void vk(const uint8_t* A, const uint8_t* B, uint32_t* D_raw, uint32_t sfa_b, uint32_t sfb_b) {
    extern __shared__ char smem[];
    uint32_t* sA = (uint32_t*)smem;
    uint32_t* sB = (uint32_t*)(smem + A_BYTES);
    uint32_t* s_tmem = (uint32_t*)(smem + A_BYTES + B_BYTES + 64);
    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;
    for (int i = tid; i < A_BYTES / 4; i += THREADS) sA[i] = ((uint32_t*)A)[i];
    for (int i = tid; i < B_BYTES / 4; i += THREADS) sB[i] = ((uint32_t*)B)[i];
    __syncthreads();
    // ① 分配 TMEM: 整个 warp0 执行, 32 列(每列 4B×128 lane), 基址写回 smem
    if (warp_id == 0)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(smem_u32(s_tmem)), "r"(32));
    __syncthreads();
    uint32_t tmem_c = *s_tmem;   // 累加器 D 的 TMEM 地址
    // scale 也住 TMEM: 用 tcgen05.st 写入 (ue8m0/ue4m3, 每行每K段1字节, 此处全行同值)
    uint32_t tmem_sfa = tmem_c + 8, tmem_sfb = tmem_c + 12;
    if (warp_id == 0) {
        uint32_t va = (sfa_b & 0xFF) * 0x01010101u, vb = (sfb_b & 0xFF) * 0x01010101u;
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfa), "r"(va), "r"(va), "r"(va), "r"(va));
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_sfb), "r"(vb), "r"(vb), "r"(vb), "r"(vb));
        asm volatile("tcgen05.wait::st.sync.aligned;");
    }
    __syncthreads();
    // ② smem descriptor: {基址>>4, LBO=128B(K方向core-matrix间距), SBO=256B(M/N方向), version=1}
    uint64_t da = make_desc(sA, 32), db = make_desc(sB, 32);
    __syncthreads();
    // ③ 发射 MMA: 只需 1 个线程! (真实kernel用 elect.sync 选举)
    //    tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p;
    //    操作数: [D的TMEM地址], desc_A, desc_B, idesc(指令描述符,编码M/N/格式), p(0=清零写入,1=累加)
    //    idesc 位段: [4:6)累加器格式 [7:10)A格式 [10:13)B格式 [17:23)N/8 [24:29)M/16
    if (tid == 0) {
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
            : : "r"(tmem_c), "l"(da), "l"(db), "r"((uint32_t)((1<<7)|(1<<10)|(1<<17)|(1<<23)|(8<<24)|(((tmem_c+8)>>30&3)<<29)|(((tmem_c+12)>>30&3)<<4))), "r"(0u), "r"(tmem_sfa), "r"(tmem_sfb));
    }
    __syncthreads();
    // ④ 读回: 16x256b = D 的 16行×8列窗口; ld 是异步的, wait::ld 之后寄存器才可靠
    //    (漏 wait::ld 是竞态 —— 全1.0数据下看不出来, 随机数据当场现形, 已实测踩过)
    if (warp_id == 0) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n\t"
                     "tcgen05.wait::ld.sync.aligned;\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));
        D_raw[lane*4+0]=r0; D_raw[lane*4+1]=r1; D_raw[lane*4+2]=r2; D_raw[lane*4+3]=r3;
    }
    __syncthreads();
    // ⑤ 释放 TMEM (不释放会泄漏, TMEM 不随 kernel 结束自动回收干净)
    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_c), "r"(32));
}

static void vr_run(const uint8_t* hA, const uint8_t* hB,
                   const uint32_t* sfa, const uint32_t* sfb, void* out) {
    static uint8_t *dA, *dB; static uint32_t* dD;
    if (!dA) { cudaMalloc(&dA, A_BYTES); cudaMalloc(&dB, B_BYTES); cudaMalloc(&dD, 512); }
    cudaMemcpy(dA, hA, A_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_BYTES, cudaMemcpyHostToDevice);
    vk<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD, sfa[0], sfb[0]);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dD, 512, cudaMemcpyDeviceToHost);
}

int main() {
    VrSpec sp = {}; sp.M = 128; sp.N = 8; sp.K = K; sp.ebits = 4; sp.is_int = 0;
    sp.dset = E2M1_SET; sp.dsetn = SETN(E2M1_SET); sp.core_matrix = 1;
    sp.probe = 1; sp.nslot = 128; sp.enc_tab = VR_ENC_E2M1; sp.sf_one = 0x7F;
    sp.sset = UE8M0_SET; sp.ssetn = SETN(UE8M0_SET); sp.nseg = 1;
    return vr_verify(sp, vr_run, /*seed=*/1);   // 随机全在CPU侧生成(见verify_random.h数据流注释); 同seed可复现
}
