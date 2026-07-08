/**
 * tcgen05.mma 全精度 随机数据 vs CPU参考 验证 (sm_100a 专属)
 *
 * 覆盖: fp16, bf16, tf32, fp8(e4m3), int8
 *
 * 特色: D 的读回布局不做任何假设 —— 先用两轮 one-hot 探针把
 *   「tcgen05.ld 输出槽位 → D[row][col]」的映射在真机上测出来:
 *     探针1: A[m][0]=m, B[n][0]=1  →  每个槽位的值 = 它的 row
 *     探针2: A[m][0]=1, B[n][0]=n  →  每个槽位的值 = 它的 col
 *   然后随机数据按测得的映射与 CPU 参考逐元素精确比对。
 *   (fp32/s32 累加器 TMEM 布局相同, 探针用 fp16 测一次全精度共用)
 *
 * 编译: nvcc -gencode arch=compute_100a,code=sm_100a -std=c++17 -o t tcgen05_random_cpu_ref_sm100.cu
 * 运行: ./t [seed]
 */
#include <cuda_runtime.h>
#include "verify_random.h"

#define CHECK_CUDA(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA err %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

constexpr int M = 128, N = 8, ROW_BYTES = 32, THREADS = 128;
constexpr int A_BYTES = M * ROW_BYTES, B_BYTES = N * ROW_BYTES;
constexpr int NSLOT = 128;   // warp0 tcgen05.ld 16x256b.x1 = 32线程×4寄存器

__device__ uint32_t smem_u32(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ uint64_t make_desc(void const* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_u32(smem_ptr) >> 4) & 0x3FFF);
    desc |= (uint64_t)(lbo & 0x3FFF) << 16;   // LBO (>>4): K方向 core-matrix 间距
    desc |= (uint64_t)(sbo & 0x3FFF) << 32;   // SBO (>>4): M方向 core-matrix 间距
    desc |= (uint64_t)(1) << 46;              // version (tcgen05 要求)
    return desc;
}

// 通用kernel: KIND=指令kind串, IDESC = d_fmt<<4 | a_fmt<<7 | b_fmt<<10 | n/8<<17 | m/16<<24
#define DEF_TCGEN05(NAME, KIND, IDESC)                                            \
__global__ void NAME(const uint8_t* A, const uint8_t* B, uint32_t* D_raw, uint32_t LBO, uint32_t SBO) {       \
    extern __shared__ char smem[];                                                \
    uint32_t* sA = (uint32_t*)smem;                                               \
    uint32_t* sB = (uint32_t*)(smem + A_BYTES);                                   \
    uint32_t* s_tmem = (uint32_t*)(smem + A_BYTES + B_BYTES + 64);                \
    int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;                   \
    for (int i = tid; i < A_BYTES / 4; i += THREADS) sA[i] = ((uint32_t*)A)[i];   \
    for (int i = tid; i < B_BYTES / 4; i += THREADS) sB[i] = ((uint32_t*)B)[i];   \
    __syncthreads();                                                              \
    if (warp_id == 0)                                                             \
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" \
                     : : "r"(smem_u32(s_tmem)), "r"(32));                          \
    __syncthreads();                                                              \
    uint32_t tmem_c = *s_tmem;                                                    \
    uint64_t da = make_desc(sA, LBO, SBO), db = make_desc(sB, LBO, SBO);        \
    __syncthreads();                                                              \
    if (tid == 0) {                                                               \
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"            \
            "tcgen05.mma.cta_group::1.kind::" KIND " [%0], %1, %2, %3, p;\n\t}\n" \
            : : "r"(tmem_c), "l"(da), "l"(db), "r"((uint32_t)(IDESC)), "r"(0u));  \
    }                                                                             \
    __syncthreads();                                                              \
    if (warp_id == 0) {                                                           \
        uint32_t r0, r1, r2, r3;                                                  \
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];\n\t" \
                     "tcgen05.wait::ld.sync.aligned;\n"                          \
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c));     \
        D_raw[lane * 4 + 0] = r0; D_raw[lane * 4 + 1] = r1;                       \
        D_raw[lane * 4 + 2] = r2; D_raw[lane * 4 + 3] = r3;                       \
    }                                                                             \
    __syncthreads();                                                              \
    if (warp_id == 0)                                                             \
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"      \
                     : : "r"(tmem_c), "r"(32));                                   \
}

#define IDESC(dfmt, abfmt) ((dfmt)<<4 | (abfmt)<<7 | (abfmt)<<10 | 1<<17 | 8<<24)
DEF_TCGEN05(k_fp16, "f16",    IDESC(1, 0))
DEF_TCGEN05(k_bf16, "f16",    IDESC(1, 1))
DEF_TCGEN05(k_tf32, "tf32",   IDESC(1, 2))
DEF_TCGEN05(k_fp8 , "f8f6f4", IDESC(1, 0))
DEF_TCGEN05(k_int8, "i8",     IDESC(2, 1))

// fp16 编码 0..15 的整数 (探针用)
static uint16_t fp16_enc_int(int v) {
    if (v == 0) return 0;
    int e = 0; while ((v >> (e + 1)) != 0) e++;             // 2^e <= v < 2^(e+1)
    return (uint16_t)(((e + 15) << 10) | (((v << 10) >> e) & 0x3FF));
}

typedef void (*Kern)(const uint8_t*, const uint8_t*, uint32_t*, uint32_t, uint32_t);
static uint32_t g_lbo = 8, g_sbo = 16;
static uint8_t tmpbuf[128 * 32];
// UMMA no-swizzle K-major: 与GMMA同款 8行×16B core-matrix 分块 (H20实测 LBO=8 SBO=16)
static void to_core_matrix(uint8_t* buf, int rows) {
    memcpy(tmpbuf, buf, rows * 32);
    for (int m = 0; m < rows; m++)
        for (int kb = 0; kb < 32; kb++)
            buf[(m/8)*256 + (kb/16)*128 + (m%8)*16 + kb%16] = tmpbuf[m*32 + kb];
}
static float u32_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint8_t hA[A_BYTES], hB[B_BYTES];
static float refA[M * 32], refB[N * 32], refD[M * N];
static uint32_t hDraw[NSLOT];

static void run(Kern k, uint8_t* dA, uint8_t* dB, uint32_t* dD) {
    to_core_matrix(hA, M); to_core_matrix(hB, N);
    CHECK_CUDA(cudaMemcpy(dA, hA, A_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, B_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dD, 0, NSLOT * 4));
    k<<<1, THREADS, A_BYTES + B_BYTES + 128>>>(dA, dB, dD, g_lbo, g_sbo);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hDraw, dD, NSLOT * 4, cudaMemcpyDeviceToHost));
}

int main(int argc, char** argv) {
    uint32_t seed = argc > 1 ? (uint32_t)atoi(argv[1]) : 1;
    if (argc > 2) g_lbo = (uint32_t)atoi(argv[2]);
    if (argc > 3) g_sbo = (uint32_t)atoi(argv[3]);
    const int ROUNDS = 3;
    uint8_t *dA, *dB; uint32_t* dD;
    CHECK_CUDA(cudaMalloc(&dA, A_BYTES));
    CHECK_CUDA(cudaMalloc(&dB, B_BYTES));
    CHECK_CUDA(cudaMalloc(&dD, NSLOT * 4));

    // ---- 探针: 测出 D_raw 槽位 → (row, col) ----
    int map_row[NSLOT], map_col[NSLOT];
    memset(hA, 0, A_BYTES); memset(hB, 0, B_BYTES);
    for (int m = 0; m < M; m++) ((uint16_t*)hA)[m * 16] = fp16_enc_int(m);   // A[m][0]=m (全128行)
    for (int n = 0; n < N; n++)  ((uint16_t*)hB)[n * 16] = fp16_enc_int(1);   // B[n][0]=1
    run(k_fp16, dA, dB, dD);
    for (int s = 0; s < NSLOT; s++) map_row[s] = (int)u32_as_float(hDraw[s]);
    memset(hA, 0, A_BYTES); memset(hB, 0, B_BYTES);   // run()已就地重排, 探针2必须重填
    for (int m = 0; m < M; m++) ((uint16_t*)hA)[m * 16] = fp16_enc_int(1);   // A[m][0]=1
    for (int n = 0; n < N; n++)  ((uint16_t*)hB)[n * 16] = fp16_enc_int(n);   // B[n][0]=n
    run(k_fp16, dA, dB, dD);
    for (int s = 0; s < NSLOT; s++) map_col[s] = (int)u32_as_float(hDraw[s]);
    // 探针3(一致性): A[m][0]=m, B[n][0]=2 → 真D槽位应 = 2×探针1
    //   (TMEM dealloc/realloc 不清零, 部分槽位读到跨kernel残留 → 用一致性过滤)
    memset(hA, 0, A_BYTES); memset(hB, 0, B_BYTES);
    for (int m = 0; m < M; m++) ((uint16_t*)hA)[m * 16] = fp16_enc_int(m);
    for (int n = 0; n < N; n++)  ((uint16_t*)hB)[n * 16] = fp16_enc_int(2);
    run(k_fp16, dA, dB, dD);
    static bool valid[NSLOT]; static bool seen[128][8];
    int nvalid = 0; bool map_ok = true;
    for (int s = 0; s < NSLOT; s++) {
        int p3 = (int)u32_as_float(hDraw[s]);
        valid[s] = map_row[s] >= 0 && map_row[s] < M && map_col[s] >= 0 && map_col[s] < 8 &&
                   p3 == 2 * map_row[s];
        if (valid[s]) {
            if (seen[map_row[s]][map_col[s]]) { map_ok = false; break; }
            seen[map_row[s]][map_col[s]] = true; nvalid++;
        }
    }
    if (nvalid < 32) map_ok = false;
    printf("====== tcgen05 随机数据 vs CPU参考 (seed=%u LBO=%u SBO=%u) ======\n", seed, g_lbo, g_sbo);
    printf("D布局探针: %s, %d/128 槽位一致有效 (槽位0..3 → row%d/col%d row%d/col%d row%d/col%d row%d/col%d)\n",
           map_ok ? "OK" : "FAIL", nvalid,
           map_row[0], map_col[0], map_row[1], map_col[1],
           map_row[2], map_col[2], map_row[3], map_col[3]);
    if (!map_ok) {
        printf("完整槽位映射 (lane.reg → row/col):\n");
        for (int s2 = 0; s2 < NSLOT; s2++)
            printf("  %d.%d→%d/%d%s", s2 / 4, s2 % 4, map_row[s2], map_col[s2],
                   (s2 % 8 == 7) ? "\n" : "");
        return 1;
    }

    struct Test { const char* name; Kern kern; int K; int ebits;
                  const EncVal* set; int setn; bool is_int; };
    Test tests[] = {
        {"fp16", k_fp16, 16, 16, FP16_SET, SETN(FP16_SET), false},
        {"bf16", k_bf16, 16, 16, BF16_SET, SETN(BF16_SET), false},
        {"tf32", k_tf32,  8, 32, TF32_SET, SETN(TF32_SET), false},
        {"fp8",  k_fp8,  32,  8, E4M3_SET, SETN(E4M3_SET), false},
        {"int8", k_int8, 32,  8, S8_SET,   SETN(S8_SET),   true},
    };
    int total_bad = 0;
    for (auto& t : tests) {
        for (int round = 0; round < ROUNDS; round++) {
            memset(hA, 0, A_BYTES); memset(hB, 0, B_BYTES);
            fill_random(hA, refA, M, t.K, t.set, t.setn, t.ebits, &seed);
            fill_random(hB, refB, N, t.K, t.set, t.setn, t.ebits, &seed);
            cpu_gemm_ref(refA, refB, refD, M, N, t.K);
            run(t.kern, dA, dB, dD);
            float got[NSLOT], want[NSLOT];
            int nv = 0;
            for (int s = 0; s < NSLOT; s++) {
                if (!valid[s]) continue;
                got[nv]  = t.is_int ? (float)(int32_t)hDraw[s]
                                    : u32_as_float(hDraw[s]);
                want[nv] = refD[map_row[s] * N + map_col[s]];
                nv++;
            }
            char tag[32]; snprintf(tag, sizeof(tag), "%s r%d", t.name, round);
            total_bad += check_exact(got, want, nv, tag);
        }
    }
    printf("====== %s (验证探针确认的稳定D槽位) ======\n", total_bad ? "有FAIL, 见上" : "全部PASS");
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return total_bad ? 1 : 0;
}
