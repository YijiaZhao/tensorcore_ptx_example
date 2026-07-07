// shl_demo.cu — 演示 shl(左移)两种写法:CUDA C 的 << 和 内联 PTX 的 shl 指令
#include <cstdio>
#include <cstdint>

__global__ void demo(const uint32_t* in, uint32_t* out_c, uint32_t* out_ptx, int n) {
    int i = threadIdx.x;
    uint32_t a = in[i];

    // 写法 1:CUDA C 运算符 <<(编译器自动生成 shl)
    out_c[i] = a << n;

    // 写法 2:内联 PTX,直接写 shl.b32
    uint32_t d;
    asm("shl.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(n));
    out_ptx[i] = d;
}

int main() {
    const int N = 6, shift = 3;             // 左移 3 位 = ×8
    uint32_t h[N] = {1, 5, 7, 16, 100, 0xFF};

    uint32_t *d_in, *d_c, *d_ptx;
    cudaMalloc(&d_in, N * 4); cudaMalloc(&d_c, N * 4); cudaMalloc(&d_ptx, N * 4);
    cudaMemcpy(d_in, h, N * 4, cudaMemcpyHostToDevice);

    demo<<<1, N>>>(d_in, d_c, d_ptx, shift);
    cudaDeviceSynchronize();

    uint32_t rc[N], rp[N];
    cudaMemcpy(rc, d_c, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(rp, d_ptx, N * 4, cudaMemcpyDeviceToHost);

    printf("shift left by %d (= x%d)\n", shift, 1 << shift);
    printf("%-8s %-12s %-12s %-6s\n", "input", "a<<n (C)", "shl.b32(PTX)", "check");
    for (int i = 0; i < N; ++i)
        printf("%-8u %-12u %-12u %s\n", h[i], rc[i], rp[i],
               (rc[i] == rp[i] && rc[i] == (h[i] << shift)) ? "OK" : "FAIL");
    cudaFree(d_in); cudaFree(d_c); cudaFree(d_ptx);
    return 0;
}
