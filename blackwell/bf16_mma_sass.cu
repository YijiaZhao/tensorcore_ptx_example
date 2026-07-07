// 最小 bf16 mma.sync,用于对比 sm_89 vs sm_120a 的 SASS
#include <cstdint>
__global__ void k(const uint32_t* A, const uint32_t* B, float* C) {
    uint32_t a0=A[0],a1=A[1],a2=A[2],a3=A[3], b0=B[0],b1=B[1];
    float c0=C[0],c1=C[1],c2=C[2],c3=C[3];
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    C[0]=c0; C[1]=c1; C[2]=c2; C[3]=c3;
}
