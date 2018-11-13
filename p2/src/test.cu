#include <stdio.h>
                                                                                                                                         
#define N 1029 
#define T_P_B 512

__global__ void double_it(int *a, int *b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        b[idx] = 2 * a[idx];
    }
}

int main() {
    int *d_a, *d_b;

    cudaMalloc((void**)&d_a, N*sizeof(N));
    cudaMalloc((void**)&d_b, N*sizeof(N));
 
    int a[N], b[N];

    for(int i = 0; i < N; ++i) {
        a[i] = i;
    }
 
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
 
    double_it<<<(N + T_P_B-1) / T_P_B, T_P_B>>>(d_a, d_b, N);
 
    cudaMemcpy(b, d_b, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < N; ++i) {
        printf("%d x2 = %d\n", a[i], b[i]);
    }
 
    cudaFree(d_a);
    cudaFree(d_b);
 }