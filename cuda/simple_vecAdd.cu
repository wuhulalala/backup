#include <stdio.h>
#include <cuda_runtime.h>

// Kernel定义
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    const int N = 5;  // 向量大小
    float h_A[N], h_B[N], h_C[N];  // 主机内存
    float *d_A, *d_B, *d_C;       // 设备内存
    
    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_A[i] = i;      // A = [0, 1, 2, 3, 4]
        h_B[i] = i * 10; // B = [0, 10, 20, 30, 40]
    }
    
    // 在GPU上分配内存
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // 复制数据到GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动内核：1个线程块，N个线程
    VecAdd<<<1, N>>>(d_A, d_B, d_C);
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    // 将结果复制回CPU
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印结果
    printf("向量加法结果:\n");
    for (int i = 0; i < N; i++) {
        printf("C[%d] = %.1f + %.1f = %.1f\n", i, h_A[i], h_B[i], h_C[i]);
    }
    
    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}