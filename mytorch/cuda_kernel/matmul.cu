#include <cuda_runtime.h>
#include "matmul.h"

// CUDA kernel for matrix multiplication
// C = A * B
// A: M x K matrix
// B: K x N matrix
// C: M x N matrix
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            int M, int N, int K) {
    // 使用共享内存来提高性能
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];
    
    // 计算当前线程负责的输出元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int tile = 0; tile < (K + 31) / 32; ++tile) {
        // 加载数据到共享内存
        if (row < M && tile * 32 + threadIdx.x < K)
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile * 32 + threadIdx.x];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && tile * 32 + threadIdx.y < K)
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // 计算当前块的乘积
        for (int k = 0; k < 32; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CUDA kernel for matrix multiplication backward
// For C = A * B:
// grad_A = grad_C * B.T
// grad_B = A.T * grad_C
__global__ void matmul_backward_kernel_A(const float* grad_C, const float* B,
                                       float* grad_A, int M, int N, int K) {
    __shared__ float shared_grad_C[32][32];
    __shared__ float shared_B[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + 31) / 32; ++tile) {
        if (row < M && tile * 32 + threadIdx.x < N)
            shared_grad_C[threadIdx.y][threadIdx.x] = grad_C[row * N + tile * 32 + threadIdx.x];
        else
            shared_grad_C[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < K && tile * 32 + threadIdx.y < N)
            shared_B[threadIdx.y][threadIdx.x] = B[col * N + tile * 32 + threadIdx.y];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < 32; ++k) {
            sum += shared_grad_C[threadIdx.y][k] * shared_B[threadIdx.x][k];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < K) {
        grad_A[row * K + col] = sum;
    }
}

__global__ void matmul_backward_kernel_B(const float* A, const float* grad_C,
                                       float* grad_B, int M, int N, int K) {
    __shared__ float shared_A[32][32];
    __shared__ float shared_grad_C[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (M + 31) / 32; ++tile) {
        if (tile * 32 + threadIdx.y < M && row < K)
            shared_A[threadIdx.y][threadIdx.x] = A[(tile * 32 + threadIdx.y) * K + row];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (tile * 32 + threadIdx.y < M && col < N)
            shared_grad_C[threadIdx.y][threadIdx.x] = grad_C[(tile * 32 + threadIdx.y) * N + col];
        else
            shared_grad_C[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < 32; ++k) {
            sum += shared_A[k][threadIdx.y] * shared_grad_C[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < K && col < N) {
        grad_B[row * N + col] = sum;
    }
}

// Wrapper function for forward pass
void cuda_matmul_forward(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    dim3 block_size(32, 32);
    dim3 num_blocks((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);
    
    matmul_kernel<<<num_blocks, block_size>>>(A, B, C, M, N, K);
}

// Wrapper functions for backward pass
void cuda_matmul_backward(const float* grad_C, const float* A, const float* B,
                         float* grad_A, float* grad_B,
                         int M, int N, int K) {
    dim3 block_size(32, 32);
    
    // Compute grad_A
    dim3 num_blocks_A((K + block_size.x - 1) / block_size.x,
                     (M + block_size.y - 1) / block_size.y);
    matmul_backward_kernel_A<<<num_blocks_A, block_size>>>(grad_C, B, grad_A, M, N, K);
    
    // Compute grad_B
    dim3 num_blocks_B((N + block_size.x - 1) / block_size.x,
                     (K + block_size.y - 1) / block_size.y);
    matmul_backward_kernel_B<<<num_blocks_B, block_size>>>(A, grad_C, grad_B, M, N, K);
} 