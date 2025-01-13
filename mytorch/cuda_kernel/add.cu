#include <cuda_runtime.h>
#include "add.h"

// CUDA核函数：执行元素级别的加法运算
// x和y是输入数组，out是输出数组，size是数组大小
__global__ void add_kernel(const float* x, const float* y, float* z, int n) {
    // 计算当前线程处理的数组索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 执行元素加法运算
        z[idx] = x[idx] + y[idx];
    }
}

// CUDA核函数：计算加法操作的反向传播（梯度计算）
// grad_output是输出梯度，grad_x和grad_y是需要计算的输入梯度
__global__ void add_backward_kernel(const float* grad_output, float* grad_x, float* grad_y, int n) {
    // 计算当前线程处理的数组索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 加法操作的梯度直接传递
        // 因为 d(x+y)/dx = 1 且 d(x+y)/dy = 1
        grad_x[idx] = grad_output[idx];
        grad_y[idx] = grad_output[idx];
    }
}

// CPU端包装函数：启动前向传播的CUDA核函数
void cuda_add_forward(const float* x, const float* y, float* z, int n) {
    // 设置CUDA线程块大小和块数量
    int block_size = 256;  // 每个块包含256个线程
    int num_blocks = (n + block_size - 1) / block_size;  // 计算需要的块数
    
    // 启动CUDA核函数
    add_kernel<<<num_blocks, block_size>>>(x, y, z, n);
}

// CPU端包装函数：启动反向传播的CUDA核函数
void cuda_add_backward(const float* grad_output, const float* x, const float* y,
                      float* grad_x, float* grad_y, int n) {
    // 设置CUDA线程块大小和块数量
    int block_size = 256;  // 每个块包含256个线程
    int num_blocks = (n + block_size - 1) / block_size;  // 计算需要的块数
    
    // 启动CUDA核函数
    add_backward_kernel<<<num_blocks, block_size>>>(grad_output, grad_x, grad_y, n);
} 