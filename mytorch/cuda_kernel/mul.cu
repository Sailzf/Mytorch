#include <cuda_runtime.h>
#include "mul.h"

// CUDA kernel for element-wise multiplication
__global__ void mul_kernel(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * y[idx];
    }
}

// CUDA kernel for element-wise multiplication backward
__global__ void mul_backward_kernel(const float* grad_output, const float* x, const float* y, 
                                  float* grad_x, float* grad_y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // For multiplication: 
        // grad_x = grad_output * y
        // grad_y = grad_output * x
        grad_x[idx] = grad_output[idx] * y[idx];
        grad_y[idx] = grad_output[idx] * x[idx];
    }
}

// Wrapper function for forward pass
void cuda_mul_forward(const float* x, const float* y, float* out, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    mul_kernel<<<num_blocks, block_size>>>(x, y, out, size);
}

// Wrapper function for backward pass
void cuda_mul_backward(const float* grad_output, const float* x, const float* y, 
                      float* grad_x, float* grad_y, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    mul_backward_kernel<<<num_blocks, block_size>>>(grad_output, x, y, grad_x, grad_y, size);
} 