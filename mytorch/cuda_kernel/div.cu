#include <cuda_runtime.h>
#include "div.h"

// CUDA kernel for element-wise division
__global__ void div_kernel(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / y[idx];
    }
}

// CUDA kernel for element-wise division backward
__global__ void div_backward_kernel(const float* grad_output, const float* x, const float* y, 
                                  float* grad_x, float* grad_y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // For division:
        // grad_x = grad_output / y
        // grad_y = -grad_output * x / (y * y)
        float y_val = y[idx];
        grad_x[idx] = grad_output[idx] / y_val;
        grad_y[idx] = -grad_output[idx] * x[idx] / (y_val * y_val);
    }
}

// CUDA kernel for division from constant
__global__ void div_from_const_kernel(const float* x, float c, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = c / x[idx];
    }
}

// CUDA kernel for division from constant backward
__global__ void div_from_const_backward_kernel(const float* grad_output, const float* x, 
                                             float c, float* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        grad_x[idx] = -c * grad_output[idx] / (x_val * x_val);
    }
}

// Wrapper function for forward pass
void cuda_div_forward(const float* x, const float* y, float* out, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    div_kernel<<<num_blocks, block_size>>>(x, y, out, size);
}

// Wrapper function for backward pass
void cuda_div_backward(const float* grad_output, const float* x, const float* y, 
                      float* grad_x, float* grad_y, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    div_backward_kernel<<<num_blocks, block_size>>>(grad_output, x, y, grad_x, grad_y, size);
}

// Wrapper function for division from constant forward
void cuda_div_from_const_forward(const float* x, float c, float* out, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    div_from_const_kernel<<<num_blocks, block_size>>>(x, c, out, size);
}

// Wrapper function for division from constant backward
void cuda_div_from_const_backward(const float* grad_output, const float* x, float c, 
                                float* grad_x, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    div_from_const_backward_kernel<<<num_blocks, block_size>>>(grad_output, x, c, grad_x, size);
} 