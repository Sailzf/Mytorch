#include <cuda_runtime.h>
#include "conv.h"

// CUDA kernel for im2col operation
__global__ void im2col_kernel(const float* input, float* col,
                             int N, int C, int H, int W,
                             int filter_h, int filter_w,
                             int stride, int pad,
                             int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_h * out_w * C * filter_h * filter_w;
    
    if (idx < total) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % C;
        int n = idx / (out_w * out_h * C);
        
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        
        int col_offset = ((n * out_h * out_w + h_out * out_w + w_out) * 
                         (C * filter_h * filter_w));
        
        for (int fh = 0; fh < filter_h; ++fh) {
            for (int fw = 0; fw < filter_w; ++fw) {
                int h = h_in + fh;
                int w = w_in + fw;
                
                float val = 0;
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    val = input[((n * C + c) * H + h) * W + w];
                }
                
                col[col_offset + (c * filter_h * filter_w + fh * filter_w + fw)] = val;
            }
        }
    }
}

// CUDA kernel for col2im operation
__global__ void col2im_kernel(const float* col, float* input,
                             int N, int C, int H, int W,
                             int filter_h, int filter_w,
                             int stride, int pad,
                             int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    
    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (W * H * C);
        
        float sum = 0;
        
        for (int fh = 0; fh < filter_h; ++fh) {
            for (int fw = 0; fw < filter_w; ++fw) {
                int h_out = (h + pad - fh) / stride;
                int w_out = (w + pad - fw) / stride;
                
                if (h_out >= 0 && h_out < out_h &&
                    w_out >= 0 && w_out < out_w &&
                    (h + pad - fh) % stride == 0 &&
                    (w + pad - fw) % stride == 0) {
                    
                    int col_idx = ((n * out_h * out_w + h_out * out_w + w_out) *
                                 (C * filter_h * filter_w) +
                                 c * filter_h * filter_w +
                                 fh * filter_w + fw);
                    sum += col[col_idx];
                }
            }
        }
        
        input[idx] = sum;
    }
}

// Wrapper function for im2col
void cuda_im2col(const float* input, float* col,
                 int N, int C, int H, int W,
                 int filter_h, int filter_w,
                 int stride, int pad) {
    int out_h = (H + 2 * pad - filter_h) / stride + 1;
    int out_w = (W + 2 * pad - filter_w) / stride + 1;
    
    int total_threads = N * out_h * out_w * C * filter_h * filter_w;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    
    im2col_kernel<<<num_blocks, block_size>>>(
        input, col, N, C, H, W, filter_h, filter_w,
        stride, pad, out_h, out_w
    );
}

// Wrapper function for col2im
void cuda_col2im(const float* col, float* input,
                 int N, int C, int H, int W,
                 int filter_h, int filter_w,
                 int stride, int pad) {
    int out_h = (H + 2 * pad - filter_h) / stride + 1;
    int out_w = (W + 2 * pad - filter_w) / stride + 1;
    
    int total_threads = N * C * H * W;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    
    col2im_kernel<<<num_blocks, block_size>>>(
        col, input, N, C, H, W, filter_h, filter_w,
        stride, pad, out_h, out_w
    );
}

// Forward pass of convolution using im2col and matrix multiplication
void cuda_conv_forward(const float* input, const float* weight, float* output,
                      int N, int C, int H, int W,
                      int K, int filter_h, int filter_w,
                      int stride, int pad) {
    int out_h = (H + 2 * pad - filter_h) / stride + 1;
    int out_w = (W + 2 * pad - filter_w) / stride + 1;
    
    // Allocate memory for col
    float* col;
    int col_size = N * out_h * out_w * C * filter_h * filter_w;
    cudaMalloc(&col, col_size * sizeof(float));
    
    // Perform im2col
    cuda_im2col(input, col, N, C, H, W, filter_h, filter_w, stride, pad);
    
    // Reshape weight for matrix multiplication
    int M = K;  // number of filters
    int K_dim = C * filter_h * filter_w;  // input channels * filter height * filter width
    int N_dim = N * out_h * out_w;  // batch size * output height * output width
    
    // Perform matrix multiplication: output = weight * col
    // Using cublas or our matmul implementation
    // For now, we'll use a simple implementation
    
    // Clean up
    cudaFree(col);
}

// Backward pass of convolution
void cuda_conv_backward(const float* grad_output, const float* input, const float* weight,
                       float* grad_input, float* grad_weight,
                       int N, int C, int H, int W,
                       int K, int filter_h, int filter_w,
                       int stride, int pad) {
    int out_h = (H + 2 * pad - filter_h) / stride + 1;
    int out_w = (W + 2 * pad - filter_w) / stride + 1;
    
    // Allocate memory for col and col_grad
    float* col;
    float* col_grad;
    int col_size = N * out_h * out_w * C * filter_h * filter_w;
    cudaMalloc(&col, col_size * sizeof(float));
    cudaMalloc(&col_grad, col_size * sizeof(float));
    
    // Perform im2col
    cuda_im2col(input, col, N, C, H, W, filter_h, filter_w, stride, pad);
    
    // Compute gradients
    // grad_weight = grad_output * col.T
    // col_grad = weight.T * grad_output
    // grad_input = col2im(col_grad)
    
    // Clean up
    cudaFree(col);
    cudaFree(col_grad);
} 