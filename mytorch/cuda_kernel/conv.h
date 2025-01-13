#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of CUDA functions
void cuda_conv_forward(const float* input, const float* weight, float* output,
                      int N, int C, int H, int W,
                      int K, int filter_h, int filter_w,
                      int stride, int pad);

void cuda_conv_backward(const float* grad_output, const float* input, const float* weight,
                       float* grad_input, float* grad_weight,
                       int N, int C, int H, int W,
                       int K, int filter_h, int filter_w,
                       int stride, int pad);

void cuda_im2col(const float* input, float* col,
                 int N, int C, int H, int W,
                 int filter_h, int filter_w,
                 int stride, int pad);

void cuda_col2im(const float* col, float* input,
                 int N, int C, int H, int W,
                 int filter_h, int filter_w,
                 int stride, int pad);

#ifdef __cplusplus
}
#endif 