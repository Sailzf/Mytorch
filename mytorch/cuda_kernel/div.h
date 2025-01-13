#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of CUDA functions
void cuda_div_forward(const float* x, const float* y, float* out, int size);
void cuda_div_backward(const float* grad_output, const float* x, const float* y, 
                      float* grad_x, float* grad_y, int size);
void cuda_div_from_const_forward(const float* x, float c, float* out, int size);
void cuda_div_from_const_backward(const float* grad_output, const float* x, float c, 
                                float* grad_x, int size);

#ifdef __cplusplus
}
#endif 