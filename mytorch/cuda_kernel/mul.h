#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of CUDA functions
void cuda_mul_forward(const float* x, const float* y, float* out, int size);
void cuda_mul_backward(const float* grad_output, const float* x, const float* y, 
                      float* grad_x, float* grad_y, int size);

#ifdef __cplusplus
}
#endif 