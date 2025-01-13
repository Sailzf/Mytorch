#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void cuda_add_forward(const float* x, const float* y, float* z, int n);

void cuda_add_backward(const float* grad_output, const float* x, const float* y,
                      float* grad_x, float* grad_y, int n);

#ifdef __cplusplus
}
#endif 