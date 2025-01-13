#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of CUDA functions
void cuda_matmul_forward(const float* A, const float* B, float* C,
                        int M, int N, int K);

void cuda_matmul_backward(const float* grad_C, const float* A, const float* B,
                         float* grad_A, float* grad_B,
                         int M, int N, int K);

#ifdef __cplusplus
}
#endif 