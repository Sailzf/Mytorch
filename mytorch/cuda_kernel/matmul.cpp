#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matmul.h"

namespace py = pybind11;

// Wrapper function for the forward pass
py::array_t<float> matmul_forward(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info A_buf = A.request();
    py::buffer_info B_buf = B.request();
    
    if (A_buf.ndim != 2 || B_buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    int M = A_buf.shape[0];  // A的行数
    int K = A_buf.shape[1];  // A的列数
    int N = B_buf.shape[1];  // B的列数
    
    if (B_buf.shape[0] != K) {
        throw std::runtime_error("Incompatible matrix dimensions");
    }
    
    auto result = py::array_t<float>({M, N});
    py::buffer_info result_buf = result.request();
    
    cuda_matmul_forward(
        static_cast<float*>(A_buf.ptr),
        static_cast<float*>(B_buf.ptr),
        static_cast<float*>(result_buf.ptr),
        M, N, K
    );
    
    return result;
}

// Wrapper function for the backward pass
std::tuple<py::array_t<float>, py::array_t<float>> matmul_backward(
    py::array_t<float> grad_C, py::array_t<float> A, py::array_t<float> B) {
    
    py::buffer_info grad_buf = grad_C.request();
    py::buffer_info A_buf = A.request();
    py::buffer_info B_buf = B.request();
    
    int M = A_buf.shape[0];
    int K = A_buf.shape[1];
    int N = B_buf.shape[1];
    
    auto grad_A = py::array_t<float>({M, K});
    auto grad_B = py::array_t<float>({K, N});
    
    py::buffer_info grad_A_buf = grad_A.request();
    py::buffer_info grad_B_buf = grad_B.request();
    
    cuda_matmul_backward(
        static_cast<float*>(grad_buf.ptr),
        static_cast<float*>(A_buf.ptr),
        static_cast<float*>(B_buf.ptr),
        static_cast<float*>(grad_A_buf.ptr),
        static_cast<float*>(grad_B_buf.ptr),
        M, N, K
    );
    
    return std::make_tuple(grad_A, grad_B);
}

PYBIND11_MODULE(cuda_matmul, m) {
    m.doc() = "CUDA implementation of matrix multiplication operation"; 
    m.def("forward", &matmul_forward, "Forward pass of matrix multiplication operation");
    m.def("backward", &matmul_backward, "Backward pass of matrix multiplication operation");
} 