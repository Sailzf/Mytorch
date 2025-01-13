#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "conv.h"

namespace py = pybind11;

// Wrapper function for the forward pass
py::array_t<float> conv_forward(py::array_t<float> input, py::array_t<float> weight,
                               int stride, int pad) {
    py::buffer_info input_buf = input.request();
    py::buffer_info weight_buf = weight.request();
    
    if (input_buf.ndim != 4 || weight_buf.ndim != 4) {
        throw std::runtime_error("Number of dimensions must be 4");
    }
    
    int N = input_buf.shape[0];   // batch size
    int C = input_buf.shape[1];   // input channels
    int H = input_buf.shape[2];   // input height
    int W = input_buf.shape[3];   // input width
    
    int K = weight_buf.shape[0];   // number of filters
    int filter_h = weight_buf.shape[2];  // filter height
    int filter_w = weight_buf.shape[3];  // filter width
    
    if (weight_buf.shape[1] != C) {
        throw std::runtime_error("Incompatible number of channels");
    }
    
    int out_h = (H + 2 * pad - filter_h) / stride + 1;
    int out_w = (W + 2 * pad - filter_w) / stride + 1;
    
    auto result = py::array_t<float>({N, K, out_h, out_w});
    py::buffer_info result_buf = result.request();
    
    cuda_conv_forward(
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(weight_buf.ptr),
        static_cast<float*>(result_buf.ptr),
        N, C, H, W,
        K, filter_h, filter_w,
        stride, pad
    );
    
    return result;
}

// Wrapper function for the backward pass
std::tuple<py::array_t<float>, py::array_t<float>> conv_backward(
    py::array_t<float> grad_output, py::array_t<float> input,
    py::array_t<float> weight, int stride, int pad) {
    
    py::buffer_info grad_buf = grad_output.request();
    py::buffer_info input_buf = input.request();
    py::buffer_info weight_buf = weight.request();
    
    int N = input_buf.shape[0];
    int C = input_buf.shape[1];
    int H = input_buf.shape[2];
    int W = input_buf.shape[3];
    
    int K = weight_buf.shape[0];
    int filter_h = weight_buf.shape[2];
    int filter_w = weight_buf.shape[3];
    
    auto grad_input = py::array_t<float>({N, C, H, W});
    auto grad_weight = py::array_t<float>({K, C, filter_h, filter_w});
    
    py::buffer_info grad_input_buf = grad_input.request();
    py::buffer_info grad_weight_buf = grad_weight.request();
    
    cuda_conv_backward(
        static_cast<float*>(grad_buf.ptr),
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(weight_buf.ptr),
        static_cast<float*>(grad_input_buf.ptr),
        static_cast<float*>(grad_weight_buf.ptr),
        N, C, H, W,
        K, filter_h, filter_w,
        stride, pad
    );
    
    return std::make_tuple(grad_input, grad_weight);
}

PYBIND11_MODULE(cuda_conv, m) {
    m.doc() = "CUDA implementation of convolution operation"; 
    m.def("forward", &conv_forward, "Forward pass of convolution operation",
          py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("pad"));
    m.def("backward", &conv_backward, "Backward pass of convolution operation",
          py::arg("grad_output"), py::arg("input"), py::arg("weight"),
          py::arg("stride"), py::arg("pad"));
} 