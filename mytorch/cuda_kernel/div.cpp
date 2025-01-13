#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "div.h"

namespace py = pybind11;

// Wrapper function for the forward pass
py::array_t<float> div_forward(py::array_t<float> x, py::array_t<float> y) {
    py::buffer_info x_buf = x.request();
    py::buffer_info y_buf = y.request();
    
    if (x_buf.size != y_buf.size) {
        throw std::runtime_error("Input shapes must match");
    }
    
    auto result = py::array_t<float>(x_buf.size);
    py::buffer_info result_buf = result.request();
    
    cuda_div_forward(
        static_cast<float*>(x_buf.ptr),
        static_cast<float*>(y_buf.ptr),
        static_cast<float*>(result_buf.ptr),
        x_buf.size
    );
    
    return result;
}

// Wrapper function for the backward pass
std::tuple<py::array_t<float>, py::array_t<float>> div_backward(
    py::array_t<float> grad_output, py::array_t<float> x, py::array_t<float> y) {
    
    py::buffer_info grad_buf = grad_output.request();
    py::buffer_info x_buf = x.request();
    py::buffer_info y_buf = y.request();
    
    auto grad_x = py::array_t<float>(x_buf.size);
    auto grad_y = py::array_t<float>(y_buf.size);
    
    py::buffer_info grad_x_buf = grad_x.request();
    py::buffer_info grad_y_buf = grad_y.request();
    
    cuda_div_backward(
        static_cast<float*>(grad_buf.ptr),
        static_cast<float*>(x_buf.ptr),
        static_cast<float*>(y_buf.ptr),
        static_cast<float*>(grad_x_buf.ptr),
        static_cast<float*>(grad_y_buf.ptr),
        grad_buf.size
    );
    
    return std::make_tuple(grad_x, grad_y);
}

// Wrapper function for division from constant forward
py::array_t<float> div_from_const_forward(py::array_t<float> x, float c) {
    py::buffer_info x_buf = x.request();
    
    auto result = py::array_t<float>(x_buf.size);
    py::buffer_info result_buf = result.request();
    
    cuda_div_from_const_forward(
        static_cast<float*>(x_buf.ptr),
        c,
        static_cast<float*>(result_buf.ptr),
        x_buf.size
    );
    
    return result;
}

// Wrapper function for division from constant backward
py::array_t<float> div_from_const_backward(
    py::array_t<float> grad_output, py::array_t<float> x, float c) {
    
    py::buffer_info grad_buf = grad_output.request();
    py::buffer_info x_buf = x.request();
    
    auto grad_x = py::array_t<float>(x_buf.size);
    py::buffer_info grad_x_buf = grad_x.request();
    
    cuda_div_from_const_backward(
        static_cast<float*>(grad_buf.ptr),
        static_cast<float*>(x_buf.ptr),
        c,
        static_cast<float*>(grad_x_buf.ptr),
        grad_buf.size
    );
    
    return grad_x;
}

PYBIND11_MODULE(cuda_div, m) {
    m.doc() = "CUDA implementation of division operation"; 
    m.def("forward", &div_forward, "Forward pass of division operation");
    m.def("backward", &div_backward, "Backward pass of division operation");
    m.def("const_forward", &div_from_const_forward, "Forward pass of division from constant");
    m.def("const_backward", &div_from_const_backward, "Backward pass of division from constant");
} 