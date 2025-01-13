#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "add.h"

namespace py = pybind11;

py::array_t<float> add_forward(py::array_t<float> x, py::array_t<float> y) {
    py::buffer_info x_buf = x.request();
    py::buffer_info y_buf = y.request();
    
    auto result = py::array_t<float>(x_buf.shape);
    py::buffer_info result_buf = result.request();
    
    cuda_add_forward(
        static_cast<float*>(x_buf.ptr),
        static_cast<float*>(y_buf.ptr),
        static_cast<float*>(result_buf.ptr),
        x_buf.shape[0]
    );
    
    return result;
}

std::tuple<py::array_t<float>, py::array_t<float>> add_backward(
    py::array_t<float> grad, py::array_t<float> x, py::array_t<float> y) {
    
    py::buffer_info grad_buf = grad.request();
    py::buffer_info x_buf = x.request();
    py::buffer_info y_buf = y.request();
    
    auto grad_x = py::array_t<float>(x_buf.shape);
    auto grad_y = py::array_t<float>(y_buf.shape);
    
    py::buffer_info grad_x_buf = grad_x.request();
    py::buffer_info grad_y_buf = grad_y.request();
    
    cuda_add_backward(
        static_cast<float*>(grad_buf.ptr),
        static_cast<float*>(x_buf.ptr),
        static_cast<float*>(y_buf.ptr),
        static_cast<float*>(grad_x_buf.ptr),
        static_cast<float*>(grad_y_buf.ptr),
        x_buf.shape[0]
    );
    
    return std::make_tuple(grad_x, grad_y);
}

PYBIND11_MODULE(cuda_add, m) {
    m.doc() = "CUDA implementation of addition operation";
    m.def("forward", &add_forward, "Forward pass of addition operation");
    m.def("backward", &add_backward, "Backward pass of addition operation");
} 