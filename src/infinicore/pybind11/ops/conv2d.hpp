#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/conv2d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_conv2d(py::module &m) {
    m.def(
        "conv2d",
        [](Tensor input,
           Tensor weight,
           std::optional<Tensor> bias,
           long long stride_h,
           long long stride_w,
           long long pad_h,
           long long pad_w,
           long long dilation_h,
           long long dilation_w,
           long long groups) {
            return op::conv2d(input,
                              weight,
                              bias,
                              stride_h,
                              stride_w,
                              pad_h,
                              pad_w,
                              dilation_h,
                              dilation_w,
                              groups);
        },
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias").none(true) = std::nullopt,
        py::arg("stride_h"),
        py::arg("stride_w"),
        py::arg("pad_h"),
        py::arg("pad_w"),
        py::arg("dilation_h"),
        py::arg("dilation_w"),
        py::arg("groups"));
}

} // namespace infinicore::ops


