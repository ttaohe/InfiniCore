#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sigmoid.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sigmoid(py::module &m) {
    m.def("sigmoid",
          &op::sigmoid,
          py::arg("input"),
          R"doc(Sigmoid activation function.)doc");

    m.def("sigmoid_",
          &op::sigmoid_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place Sigmoid activation function with explicit output.)doc");
}

} // namespace infinicore::ops


