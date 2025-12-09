#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sum.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sum(py::module &m) {
    m.def(
        "sum",
        &op::sum,
        py::arg("input"),
        py::arg("dim"),
        py::arg("keepdim"),
        R"doc(Sum reduction over a single dimension.

Higher-level handling of dim=None or tuple dims is done in the Python wrapper.)doc");

    m.def(
        "sum_",
        &op::sum_,
        py::arg("out"),
        py::arg("input"),
        py::arg("dim"),
        py::arg("keepdim"),
        R"doc(In-place sum reduction that writes the result into `out`.)doc");
}

} // namespace infinicore::ops


