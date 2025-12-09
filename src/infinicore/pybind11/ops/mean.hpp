#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mean.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_mean(py::module &m) {
    m.def(
        "mean",
        &op::mean,
        py::arg("input"),
        py::arg("dim"),
        py::arg("keepdim"),
        R"doc(Mean reduction over a single dimension.

Higher-level handling of dim=None or tuple dims is done in the Python wrapper.)doc");

    m.def(
        "mean_",
        &op::mean_,
        py::arg("out"),
        py::arg("input"),
        py::arg("dim"),
        py::arg("keepdim"),
        R"doc(In-place mean reduction that writes the result into `out`.)doc");
}

} // namespace infinicore::ops


