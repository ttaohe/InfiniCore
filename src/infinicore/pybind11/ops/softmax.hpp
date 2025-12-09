#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/softmax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_softmax(py::module &m) {
    m.def(
        "softmax",
        &op::softmax,
        py::arg("input"),
        py::arg("dim"),
        R"doc(Generic softmax over a given dimension (InfiniCore frontend).

Semantics:
  Equivalent to torch.nn.functional.softmax(input, dim=dim) for supported dtypes/devices.
)doc");

    m.def(
        "softmax_",
        &op::softmax_,
        py::arg("out"),
        py::arg("input"),
        py::arg("dim"),
        R"doc(In-place softmax that writes the result into `out`.)doc");

    m.def(
        "log_softmax",
        &op::log_softmax,
        py::arg("input"),
        py::arg("dim"),
        R"doc(Generic log_softmax over a given dimension (InfiniCore frontend).)doc");

    m.def(
        "log_softmax_",
        &op::log_softmax_,
        py::arg("out"),
        py::arg("input"),
        py::arg("dim"),
        R"doc(In-place log_softmax that writes the result into `out`.)doc");
}

} // namespace infinicore::ops


