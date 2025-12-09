#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/scatter_add.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scatter_add(py::module &m) {
    m.def(
        "scatter_add",
        &op::scatter_add,
        py::arg("input"),
        py::arg("dim"),
        py::arg("index"),
        py::arg("src"),
        R"doc(Scatter-add operation.

Semantics (CPU-only for now):
  out = input.clone()
  out.scatter_add_(dim, index, src)

This function returns `out` and does not modify `input`.)doc");

    m.def(
        "scatter_add_",
        &op::scatter_add_,
        py::arg("out"),
        py::arg("input"),
        py::arg("dim"),
        py::arg("index"),
        py::arg("src"),
        R"doc(In-place scatter-add into the provided `out` tensor.)doc");
}

} // namespace infinicore::ops


