#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/topk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_topk(py::module &m) {
    m.def(
        "topk",
        &op::topk,
        py::arg("input"),
        py::arg("k"),
        py::arg("dim") = -1,
        py::arg("largest") = true,
        py::arg("sorted") = true,
        R"doc(TopK operator.

Returns (values, indices) similar to torch.topk, currently CPU-only.)doc");
}

} // namespace infinicore::ops


