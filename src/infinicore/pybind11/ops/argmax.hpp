#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/argmax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_argmax(py::module &m) {
    m.def(
        "argmax",
        &op::argmax,
        py::arg("input"),
        py::arg("dim"),
        R"doc(ArgMax reduction along a specific dimension.

This operator reduces `input` along the given `dim` and returns an index tensor
with dtype=I64. The output shape is `input.shape` with `dim` removed.)doc");

    m.def(
        "argmax_",
        &op::argmax_,
        py::arg("out"),
        py::arg("input"),
        py::arg("dim"),
        R"doc(In-place ArgMax that writes the result into the provided `out` tensor.)doc");
}

} // namespace infinicore::ops


