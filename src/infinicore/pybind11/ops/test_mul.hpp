#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/test_mul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_test_mul(py::module &m) {
    m.def(
        "test_mul",
        &op::test_mul,
        py::arg("a"),
        py::arg("b"),
        R"doc(Simple example operator: vector dot product `test_mul`.

This operator expects two 1D contiguous F32 tensors on CPU with the same shape
and returns a scalar tensor holding the dot product result.)doc");

    m.def(
        "test_mul_",
        &op::test_mul_,
        py::arg("out"),
        py::arg("a"),
        py::arg("b"),
        R"doc(In-place version of `test_mul`.

`out` must be a scalar F32 tensor on CPU. The result of `dot(a, b)` will be
written into `out`.)doc");
}

} // namespace infinicore::ops


