#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/one_hot.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_one_hot(py::module &m) {
    m.def(
        "one_hot",
        &op::one_hot,
        py::arg("indices"),
        py::arg("num_classes") = -1,
        R"doc(One-hot encoding of integer tensor `indices`.

Arguments:
  indices (Tensor): Integer tensor of indices (dtype=I64).
  num_classes (int, optional): Number of classes. If <=0, it will be inferred
      as `max(indices) + 1` on CPU.

Returns:
  Tensor: One-hot encoded tensor with shape `indices.shape + [num_classes]`
      and dtype=I64.)doc");

    m.def(
        "one_hot_",
        &op::one_hot_,
        py::arg("out"),
        py::arg("indices"),
        py::arg("num_classes"),
        R"doc(In-place one-hot encoding that writes result into `out`.)doc");
}

} // namespace infinicore::ops


