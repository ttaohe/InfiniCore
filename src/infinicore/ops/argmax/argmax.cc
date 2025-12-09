#include "infinicore/ops/argmax.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<ArgMax::schema> &ArgMax::dispatcher() {
    static common::OpDispatcher<ArgMax::schema> dispatcher_;
    return dispatcher_;
}

void ArgMax::execute(Tensor out, Tensor input, long long dim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);

    // 归一化 dim（支持负数）
    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("ArgMax does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("ArgMax dim is out of range.");
    }

    // 仅支持输入为 F32 / I32，输出为 I64
    auto in_dtype = input->dtype();
    if (in_dtype != DataType::F32 && in_dtype != DataType::I32) {
        throw std::runtime_error("ArgMax only supports F32 and I32 input dtypes for now.");
    }
    if (out->dtype() != DataType::I64) {
        throw std::runtime_error("ArgMax output tensor must have dtype=I64.");
    }

    // 保证输入为连续内存，便于 kernel 简化实现
    Tensor input_contiguous =
        input->is_contiguous() ? input : input->contiguous();

    infinicore::context::setDevice(out->device());

    auto device_type = out->device().getType();
    auto fn = dispatcher().lookup(device_type);
    if (fn == nullptr) {
        throw std::runtime_error("No ArgMax implementation found for device type.");
    }

    fn(out, input_contiguous, d);
}

Tensor argmax(Tensor input, long long dim) {
    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("ArgMax does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("ArgMax dim is out of range.");
    }

    Shape in_shape = input->shape();
    Shape out_shape;
    out_shape.reserve(ndim > 0 ? ndim - 1 : 0);
    for (Size i = 0; i < ndim; ++i) {
        if (static_cast<long long>(i) == d) {
            continue;
        }
        out_shape.push_back(in_shape[i]);
    }

    auto out = Tensor::empty(out_shape, DataType::I64, input->device());
    argmax_(out, input, d);
    return out;
}

void argmax_(Tensor out, Tensor input, long long dim) {
    ArgMax::execute(out, input, dim);
}

} // namespace infinicore::op


