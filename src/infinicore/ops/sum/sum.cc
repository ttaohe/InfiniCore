#include "infinicore/ops/sum.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Sum::schema> &Sum::dispatcher() {
    static common::OpDispatcher<Sum::schema> dispatcher_;
    return dispatcher_;
}

void Sum::execute(Tensor out, Tensor input, long long dim, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);

    auto device = input->device();
    infinicore::context::setDevice(device);

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("sum does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("sum: dim out of range.");
    }

    auto in_dtype = input->dtype();
    if (!(in_dtype == DataType::F16 || in_dtype == DataType::BF16 || in_dtype == DataType::F32)) {
        throw std::runtime_error("sum: only F16/BF16/F32 input dtypes are supported.");
    }
    if (out->dtype() != in_dtype) {
        throw std::runtime_error("sum: out dtype must equal input dtype.");
    }

    const auto &in_shape = input->shape();
    Shape expected_shape;
    if (keepdim) {
        expected_shape = in_shape;
        expected_shape[static_cast<Size>(d)] = 1;
    } else {
        expected_shape.reserve(ndim - 1);
        for (Size i = 0; i < ndim; ++i) {
            if (static_cast<long long>(i) == d) {
                continue;
            }
            expected_shape.push_back(in_shape[i]);
        }
    }

    if (out->shape() != expected_shape) {
        throw std::runtime_error("sum: output shape mismatch.");
    }

    // 确保输入为连续内存，便于 kernel 简化实现
    Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

    // 为 out 选择一个工作区：若 out 连续则直接使用，否则创建临时 tensor，计算完成后 copy 回 out
    Tensor work_out = out;
    bool need_copy_back = false;
    if (!out->is_contiguous()) {
        work_out = Tensor::empty(expected_shape, in_dtype, device);
        need_copy_back = true;
    }

    auto fn = dispatcher().lookup(device.getType());
    if (fn == nullptr) {
        throw std::runtime_error("No Sum implementation found for device type.");
    }

    fn(work_out, input_contiguous, d, keepdim);

    if (need_copy_back) {
        out->copy_from(work_out);
    }
}

Tensor sum(Tensor input, long long dim, bool keepdim) {
    auto device = input->device();
    auto dtype = input->dtype();

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("sum does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("sum: dim out of range.");
    }

    const auto &in_shape = input->shape();
    Shape out_shape;
    if (keepdim) {
        out_shape = in_shape;
        out_shape[static_cast<Size>(d)] = 1;
    } else {
        out_shape.reserve(ndim - 1);
        for (Size i = 0; i < ndim; ++i) {
            if (static_cast<long long>(i) == d) {
                continue;
            }
            out_shape.push_back(in_shape[i]);
        }
    }

    auto out = Tensor::empty(out_shape, dtype, device);
    sum_(out, input, dim, keepdim);
    return out;
}

void sum_(Tensor out, Tensor input, long long dim, bool keepdim) {
    Sum::execute(out, input, dim, keepdim);
}

} // namespace infinicore::op


