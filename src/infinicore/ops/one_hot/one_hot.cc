#include "infinicore/ops/one_hot.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<OneHot::schema> &OneHot::dispatcher() {
    static common::OpDispatcher<OneHot::schema> dispatcher_;
    return dispatcher_;
}

void OneHot::execute(Tensor out, Tensor indices, long long num_classes) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, indices);

    if (indices->dtype() != DataType::I64) {
        throw std::runtime_error("OneHot expects indices tensor with dtype=I64.");
    }
    if (out->dtype() != DataType::I64) {
        throw std::runtime_error("OneHot expects output tensor with dtype=I64.");
    }

    infinicore::context::setDevice(out->device());

    auto device_type = out->device().getType();
    auto fn = dispatcher().lookup(device_type);
    if (fn == nullptr) {
        throw std::runtime_error("No OneHot implementation found for device type.");
    }

    fn(out, indices, num_classes);
}

Tensor one_hot(Tensor indices, long long num_classes) {
    // 当前仅支持 CPU 实现，防止误用其他设备
    if (indices->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("OneHot currently only supports CPU device.");
    }

    // 确保 indices 连续
    Tensor idx = indices->is_contiguous() ? indices : indices->contiguous();

    if (idx->dtype() != DataType::I64) {
        throw std::runtime_error("OneHot expects indices tensor with dtype=I64.");
    }

    Size numel = idx->numel();
    const long long *idx_ptr = reinterpret_cast<const long long *>(idx->data());

    long long classes = num_classes;
    if (classes <= 0) {
        // 自动根据最大索引推断 num_classes
        long long max_val = 0;
        for (Size i = 0; i < numel; ++i) {
            max_val = std::max(max_val, idx_ptr[i]);
        }
        classes = max_val + 1;
    }

    if (classes <= 0) {
        throw std::runtime_error("OneHot inferred num_classes must be positive.");
    }

    // 输出 shape = indices.shape + [num_classes]
    Shape out_shape = idx->shape();
    out_shape.push_back(static_cast<Size>(classes));

    auto out = Tensor::zeros(out_shape, DataType::I64, idx->device());
    one_hot_(out, idx, classes);
    return out;
}

void one_hot_(Tensor out, Tensor indices, long long num_classes) {
    OneHot::execute(out, indices, num_classes);
}

} // namespace infinicore::op


