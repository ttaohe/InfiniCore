#include "infinicore/ops/one_hot.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op::one_hot_impl::cpu {

void calculate(Tensor out, Tensor indices, long long num_classes) {
    if (out->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("OneHot CPU kernel only supports CPU device.");
    }

    if (indices->dtype() != DataType::I64 || out->dtype() != DataType::I64) {
        throw std::runtime_error("OneHot CPU kernel expects I64 dtype for both indices and output.");
    }

    // indices: 任意 shape，out: indices.shape + [num_classes]
    Size numel = indices->numel();

    const long long *idx_ptr = reinterpret_cast<const long long *>(indices->data());
    long long *out_ptr = reinterpret_cast<long long *>(out->data());

    const auto &out_shape = out->shape();
    if (out_shape.empty()) {
        throw std::runtime_error("OneHot output tensor must have rank >= 1.");
    }

    Size classes = out_shape.back();

    if (num_classes > 0 && static_cast<Size>(num_classes) > classes) {
        throw std::runtime_error("OneHot: provided num_classes is larger than output last dimension.");
    }

    Size effective_classes = static_cast<Size>(num_classes > 0 ? num_classes : static_cast<long long>(classes));

    // 为避免依赖上层是否正确清零，这里显式将输出张量全部写 0
    Size total_elems = out->numel();
    for (Size i = 0; i < total_elems; ++i) {
        out_ptr[i] = 0;
    }

    // 将对应位置写成 1
    for (Size i = 0; i < numel; ++i) {
        long long cls = idx_ptr[i];
        if (cls < 0 || static_cast<Size>(cls) >= effective_classes) {
            throw std::runtime_error("OneHot: index out of range of num_classes.");
        }

        Size offset = i * classes + static_cast<Size>(cls);
        out_ptr[offset] = 1;
    }
}

static bool registered = []() {
    OneHot::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::one_hot_impl::cpu


