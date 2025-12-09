#include "infinicore/ops/scatter_add.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<ScatterAdd::schema> &ScatterAdd::dispatcher() {
    static common::OpDispatcher<ScatterAdd::schema> dispatcher_;
    return dispatcher_;
}

void ScatterAdd::execute(Tensor out, Tensor input, Tensor index, Tensor src, long long dim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, index, src);

    auto device = out->device();
    infinicore::context::setDevice(device);

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("scatter_add does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("scatter_add: dim out of range.");
    }

    // 形状检查：out/input/index/src 必须一致
    if (out->shape() != input->shape() || out->shape() != index->shape() || out->shape() != src->shape()) {
        throw std::runtime_error("scatter_add: out, input, index and src must have the same shape.");
    }

    // dtype 检查
    auto in_dtype = input->dtype();
    if (!(in_dtype == DataType::F16 || in_dtype == DataType::BF16 || in_dtype == DataType::F32)) {
        throw std::runtime_error("scatter_add: only F16/BF16/F32 input dtypes are supported currently.");
    }
    if (src->dtype() != in_dtype || out->dtype() != in_dtype) {
        throw std::runtime_error("scatter_add: src and out must have the same dtype as input.");
    }
    if (index->dtype() != DataType::I64) {
        throw std::runtime_error("scatter_add: index tensor must have dtype=I64.");
    }

    auto device_type = device.getType();
    auto fn = dispatcher().lookup(device_type);
    if (fn == nullptr) {
        throw std::runtime_error("No ScatterAdd implementation found for device type.");
    }

    fn(out, input, index, src, d);
}

Tensor scatter_add(Tensor input, long long dim, Tensor index, Tensor src) {
    auto device = input->device();
    auto dtype = input->dtype();

    auto out = Tensor::empty(input->shape(), dtype, device);
    scatter_add_(out, input, dim, index, src);
    return out;
}

void scatter_add_(Tensor out, Tensor input, long long dim, Tensor index, Tensor src) {
    ScatterAdd::execute(out, input, index, src, dim);
}

} // namespace infinicore::op


