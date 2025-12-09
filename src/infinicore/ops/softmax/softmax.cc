#include "infinicore/ops/softmax.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Softmax::schema> &Softmax::dispatcher() {
    static common::OpDispatcher<Softmax::schema> dispatcher_;
    return dispatcher_;
}

void Softmax::execute(Tensor out, Tensor input, long long dim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);

    auto device = out->device();
    infinicore::context::setDevice(device);

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("softmax does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("softmax: dim out of range.");
    }

    if (out->shape() != input->shape()) {
        throw std::runtime_error("softmax: out and input must have the same shape.");
    }

    auto dtype = input->dtype();
    if (!(dtype == DataType::F16 || dtype == DataType::BF16 || dtype == DataType::F32)) {
        throw std::runtime_error("softmax: only F16/BF16/F32 dtypes are supported.");
    }
    if (out->dtype() != dtype) {
        throw std::runtime_error("softmax: out dtype must equal input dtype.");
    }

    auto fn = dispatcher().lookup(device.getType());
    if (fn == nullptr) {
        throw std::runtime_error("No Softmax implementation found for device type.");
    }

    // InfiniOP softmax 支持 axis，因此可以直接在任意 dim 上计算
    fn(out, input, d);
}

Tensor softmax(Tensor input, long long dim) {
    auto device = input->device();
    auto dtype = input->dtype();

    auto out = Tensor::empty(input->shape(), dtype, device);
    softmax_(out, input, dim);
    return out;
}

void softmax_(Tensor out, Tensor input, long long dim) {
    Softmax::execute(out, input, dim);
}


common::OpDispatcher<LogSoftmax::schema> &LogSoftmax::dispatcher() {
    static common::OpDispatcher<LogSoftmax::schema> dispatcher_;
    return dispatcher_;
}

void LogSoftmax::execute(Tensor out, Tensor input, long long dim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);

    auto device = out->device();
    infinicore::context::setDevice(device);

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("log_softmax does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("log_softmax: dim out of range.");
    }

    if (out->shape() != input->shape()) {
        throw std::runtime_error("log_softmax: out and input must have the same shape.");
    }

    auto dtype = input->dtype();
    if (!(dtype == DataType::F16 || dtype == DataType::BF16 || dtype == DataType::F32)) {
        throw std::runtime_error("log_softmax: only F16/BF16/F32 dtypes are supported.");
    }
    if (out->dtype() != dtype) {
        throw std::runtime_error("log_softmax: out dtype must equal input dtype.");
    }

    auto fn = dispatcher().lookup(device.getType());
    if (fn == nullptr) {
        throw std::runtime_error("No LogSoftmax implementation found for device type.");
    }

    // InfiniOP LogSoftmax 总是在最后一维上做计算，因此对于 dim != last 的情况，
    // 需要通过 permute 将目标维度移到最后一维，再在计算后移回原位置。
    if (static_cast<long long>(ndim - 1) == d) {
        // 目标维度本身就是最后一维，直接调用
        fn(out, input, d);
        return;
    }

    // 构造一个排列，将 dim 与最后一维交换
    Shape order(ndim);
    for (Size i = 0; i < ndim; ++i) {
        order[i] = i;
    }
    std::swap(order[static_cast<Size>(d)], order[ndim - 1]);

    // 创建视图（不拷贝数据）
    Tensor input_perm = input->permute(order);
    Tensor out_perm = out->permute(order);

    // 在最后一维上做 log_softmax（此时 dim == ndim - 1）
    fn(out_perm, input_perm, static_cast<long long>(ndim - 1));
}

Tensor log_softmax(Tensor input, long long dim) {
    auto device = input->device();
    auto dtype = input->dtype();

    auto out = Tensor::empty(input->shape(), dtype, device);
    log_softmax_(out, input, dim);
    return out;
}

void log_softmax_(Tensor out, Tensor input, long long dim) {
    LogSoftmax::execute(out, input, dim);
}

} // namespace infinicore::op


