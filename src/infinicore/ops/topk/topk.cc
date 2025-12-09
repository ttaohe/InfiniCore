#include "infinicore/ops/topk.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<TopK::schema> &TopK::dispatcher() {
    static common::OpDispatcher<TopK::schema> dispatcher_;
    return dispatcher_;
}

void TopK::execute(Tensor values,
                   Tensor indices,
                   Tensor input,
                   long long k,
                   long long dim,
                   bool largest,
                   bool sorted) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(values, indices, input);

    auto device = input->device();
    infinicore::context::setDevice(device);

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("topk does not support scalar tensors.");
    }

    if (k <= 0) {
        throw std::runtime_error("topk: k must be positive.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("topk: dim out of range.");
    }

    const auto &in_shape = input->shape();
    Size dim_size = in_shape[static_cast<Size>(d)];
    if (k > static_cast<long long>(dim_size)) {
        throw std::runtime_error("topk: k cannot be larger than the size of the selected dimension.");
    }

    // 检查输出形状：除 dim 维度外与输入一致，dim 上长度为 k
    auto check_out_shape = [&](const Tensor &t, const char *name) {
        const auto &shape = t->shape();
        if (shape.size() != in_shape.size()) {
            throw std::runtime_error(std::string("topk: ") + name + " rank mismatch.");
        }
        for (Size i = 0; i < ndim; ++i) {
            if (i == static_cast<Size>(d)) {
                if (shape[i] != static_cast<Size>(k)) {
                    throw std::runtime_error(std::string("topk: ") + name + " dim size mismatch on topk dim.");
                }
            } else {
                if (shape[i] != in_shape[i]) {
                    throw std::runtime_error(std::string("topk: ") + name + " shape mismatch.");
                }
            }
        }
    };

    check_out_shape(values, "values");
    check_out_shape(indices, "indices");

    // dtype 检查
    auto in_dtype = input->dtype();
    if (!(in_dtype == DataType::F16 || in_dtype == DataType::BF16 || in_dtype == DataType::F32)) {
        throw std::runtime_error("topk: only F16/BF16/F32 input dtypes are supported.");
    }
    if (values->dtype() != in_dtype) {
        throw std::runtime_error("topk: values dtype must equal input dtype.");
    }
    if (indices->dtype() != DataType::I64) {
        throw std::runtime_error("topk: indices dtype must be I64.");
    }

    auto fn = dispatcher().lookup(device.getType());
    if (fn == nullptr) {
        throw std::runtime_error("No TopK implementation found for device type.");
    }

    // 为简化 kernel 实现，输入统一转为连续内存
    Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

    fn(values, indices, input_contiguous, k, d, largest, sorted);
}

std::pair<Tensor, Tensor> topk(Tensor input,
                               long long k,
                               long long dim,
                               bool largest,
                               bool sorted) {
    auto device = input->device();
    auto dtype = input->dtype();

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("topk does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("topk: dim out of range.");
    }

    const auto &in_shape = input->shape();
    auto out_shape = in_shape;
    out_shape[static_cast<Size>(d)] = static_cast<Size>(k);

    auto values = Tensor::empty(out_shape, dtype, device);
    auto indices = Tensor::empty(out_shape, DataType::I64, device);

    topk_(values, indices, input, k, d, largest, sorted);
    return {values, indices};
}

void topk_(Tensor values,
           Tensor indices,
           Tensor input,
           long long k,
           long long dim,
           bool largest,
           bool sorted) {
    TopK::execute(values, indices, input, k, dim, largest, sorted);
}

} // namespace infinicore::op


