#include "infinicore/ops/mean.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Mean::schema> &Mean::dispatcher() {
    static common::OpDispatcher<Mean::schema> dispatcher_;
    return dispatcher_;
}

void Mean::execute(Tensor out, Tensor input, long long dim, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input);

    auto device = input->device();
    infinicore::context::setDevice(device);

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("mean does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("mean: dim out of range.");
    }

    auto in_dtype = input->dtype();
    if (!(in_dtype == DataType::F16 || in_dtype == DataType::BF16 || in_dtype == DataType::F32)) {
        throw std::runtime_error("mean: only F16/BF16/F32 input dtypes are supported.");
    }
    if (out->dtype() != in_dtype) {
        throw std::runtime_error("mean: out dtype must equal input dtype.");
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
        throw std::runtime_error("mean: output shape mismatch.");
    }

    Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

    Tensor work_out = out;
    bool need_copy_back = false;
    if (!out->is_contiguous()) {
        work_out = Tensor::empty(expected_shape, in_dtype, device);
        need_copy_back = true;
    }

    auto fn = dispatcher().lookup(device.getType());
    if (fn == nullptr) {
        throw std::runtime_error("No Mean implementation found for device type.");
    }

    fn(work_out, input_contiguous, d, keepdim);

    if (need_copy_back) {
        // 对 CPU，手动实现一个简单、确定的 contiguous->strided 拷贝，
        // 避免依赖更复杂的 rearrange_ 逻辑导致额外的数值/索引误差。
        if (device.getType() == Device::Type::CPU) {
            const auto &shape = expected_shape;
            const auto &out_strides = out->strides();
            Size ndim_copy = shape.size();

            const std::byte *src = work_out->data();
            std::byte *dst = out->data();

            Size elem_size = dsize(in_dtype);
            Size total = work_out->numel();

            // 多维索引 [i0, i1, ..., in-1]
            std::vector<Size> idx(ndim_copy, 0);

            for (Size linear = 0; linear < total; ++linear) {
                // 计算目标张量中的元素偏移（单位：元素）
                Size dst_offset = 0;
                for (Size k = 0; k < ndim_copy; ++k) {
                    dst_offset += idx[k] * out_strides[k];
                }

                std::memcpy(
                    dst + dst_offset * elem_size,
                    src + linear * elem_size,
                    static_cast<size_t>(elem_size)
                );

                // 递增多维索引
                for (long long dim_i = static_cast<long long>(ndim_copy) - 1; dim_i >= 0; --dim_i) {
                    Size dim_index = static_cast<Size>(dim_i);
                    idx[dim_index]++;
                    if (idx[dim_index] < shape[dim_index]) {
                        break;
                    }
                    idx[dim_index] = 0;
                }
            }
        } else {
            out->copy_from(work_out);
        }
    }
}

Tensor mean(Tensor input, long long dim, bool keepdim) {
    auto device = input->device();
    auto dtype = input->dtype();

    Size ndim = input->ndim();
    if (ndim == 0) {
        throw std::runtime_error("mean does not support scalar tensors.");
    }

    long long d = dim;
    if (d < 0) {
        d += static_cast<long long>(ndim);
    }
    if (d < 0 || d >= static_cast<long long>(ndim)) {
        throw std::runtime_error("mean: dim out of range.");
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
    mean_(out, input, dim, keepdim);
    return out;
}

void mean_(Tensor out, Tensor input, long long dim, bool keepdim) {
    Mean::execute(out, input, dim, keepdim);
}

} // namespace infinicore::op


