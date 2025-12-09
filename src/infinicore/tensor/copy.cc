#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
namespace infinicore {
Tensor TensorImpl::to(Device device) const {
    if (device == data_.memory->device()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        std::shared_ptr<TensorImpl> _t = empty(meta_.shape, meta_.dtype, device);
        _t->copy_from(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
        return Tensor(_t);
    }
}

void TensorImpl::copy_from(Tensor src) {
    if (src->shape() != this->shape()) {
        throw std::runtime_error("Cannot copy from tensor with different shape");
    }
    if (this->device() == src->device()) {
        // 同一设备上的拷贝（例如 CPU->CPU）：默认走 rearrange_。
        // 但为了在 CPU 上精确支持任意 strides（尤其是 F16/BF16 等情况），
        // 这里对 CPU 设备做一个更直接、确定的多维索引拷贝实现，
        // 避免潜在的后端实现差异造成数值/索引误差。
        if (this->device().getType() == Device::Type::CPU) {
            const auto &shape = this->shape();
            const auto &dst_strides = this->strides();
            const auto &src_strides = src->strides();

            Size ndim = shape.size();
            Size elem_size = dsize(this->dtype());
            Size total = this->numel();

            const std::byte *src_data = src->data();
            std::byte *dst_data = this->data();

            std::vector<Size> idx(ndim, 0);

            for (Size linear = 0; linear < total; ++linear) {
                Size src_offset = 0;
                Size dst_offset = 0;
                for (Size k = 0; k < ndim; ++k) {
                    src_offset += idx[k] * src_strides[k];
                    dst_offset += idx[k] * dst_strides[k];
                }

                std::memcpy(
                    dst_data + dst_offset * elem_size,
                    src_data + src_offset * elem_size,
                    static_cast<size_t>(elem_size));

                // 递增多维索引
                for (long long dim_i = static_cast<long long>(ndim) - 1; dim_i >= 0; --dim_i) {
                    Size dim_index = static_cast<Size>(dim_i);
                    idx[dim_index]++;
                    if (idx[dim_index] < shape[dim_index]) {
                        break;
                    }
                    idx[dim_index] = 0;
                }
            }
        } else {
            op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
        }
    } else {
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }

        // Use nbytes() to get the actual tensor size, not the full memory size
        size_t copy_size = std::min(this->nbytes(), src->nbytes());
        if (this->device().getType() == Device::Type::CPU) {
            if (this->is_contiguous()) {
                context::memcpyD2H(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyD2H(local_src->data(), src->data(), this->data_.memory->size());
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else if (src->device().getType() == Device::Type::CPU) {

            if (this->is_contiguous()) {
                context::memcpyH2D(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyH2D(local_src->data(), src->data(), copy_size);
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        }
    }
}

Tensor TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        return op::rearrange(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
    }
}

} // namespace infinicore
