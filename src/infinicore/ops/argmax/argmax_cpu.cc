#include "infinicore/ops/argmax.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op::argmax_impl::cpu {

template <typename T>
void argmax_cpu_impl(Tensor out, Tensor input, long long dim) {
    const auto &shape = input->shape();
    Size ndim = input->ndim();

    Size reduce_dim = shape[static_cast<Size>(dim)];

    Size outer = 1;
    for (Size i = 0; i < static_cast<Size>(dim); ++i) {
        outer *= shape[i];
    }

    Size inner = 1;
    for (Size i = static_cast<Size>(dim) + 1; i < ndim; ++i) {
        inner *= shape[i];
    }

    const T *in_ptr = reinterpret_cast<const T *>(input->data());
    auto *out_ptr = reinterpret_cast<long long *>(out->data());

    // out 元素个数 = outer * inner
    for (Size o = 0; o < outer; ++o) {
        for (Size in_idx = 0; in_idx < inner; ++in_idx) {
            Size base = (o * reduce_dim) * inner + in_idx;

            Size best_idx = 0;
            T best_val = in_ptr[base];

            for (Size r = 1; r < reduce_dim; ++r) {
                Size idx = (o * reduce_dim + r) * inner + in_idx;
                T v = in_ptr[idx];
                if (v > best_val) {
                    best_val = v;
                    best_idx = r;
                }
            }

            *out_ptr++ = static_cast<long long>(best_idx);
        }
    }
}

void calculate(Tensor out, Tensor input, long long dim) {
    if (out->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("ArgMax CPU kernel only supports CPU device.");
    }

    auto in_dtype = input->dtype();
    if (in_dtype == DataType::F32) {
        argmax_cpu_impl<float>(out, input, dim);
    } else if (in_dtype == DataType::I32) {
        argmax_cpu_impl<int32_t>(out, input, dim);
    } else {
        throw std::runtime_error("ArgMax CPU kernel only supports F32 and I32 input dtypes.");
    }
}

static bool registered = []() {
    ArgMax::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::argmax_impl::cpu


