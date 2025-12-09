#include "infinicore/ops/conv2d.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Conv2d::schema> &Conv2d::dispatcher() {
    static common::OpDispatcher<Conv2d::schema> dispatcher_;
    return dispatcher_;
}

static inline void check_conv2d_shapes(Tensor out,
                                       Tensor input,
                                       Tensor weight,
                                       const std::optional<Tensor> &bias,
                                       long long stride_h,
                                       long long stride_w,
                                       long long pad_h,
                                       long long pad_w,
                                       long long dilation_h,
                                       long long dilation_w,
                                       long long groups) {
    auto in_ndim = input->ndim();
    auto w_ndim = weight->ndim();

    if (in_ndim != 4 || w_ndim != 4) {
        throw std::runtime_error("conv2d: only 4D NCHW tensors are supported.");
    }

    const auto &in_shape = input->shape();
    const auto &w_shape = weight->shape();

    Size N = in_shape[0];
    Size C_in = in_shape[1];
    Size H_in = in_shape[2];
    Size W_in = in_shape[3];

    Size C_out = w_shape[0];
    Size C_per_group = w_shape[1];
    Size K_h = w_shape[2];
    Size K_w = w_shape[3];

    if (groups <= 0) {
        throw std::runtime_error("conv2d: groups must be positive.");
    }

    if (groups != 1) {
        throw std::runtime_error("conv2d: only groups=1 is supported currently.");
    }

    if (C_in != static_cast<Size>(groups) * C_per_group) {
        throw std::runtime_error("conv2d: input channels must equal groups * weight[1].");
    }

    auto compute_out_dim = [](Size in, long long pad, Size k, long long stride, long long dilation) -> Size {
        long long num = static_cast<long long>(in) + 2 * pad - dilation * (static_cast<long long>(k) - 1) - 1;
        if (num < 0) {
            throw std::runtime_error("conv2d: computed output size is negative.");
        }
        long long out = num / stride + 1;
        if (out <= 0) {
            throw std::runtime_error("conv2d: computed output size is non-positive.");
        }
        return static_cast<Size>(out);
    };

    Size H_out = compute_out_dim(H_in, pad_h, K_h, stride_h, dilation_h);
    Size W_out = compute_out_dim(W_in, pad_w, K_w, stride_w, dilation_w);

    // Check out tensor shape
    const auto &out_shape = out->shape();
    if (out_shape.size() != 4 ||
        out_shape[0] != N ||
        out_shape[1] != C_out ||
        out_shape[2] != H_out ||
        out_shape[3] != W_out) {
        throw std::runtime_error("conv2d: output shape mismatch.");
    }

    // Check bias
    if (bias.has_value()) {
        if (bias.value()->ndim() != 1 || bias.value()->shape()[0] != C_out) {
            throw std::runtime_error("conv2d: bias must be 1D of shape [C_out].");
        }
    }
}

void Conv2d::execute(Tensor out,
                     Tensor input,
                     Tensor weight,
                     std::optional<Tensor> bias,
                     long long stride_h,
                     long long stride_w,
                     long long pad_h,
                     long long pad_w,
                     long long dilation_h,
                     long long dilation_w,
                     long long groups) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight);
    if (bias.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, bias.value());
    }

    auto device = out->device();
    infinicore::context::setDevice(device);

    auto dtype = input->dtype();
    if (!(dtype == DataType::F16 || dtype == DataType::BF16 || dtype == DataType::F32)) {
        throw std::runtime_error("conv2d: only F16/BF16/F32 dtypes are supported.");
    }
    if (out->dtype() != dtype || weight->dtype() != dtype ||
        (bias.has_value() && bias.value()->dtype() != dtype)) {
        throw std::runtime_error("conv2d: all tensors must have the same dtype.");
    }

    check_conv2d_shapes(out,
                        input,
                        weight,
                        bias,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                        dilation_h,
                        dilation_w,
                        groups);

    // 为了简化 kernel 逻辑，这里统一在 C++ 层处理非连续张量：
    Tensor input_contig = input->is_contiguous() ? input : input->contiguous();
    Tensor weight_contig = weight->is_contiguous() ? weight : weight->contiguous();
    std::optional<Tensor> bias_contig;
    if (bias.has_value()) {
        bias_contig = bias.value()->is_contiguous() ? bias.value() : bias.value()->contiguous();
    }

    Tensor out_work = out;
    bool need_copy_back = false;
    if (!out->is_contiguous()) {
        out_work = Tensor::empty(out->shape(), dtype, device);
        need_copy_back = true;
    }

    auto fn = dispatcher().lookup(device.getType());
    if (fn == nullptr) {
        throw std::runtime_error("No Conv2d implementation found for device type.");
    }

    fn(out_work,
       input_contig,
       weight_contig,
       bias_contig,
       stride_h,
       stride_w,
       pad_h,
       pad_w,
       dilation_h,
       dilation_w,
       groups);

    if (need_copy_back) {
        out->copy_from(out_work);
    }
}

Tensor conv2d(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              long long stride_h,
              long long stride_w,
              long long pad_h,
              long long pad_w,
              long long dilation_h,
              long long dilation_w,
              long long groups) {
    const auto &in_shape = input->shape();
    const auto &w_shape = weight->shape();

    if (in_shape.size() != 4 || w_shape.size() != 4) {
        throw std::runtime_error("conv2d: only 4D NCHW tensors are supported.");
    }

    Size N = in_shape[0];
    Size C_in = in_shape[1];
    Size H_in = in_shape[2];
    Size W_in = in_shape[3];

    Size C_out = w_shape[0];
    Size C_per_group = w_shape[1];
    Size K_h = w_shape[2];
    Size K_w = w_shape[3];

    if (groups <= 0) {
        throw std::runtime_error("conv2d: groups must be positive.");
    }
    if (groups != 1) {
        throw std::runtime_error("conv2d: only groups=1 is supported currently.");
    }
    if (C_in != static_cast<Size>(groups) * C_per_group) {
        throw std::runtime_error("conv2d: input channels must equal groups * weight[1].");
    }

    auto compute_out_dim = [](Size in, long long pad, Size k, long long stride, long long dilation) -> Size {
        long long num = static_cast<long long>(in) + 2 * pad - dilation * (static_cast<long long>(k) - 1) - 1;
        if (num < 0) {
            throw std::runtime_error("conv2d: computed output size is negative.");
        }
        long long out = num / stride + 1;
        if (out <= 0) {
            throw std::runtime_error("conv2d: computed output size is non-positive.");
        }
        return static_cast<Size>(out);
    };

    Size H_out = compute_out_dim(H_in, pad_h, K_h, stride_h, dilation_h);
    Size W_out = compute_out_dim(W_in, pad_w, K_w, stride_w, dilation_w);

    Shape out_shape{N, C_out, H_out, W_out};

    auto out = Tensor::empty(out_shape, input->dtype(), input->device());
    conv2d_(out,
            input,
            weight,
            bias,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups);
    return out;
}

void conv2d_(Tensor out,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias,
             long long stride_h,
             long long stride_w,
             long long pad_h,
             long long pad_w,
             long long dilation_h,
             long long dilation_w,
             long long groups) {
    Conv2d::execute(out,
                    input,
                    weight,
                    bias,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dilation_h,
                    dilation_w,
                    groups);
}

} // namespace infinicore::op


