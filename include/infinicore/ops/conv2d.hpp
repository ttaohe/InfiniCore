#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

/**
 * @brief 2D 卷积算子，形状约定：
 *  - input:  [N, C_in, H_in, W_in]
 *  - weight: [C_out, C_in / groups, K_h, K_w]
 *  - bias:   [C_out] 或 None
 *
 * groups 目前仅支持 1（标准卷积），否则在 C++ 层抛异常。
 */
class Conv2d {
public:
    using schema = void (*)(Tensor out,
                            Tensor input,
                            Tensor weight,
                            std::optional<Tensor> bias,
                            long long stride_h,
                            long long stride_w,
                            long long pad_h,
                            long long pad_w,
                            long long dilation_h,
                            long long dilation_w,
                            long long groups);

    static void execute(Tensor out,
                        Tensor input,
                        Tensor weight,
                        std::optional<Tensor> bias,
                        long long stride_h,
                        long long stride_w,
                        long long pad_h,
                        long long pad_w,
                        long long dilation_h,
                        long long dilation_w,
                        long long groups);

    static common::OpDispatcher<schema> &dispatcher();
};

Tensor conv2d(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              long long stride_h,
              long long stride_w,
              long long pad_h,
              long long pad_w,
              long long dilation_h,
              long long dilation_w,
              long long groups);

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
             long long groups);

} // namespace infinicore::op


