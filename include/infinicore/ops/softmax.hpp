#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/**
 * @brief 通用 Softmax 算子：沿指定维度做 softmax。
 *
 * 约定：
 *  - 支持 float16 / bfloat16 / float32 输入和输出；
 *  - 支持任意维度张量和任意有效 dim（支持负数）；
 *  - 内部通过 InfiniOP softmax 实现，axis 与 dim 一一对应。
 */
class Softmax {
public:
    using schema = void (*)(Tensor out, Tensor input, long long dim);

    static void execute(Tensor out, Tensor input, long long dim);

    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief Out-of-place softmax：语义类似 `torch.nn.functional.softmax(input, dim=dim)`.
 */
Tensor softmax(Tensor input, long long dim);

/**
 * @brief In-place 版本，将结果写入给定输出张量。
 */
void softmax_(Tensor out, Tensor input, long long dim);


/**
 * @brief 通用 LogSoftmax 算子：沿指定维度做 log_softmax。
 */
class LogSoftmax {
public:
    using schema = void (*)(Tensor out, Tensor input, long long dim);

    static void execute(Tensor out, Tensor input, long long dim);

    static common::OpDispatcher<schema> &dispatcher();
};

Tensor log_softmax(Tensor input, long long dim);
void log_softmax_(Tensor out, Tensor input, long long dim);

} // namespace infinicore::op


