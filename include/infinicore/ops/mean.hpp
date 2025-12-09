#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/**
 * @brief 通用 Mean 算子：沿给定维度或整体求平均。
 *
 * 约定：
 *  - 支持 float16 / bfloat16 / float32 输入和输出；
 *  - `dim` 为单个整型维度索引（Python 层负责处理 dim=None 或 tuple 的情况）；
 *  - `keepdim` 控制是否保留被约简的维度长度为 1；
 *  - 当前仅实现 CPU 版本。
 */
class Mean {
public:
    using schema = void (*)(Tensor out, Tensor input, long long dim, bool keepdim);

    static void execute(Tensor out, Tensor input, long long dim, bool keepdim);

    static common::OpDispatcher<schema> &dispatcher();
};

Tensor mean(Tensor input, long long dim, bool keepdim);
void mean_(Tensor out, Tensor input, long long dim, bool keepdim);

} // namespace infinicore::op


