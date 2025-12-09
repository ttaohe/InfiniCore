#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/**
 * @brief 通用 ArgMax 算子：沿给定维度求最大值的下标。
 *
 * 约定：
 *  - 目前仅支持输入 dtype 为 F32 或 I32；
 *  - 输出 dtype 固定为 I64（与 PyTorch `argmax` 一致）；
 *  - 内核要求输入/输出张量为连续内存，必要时在高层先调用 `contiguous()`。
 */
class ArgMax {
public:
    using schema = void (*)(Tensor out, Tensor input, long long dim);

    static void execute(Tensor out, Tensor input, long long dim);

    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief ArgMax 的 out-of-place 接口。
 *
 * @param input 输入张量
 * @param dim   归约维度（支持负数，下标从最后一维开始）
 *
 * @return 沿 dim 归约后的索引张量，dtype=I64
 */
Tensor argmax(Tensor input, long long dim);

/**
 * @brief ArgMax 的 in-place 接口（将结果写入给定输出）。
 *
 * @param out   输出张量（预先分配好正确 shape、dtype=I64）
 * @param input 输入张量
 * @param dim   归约维度
 */
void argmax_(Tensor out, Tensor input, long long dim);

} // namespace infinicore::op


