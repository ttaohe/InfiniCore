#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/**
 * @brief OneHot 算子：将整数索引张量转换为 one-hot 表示。
 *
 * 约定：
 *  - 输入 indices dtype 为 I64；
 *  - 输出 dtype 固定为 I64；
 *  - 输出 shape 为 `indices.shape + [num_classes]`；
 *  - 目前仅实现 CPU 版本，后续可扩展到其他设备。
 */
class OneHot {
public:
    using schema = void (*)(Tensor out, Tensor indices, long long num_classes);

    static void execute(Tensor out, Tensor indices, long long num_classes);

    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief Out-of-place OneHot 接口。
 *
 * @param indices     输入索引张量（I64）
 * @param num_classes 类别数；若 <= 0，则根据 indices 中最大值自动推断
 *
 * @return 输出 one-hot 张量，dtype=I64，shape = indices.shape + [num_classes]
 */
Tensor one_hot(Tensor indices, long long num_classes);

/**
 * @brief In-place OneHot 接口（将结果写入给定输出）。
 *
 * @param out         输出张量（预先分配好正确 shape，dtype=I64）
 * @param indices     输入索引张量（I64）
 * @param num_classes 类别数
 */
void one_hot_(Tensor out, Tensor indices, long long num_classes);

} // namespace infinicore::op


