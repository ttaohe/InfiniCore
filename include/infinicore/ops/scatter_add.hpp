#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/**
 * @brief 通用 ScatterAdd 算子：语义对齐 `torch.scatter_add`/`Tensor.scatter_add_` 的常见用法。
 *
 * 约定（当前实现）：
 *  - 仅支持 CPU 设备；
 *  - 支持 float16 / bfloat16 / float32 三种浮点输入 dtype（与单测保持一致）；
 *  - index dtype 固定为 I64；
 *  - out/input/src 必须 shape 一致；
 *  - 内核实现为“在 out 上 in-place 加上 src 的散射值”，其中 base 值来自 input。
 */
class ScatterAdd {
public:
    using schema = void (*)(Tensor out, Tensor input, Tensor index, Tensor src, long long dim);

    static void execute(Tensor out, Tensor input, Tensor index, Tensor src, long long dim);

    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief Out-of-place 接口：返回一个新的张量，语义类似 `torch.scatter_add(input, dim, index, src)`。
 */
Tensor scatter_add(Tensor input, long long dim, Tensor index, Tensor src);

/**
 * @brief In-place 接口：将结果写到 out 中。若 out 与 input 为同一张量，则语义类似 `Tensor.scatter_add_`。
 */
void scatter_add_(Tensor out, Tensor input, long long dim, Tensor index, Tensor src);

} // namespace infinicore::op


