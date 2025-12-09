#pragma once

#include "../device.hpp"
#include "common/op.hpp"

#include <utility>

namespace infinicore::op {

/**
 * @brief TopK 算子：语义对齐 `torch.topk` 的常见用法。
 *
 * 约定：
 *  - 当前仅支持 CPU 设备；
 *  - 支持 float16 / bfloat16 / float32 三种浮点输入 dtype；
 *  - indices dtype 固定为 I64；
 *  - dim 支持负数，下标从最后一维开始；
 *  - largest/sorted 仅影响排序方向和是否返回有序结果（见实现说明）。
 */
class TopK {
public:
    using schema = void (*)(Tensor values,
                            Tensor indices,
                            Tensor input,
                            long long k,
                            long long dim,
                            bool largest,
                            bool sorted);

    static void execute(Tensor values,
                        Tensor indices,
                        Tensor input,
                        long long k,
                        long long dim,
                        bool largest,
                        bool sorted);

    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief Out-of-place TopK 接口：返回 (values, indices)。
 */
std::pair<Tensor, Tensor> topk(Tensor input,
                               long long k,
                               long long dim,
                               bool largest,
                               bool sorted);

/**
 * @brief In-place 接口：将结果写入给定输出张量。
 */
void topk_(Tensor values,
           Tensor indices,
           Tensor input,
           long long k,
           long long dim,
           bool largest,
           bool sorted);

} // namespace infinicore::op


