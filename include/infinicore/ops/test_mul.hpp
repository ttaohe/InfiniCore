#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

/**
 * @brief 一个示例算子：向量点积 test_mul
 *
 * 语义：给定两个同长度的一维向量 a, b，计算
 *      out = sum_i a[i] * b[i]
 *
 * 为了简化示例，本算子仅在 CPU + F32 标量输出场景下实现真实 kernel，
 * 其他情况会抛出异常。
 */
class TestMul {
public:
    /// Kernel 函数签名：out, a, b
    using schema = void (*)(Tensor, Tensor, Tensor);

    /// 统一执行入口，内部通过 dispatcher 分发到具体设备 kernel
    static void execute(Tensor out, Tensor a, Tensor b);

    /// 获取全局分发器（每种算子一个全局 dispatcher 实例）
    static common::OpDispatcher<schema> &dispatcher();
};

/**
 * @brief test_mul 的 out-of-place 接口
 *
 * 输入：
 *  - a, b: 同设备、同 dtype、同 numel 的一维张量
 *
 * 返回：
 *  - 一个 0 维标量 Tensor，dtype 与 a 相同，值为 dot(a, b)
 */
Tensor test_mul(Tensor a, Tensor b);

/**
 * @brief test_mul 的 in-place 接口
 *
 * 输入：
 *  - out: 标量 Tensor，dtype/设备与 a/b 一致，用于写入结果
 *  - a, b: 同设备、同 dtype、同 numel 的一维张量
 */
void test_mul_(Tensor out, Tensor a, Tensor b);

} // namespace infinicore::op


