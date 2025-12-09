#include "infinicore/ops/test_mul.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<TestMul::schema> &TestMul::dispatcher() {
    static common::OpDispatcher<TestMul::schema> dispatcher_;
    return dispatcher_;
};

void TestMul::execute(Tensor out, Tensor a, Tensor b) {
    // 简单的设备一致性检查
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b);

    // 将运行时设备设置为与输出张量一致
    infinicore::context::setDevice(out->device());

    // 根据设备类型查找并调用对应 kernel
    dispatcher().lookup(out->device().getType())(out, a, b);
}

Tensor test_mul(Tensor a, Tensor b) {
    // 要求 a/b 处于同一设备
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b);

    auto device = a->device();
    auto dtype = a->dtype();

    // 创建一个 0 维标量 Tensor 作为输出
    Shape scalar_shape{}; // 空 shape 表示标量，见 nn::Parameter 等用法
    auto out = Tensor::empty(scalar_shape, dtype, device);

    test_mul_(out, a, b);
    return out;
}

void test_mul_(Tensor out, Tensor a, Tensor b) {
    TestMul::execute(out, a, b);
}

} // namespace infinicore::op


