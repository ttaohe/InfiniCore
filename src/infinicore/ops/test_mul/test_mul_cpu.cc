#include "infinicore/ops/test_mul.hpp"

#include "../../utils.hpp"

namespace infinicore::op::test_mul_impl::cpu {

/**
 * @brief CPU 上的简单向量点积实现
 *
 * 约定与限制（示例代码，为了简化逻辑）：
 *  - 仅支持 Device::Type::CPU
 *  - 仅支持 DataType::F32
 *  - 要求 a、b 为一维张量，且：
 *      - 形状相同
 *      - 均为 contiguous
 *  - out 为 0 维标量张量（shape == {}），且 dtype == F32、device == CPU
 */
void calculate(Tensor out, Tensor a, Tensor b) {
    // 1. 设备检查：本 kernel 只支持 CPU
    if (out->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("test_mul CPU kernel only supports CPU tensors.");
    }

    // 2. dtype 检查：示例仅实现 F32
    if (out->dtype() != DataType::F32 || a->dtype() != DataType::F32 || b->dtype() != DataType::F32) {
        throw std::runtime_error("test_mul CPU kernel only supports F32 tensors.");
    }

    // 3. 形状 & 连续性检查（只支持一维向量点积）
    if (a->ndim() != 1 || b->ndim() != 1) {
        throw std::runtime_error("test_mul CPU kernel expects 1D vectors for a and b.");
    }

    if (a->shape() != b->shape()) {
        throw std::runtime_error("test_mul CPU kernel expects a and b to have the same shape.");
    }

    if (!a->is_contiguous() || !b->is_contiguous()) {
        throw std::runtime_error("test_mul CPU kernel only supports contiguous input tensors.");
    }

    if (out->numel() != 1) {
        throw std::runtime_error("test_mul CPU kernel expects output tensor to be a scalar.");
    }

    // 4. 真正的点积计算
    Size n = a->numel();

    const float *a_ptr = reinterpret_cast<const float *>(a->data());
    const float *b_ptr = reinterpret_cast<const float *>(b->data());
    float *out_ptr = reinterpret_cast<float *>(out->data());

    float acc = 0.0f;
    for (Size i = 0; i < n; ++i) {
        acc += a_ptr[i] * b_ptr[i];
    }

    *out_ptr = acc;
}

// 5. 在加载动态库时为 CPU 设备注册 kernel
static bool registered = []() {
    TestMul::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::test_mul_impl::cpu


