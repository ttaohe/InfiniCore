#include "infinicore/ops/scatter_add.hpp"

#include "../../utils.hpp"

#include <cstdint>
#include <stdexcept>

namespace infinicore::op::scatter_add_impl::cpu {

// Helper: convert IEEE 754 half-precision (float16) bits to float32
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    uint32_t f_bits;
    if (exp == 0) {
        if (mant == 0) {
            // Zero
            f_bits = sign << 31;
        } else {
            // Subnormal -> normalized
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            exp = exp - 1 + 127 - 15; // adjust bias
            f_bits = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        // Inf or NaN
        f_bits = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        // Normalized
        exp = exp - 15 + 127;
        f_bits = (sign << 31) | (exp << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &f_bits, sizeof(float));
    return f;
}

// Helper: convert float32 to IEEE 754 half-precision (float16) bits
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(uint32_t));

    uint32_t sign = (x >> 31) & 0x1;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;

    uint16_t h;
    if (exp <= 0) {
        if (exp < -10) {
            // Too small -> zero
            h = static_cast<uint16_t>(sign << 15);
        } else {
            // Subnormal
            mant |= 0x800000;
            uint32_t shift = static_cast<uint32_t>(1 - exp);
            mant = mant >> (shift + 13);
            h = static_cast<uint16_t>((sign << 15) | mant);
        }
    } else if (exp >= 31) {
        // Overflow -> Inf
        h = static_cast<uint16_t>((sign << 15) | 0x7c00);
    } else {
        // Normalized
        h = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }
    return h;
}

// Helper: BF16 <-> F32 conversions
static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(uint32_t));
    return static_cast<uint16_t>(u >> 16);
}

template <typename T>
void scatter_add_cpu_impl(Tensor out, Tensor input, Tensor index, Tensor src, long long dim) {
    const auto &shape = input->shape();
    const auto &in_strides = input->strides();
    const auto &out_strides = out->strides();
    const auto &idx_strides = index->strides();
    const auto &src_strides = src->strides();

    Size ndim = input->ndim();

    const long long *idx_ptr = reinterpret_cast<const long long *>(index->data());
    const T *src_ptr = reinterpret_cast<const T *>(src->data());
    const T *in_ptr = reinterpret_cast<const T *>(input->data());
    T *out_ptr = reinterpret_cast<T *>(out->data());

    // 先将 out 填充为 input 的值（支持任意 strides）
    // 遍历所有逻辑坐标，计算 input/out 的线性偏移
    Shape idx(ndim, 0);
    bool done = false;
    while (!done) {
        // 计算线性偏移
        size_t in_offset = 0;
        size_t out_offset = 0;
        for (Size d = 0; d < ndim; ++d) {
            in_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(in_strides[d]);
            out_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(out_strides[d]);
        }

        out_ptr[out_offset] = in_ptr[in_offset];

        // 递增 idx（多维计数器）
        for (Size d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
            if (d == 0) {
                done = true;
            }
        }
    }

    // 再执行 scatter_add：沿给定 dim，根据 index 在 out 上累加 src
    // 重新从 0 开始遍历所有坐标
    std::fill(idx.begin(), idx.end(), 0);
    done = false;
    while (!done) {
        // 计算 index/src 的线性偏移
        size_t idx_offset = 0;
        size_t src_offset = 0;
        for (Size d = 0; d < ndim; ++d) {
            idx_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(idx_strides[d]);
            src_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(src_strides[d]);
        }

        long long idx_val = idx_ptr[idx_offset];
        if (idx_val < 0 || idx_val >= static_cast<long long>(shape[static_cast<Size>(dim)])) {
            throw std::runtime_error("scatter_add: index out of range along dim.");
        }

        // 目标位置坐标：将维度 dim 替换为 idx_val
        size_t out_offset = 0;
        for (Size d = 0; d < ndim; ++d) {
            Size coord = (static_cast<long long>(d) == dim) ? static_cast<Size>(idx_val) : idx[d];
            out_offset += static_cast<size_t>(coord) * static_cast<size_t>(out_strides[d]);
        }

        // 对于 float 类型可以直接累加；对于半精度/bfloat16，需要先转成 float32 再累加
        if constexpr (std::is_same_v<T, float>) {
            out_ptr[out_offset] += src_ptr[src_offset];
        } else {
            // 这里假定 T 是 uint16_t，具体是 F16 还是 BF16 由上层 dtype 控制
            uint16_t &dst_bits = *reinterpret_cast<uint16_t *>(&out_ptr[out_offset]);
            uint16_t src_bits = *reinterpret_cast<const uint16_t *>(&src_ptr[src_offset]);

            // 注意：这里仅用于 scatter_add，误差容忍由单测控制
            // 转成 float32 做加法，再转换回相应格式
            // 判断是 F16 还是 BF16 由 dim 上层对应的 dtype 决定，这里统一用 F16 转换，
            // 实际上我们在 calculate 中对 F16/BF16 分别实例化为 uint16_t，
            // 并通过不同的转换函数进行运算。
            // 为了简单，这里假设 T=float 已处理，T=uint16_t 的情况交由专门的实现。
        }

        // 递增 idx（多维计数器）
        for (Size d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
            if (d == 0) {
                done = true;
            }
        }
    }
}

// 专门处理 float32 的实现
static void scatter_add_f32(Tensor out, Tensor input, Tensor index, Tensor src, long long dim) {
    scatter_add_cpu_impl<float>(out, input, index, src, dim);
}

// 专门处理 F16/BF16：使用 uint16_t 存储，内部转换为 float32 运算
static void scatter_add_f16_bf16(Tensor out, Tensor input, Tensor index, Tensor src, long long dim, bool is_bf16) {
    const auto &shape = input->shape();
    const auto &in_strides = input->strides();
    const auto &out_strides = out->strides();
    const auto &idx_strides = index->strides();
    const auto &src_strides = src->strides();

    Size ndim = input->ndim();

    const long long *idx_ptr = reinterpret_cast<const long long *>(index->data());
    const uint16_t *src_ptr = reinterpret_cast<const uint16_t *>(src->data());
    const uint16_t *in_ptr = reinterpret_cast<const uint16_t *>(input->data());
    uint16_t *out_ptr = reinterpret_cast<uint16_t *>(out->data());

    // 先将 out 填充为 input 的值（支持任意 strides）
    Shape idx(ndim, 0);
    bool done = false;
    while (!done) {
        size_t in_offset = 0;
        size_t out_offset = 0;
        for (Size d = 0; d < ndim; ++d) {
            in_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(in_strides[d]);
            out_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(out_strides[d]);
        }

        out_ptr[out_offset] = in_ptr[in_offset];

        for (Size d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
            if (d == 0) {
                done = true;
            }
        }
    }

    // 再执行 scatter_add
    std::fill(idx.begin(), idx.end(), 0);
    done = false;
    while (!done) {
        size_t idx_offset = 0;
        size_t src_offset = 0;
        for (Size d = 0; d < ndim; ++d) {
            idx_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(idx_strides[d]);
            src_offset += static_cast<size_t>(idx[d]) * static_cast<size_t>(src_strides[d]);
        }

        long long idx_val = idx_ptr[idx_offset];
        if (idx_val < 0 || idx_val >= static_cast<long long>(shape[static_cast<Size>(dim)])) {
            throw std::runtime_error("scatter_add: index out of range along dim.");
        }

        size_t out_offset = 0;
        for (Size d = 0; d < ndim; ++d) {
            Size coord = (static_cast<long long>(d) == dim) ? static_cast<Size>(idx_val) : idx[d];
            out_offset += static_cast<size_t>(coord) * static_cast<size_t>(out_strides[d]);
        }

        // 半精度/BF16：转 float32 做加法，再转回
        uint16_t dst_bits = out_ptr[out_offset];
        uint16_t src_bits = src_ptr[src_offset];

        float dst_f = is_bf16 ? bf16_to_f32(dst_bits) : f16_to_f32(dst_bits);
        float src_f = is_bf16 ? bf16_to_f32(src_bits) : f16_to_f32(src_bits);

        float sum = dst_f + src_f;
        out_ptr[out_offset] = is_bf16 ? f32_to_bf16(sum) : f32_to_f16(sum);

        for (Size d = ndim; d-- > 0;) {
            idx[d]++;
            if (idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0;
            if (d == 0) {
                done = true;
            }
        }
    }
}

void calculate(Tensor out, Tensor input, Tensor index, Tensor src, long long dim) {
    if (out->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("ScatterAdd CPU kernel only supports CPU device.");
    }

    auto dtype = input->dtype();
    if (dtype == DataType::F16) {
        scatter_add_f16_bf16(out, input, index, src, dim, /*is_bf16=*/false);
    } else if (dtype == DataType::BF16) {
        scatter_add_f16_bf16(out, input, index, src, dim, /*is_bf16=*/true);
    } else if (dtype == DataType::F32) {
        scatter_add_f32(out, input, index, src, dim);
    } else {
        throw std::runtime_error("ScatterAdd CPU kernel only supports F16/BF16/F32.");
    }
}

static bool registered = []() {
    ScatterAdd::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::scatter_add_impl::cpu


