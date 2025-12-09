#include "infinicore/ops/sum.hpp"

#include "../../utils.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace infinicore::op::sum_impl::cpu {

// float16 <-> float32
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    uint32_t f_bits;
    if (exp == 0) {
        if (mant == 0) {
            f_bits = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            exp = exp - 1 + 127 - 15;
            f_bits = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        f_bits = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        exp = exp - 15 + 127;
        f_bits = (sign << 31) | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &f_bits, sizeof(float));
    return f;
}

static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(uint32_t));

    uint32_t sign = (x >> 31) & 0x1;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;

    uint16_t h;
    if (exp <= 0) {
        if (exp < -10) {
            h = static_cast<uint16_t>(sign << 15);
        } else {
            mant |= 0x800000;
            uint32_t shift = static_cast<uint32_t>(1 - exp);
            mant = mant >> (shift + 13);
            h = static_cast<uint16_t>((sign << 15) | mant);
        }
    } else if (exp >= 31) {
        h = static_cast<uint16_t>((sign << 15) | 0x7c00);
    } else {
        h = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }
    return h;
}

// bfloat16 <-> float32
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

template <typename LoadFn, typename StoreFn>
void sum_contiguous_impl(Tensor out, Tensor input, long long dim, bool keepdim, LoadFn load, StoreFn store) {
    const auto &shape = input->shape();
    Size ndim = input->ndim();

    Size d = static_cast<Size>(dim);
    Size dim_size = shape[d];

    // treat as [outer, dim_size, inner]
    Size outer = 1;
    for (Size i = 0; i < d; ++i) {
        outer *= shape[i];
    }
    Size inner = 1;
    for (Size i = d + 1; i < ndim; ++i) {
        inner *= shape[i];
    }

    Size rows = outer * inner;

    const std::byte *in_raw = input->data();
    std::byte *out_raw = out->data();

    for (Size row = 0; row < rows; ++row) {
        Size outer_idx = (inner == 0) ? 0 : row / inner;
        Size inner_idx = (inner == 0) ? 0 : row % inner;

        Size base = (outer_idx * dim_size) * inner + inner_idx;

        float acc = 0.0f;
        for (Size j = 0; j < dim_size; ++j) {
            Size in_offset = base + j * inner;
            acc += load(in_raw, in_offset);
        }

        Size out_offset = row; // contiguous layout for both keepdim true/false
        store(out_raw, out_offset, acc);
    }
}

void calculate(Tensor out, Tensor input, long long dim, bool keepdim) {
    if (out->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("Sum CPU kernel only supports CPU device.");
    }

    auto dtype = input->dtype();

    if (dtype == DataType::F32) {
        auto load = [](const std::byte *data, Size offset) -> float {
            const float *p = reinterpret_cast<const float *>(data);
            return p[offset];
        };
        auto store = [](std::byte *data, Size offset, float v) {
            float *p = reinterpret_cast<float *>(data);
            p[offset] = v;
        };
        sum_contiguous_impl(out, input, dim, keepdim, load, store);
    } else if (dtype == DataType::F16) {
        auto load = [](const std::byte *data, Size offset) -> float {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
            return f16_to_f32(p[offset]);
        };
        auto store = [](std::byte *data, Size offset, float v) {
            uint16_t *p = reinterpret_cast<uint16_t *>(data);
            p[offset] = f32_to_f16(v);
        };
        sum_contiguous_impl(out, input, dim, keepdim, load, store);
    } else if (dtype == DataType::BF16) {
        auto load = [](const std::byte *data, Size offset) -> float {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
            return bf16_to_f32(p[offset]);
        };
        auto store = [](std::byte *data, Size offset, float v) {
            uint16_t *p = reinterpret_cast<uint16_t *>(data);
            p[offset] = f32_to_bf16(v);
        };
        sum_contiguous_impl(out, input, dim, keepdim, load, store);
    } else {
        throw std::runtime_error("Sum CPU kernel only supports F16/BF16/F32.");
    }
}

static bool registered = []() {
    Sum::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::sum_impl::cpu


