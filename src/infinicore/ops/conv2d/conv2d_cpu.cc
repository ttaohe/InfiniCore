#include "infinicore/ops/conv2d.hpp"

#include "../../utils.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace infinicore::op::conv2d_impl::cpu {

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

template <typename LoadIn, typename LoadW, typename LoadBias, typename StoreOut>
void conv2d_contiguous_impl(Tensor out,
                            Tensor input,
                            Tensor weight,
                            std::optional<Tensor> bias,
                            long long stride_h,
                            long long stride_w,
                            long long pad_h,
                            long long pad_w,
                            long long dilation_h,
                            long long dilation_w,
                            LoadIn load_in,
                            LoadW load_w,
                            LoadBias load_bias,
                            StoreOut store_out) {
    const auto &in_shape = input->shape();
    const auto &w_shape = weight->shape();
    const auto &out_shape = out->shape();

    Size N = in_shape[0];
    Size C_in = in_shape[1];
    Size H_in = in_shape[2];
    Size W_in = in_shape[3];

    Size C_out = w_shape[0];
    Size K_h = w_shape[2];
    Size K_w = w_shape[3];

    Size H_out = out_shape[2];
    Size W_out = out_shape[3];

    const std::byte *in_raw = input->data();
    const std::byte *w_raw = weight->data();
    const std::byte *b_raw = bias.has_value() ? bias.value()->data() : nullptr;
    std::byte *out_raw = out->data();

    for (Size n = 0; n < N; ++n) {
        for (Size oc = 0; oc < C_out; ++oc) {
            for (Size oh = 0; oh < H_out; ++oh) {
                for (Size ow = 0; ow < W_out; ++ow) {
                    float acc = 0.0f;
                    if (b_raw != nullptr) {
                        acc += load_bias(b_raw, oc);
                    }

                    long long ih_base = static_cast<long long>(oh) * stride_h - pad_h;
                    long long iw_base = static_cast<long long>(ow) * stride_w - pad_w;

                    for (Size ic = 0; ic < C_in; ++ic) {
                        for (Size kh = 0; kh < K_h; ++kh) {
                            long long ih = ih_base + static_cast<long long>(kh) * dilation_h;
                            if (ih < 0 || ih >= static_cast<long long>(H_in)) {
                                continue;
                            }

                            for (Size kw = 0; kw < K_w; ++kw) {
                                long long iw = iw_base + static_cast<long long>(kw) * dilation_w;
                                if (iw < 0 || iw >= static_cast<long long>(W_in)) {
                                    continue;
                                }

                                // input offset: ((n*C_in + ic)*H_in + ih)*W_in + iw
                                Size in_offset = (((n * C_in + ic) * H_in + static_cast<Size>(ih)) * W_in
                                                  + static_cast<Size>(iw));
                                // weight offset: ((oc*C_in + ic)*K_h + kh)*K_w + kw
                                Size w_offset = (((oc * C_in + ic) * K_h + kh) * K_w + kw);

                                float v_in = load_in(in_raw, in_offset);
                                float v_w = load_w(w_raw, w_offset);
                                acc += v_in * v_w;
                            }
                        }
                    }

                    // out offset: ((n*C_out + oc)*H_out + oh)*W_out + ow
                    Size out_offset = (((n * C_out + oc) * H_out + oh) * W_out + ow);
                    store_out(out_raw, out_offset, acc);
                }
            }
        }
    }
}

void calculate(Tensor out,
               Tensor input,
               Tensor weight,
               std::optional<Tensor> bias,
               long long stride_h,
               long long stride_w,
               long long pad_h,
               long long pad_w,
               long long dilation_h,
               long long dilation_w,
               long long /*groups*/) {
    if (out->device().getType() != Device::Type::CPU) {
        throw std::runtime_error("Conv2d CPU kernel only supports CPU device.");
    }

    auto dtype = input->dtype();

    if (dtype == DataType::F32) {
        auto load_in = [](const std::byte *data, Size offset) -> float {
            const float *p = reinterpret_cast<const float *>(data);
            return p[offset];
        };
        auto load_w = load_in;
        auto load_bias = [](const std::byte *data, Size oc) -> float {
            const float *p = reinterpret_cast<const float *>(data);
            return p[oc];
        };
        auto store_out = [](std::byte *data, Size offset, float v) {
            float *p = reinterpret_cast<float *>(data);
            p[offset] = v;
        };
        conv2d_contiguous_impl(out,
                               input,
                               weight,
                               bias,
                               stride_h,
                               stride_w,
                               pad_h,
                               pad_w,
                               dilation_h,
                               dilation_w,
                               load_in,
                               load_w,
                               load_bias,
                               store_out);
    } else if (dtype == DataType::F16) {
        auto load_in = [](const std::byte *data, Size offset) -> float {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
            return f16_to_f32(p[offset]);
        };
        auto load_w = load_in;
        auto load_bias = [](const std::byte *data, Size oc) -> float {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
            return f16_to_f32(p[oc]);
        };
        auto store_out = [](std::byte *data, Size offset, float v) {
            uint16_t *p = reinterpret_cast<uint16_t *>(data);
            p[offset] = f32_to_f16(v);
        };
        conv2d_contiguous_impl(out,
                               input,
                               weight,
                               bias,
                               stride_h,
                               stride_w,
                               pad_h,
                               pad_w,
                               dilation_h,
                               dilation_w,
                               load_in,
                               load_w,
                               load_bias,
                               store_out);
    } else if (dtype == DataType::BF16) {
        auto load_in = [](const std::byte *data, Size offset) -> float {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
            return bf16_to_f32(p[offset]);
        };
        auto load_w = load_in;
        auto load_bias = [](const std::byte *data, Size oc) -> float {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
            return bf16_to_f32(p[oc]);
        };
        auto store_out = [](std::byte *data, Size offset, float v) {
            uint16_t *p = reinterpret_cast<uint16_t *>(data);
            p[offset] = f32_to_bf16(v);
        };
        conv2d_contiguous_impl(out,
                               input,
                               weight,
                               bias,
                               stride_h,
                               stride_w,
                               pad_h,
                               pad_w,
                               dilation_h,
                               dilation_w,
                               load_in,
                               load_w,
                               load_bias,
                               store_out);
    } else {
        throw std::runtime_error("Conv2d CPU kernel only supports F16/BF16/F32.");
    }
}

static bool registered = []() {
    Conv2d::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::conv2d_impl::cpu



