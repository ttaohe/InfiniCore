#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/argmax.hpp"
#include "ops/attention.hpp"
#include "ops/causal_softmax.hpp"
#include "ops/softmax.hpp"
#include "ops/embedding.hpp"
#include "ops/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/mul.hpp"
#include "ops/one_hot.hpp"
#include "ops/random_sample.hpp"
#include "ops/rearrange.hpp"
#include "ops/rms_norm.hpp"
#include "ops/rope.hpp"
#include "ops/scatter_add.hpp"
#include "ops/topk.hpp"
#include "ops/sum.hpp"
#include "ops/mean.hpp"
#include "ops/silu.hpp"
#include "ops/sigmoid.hpp"
#include "ops/swiglu.hpp"
#include "ops/test_mul.hpp"
#include "ops/conv2d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_argmax(m);
    bind_attention(m);
    bind_causal_softmax(m);
    bind_softmax(m);
    bind_random_sample(m);
    bind_linear(m);
    bind_matmul(m);
    bind_mul(m);
    bind_one_hot(m);
    bind_rearrange(m);
    bind_rms_norm(m);
    bind_scatter_add(m);
    bind_topk(m);
    bind_sum(m);
    bind_mean(m);
    bind_silu(m);
    bind_sigmoid(m);
    bind_swiglu(m);
    bind_rope(m);
    bind_embedding(m);
    bind_test_mul(m);
    bind_conv2d(m);
}

} // namespace infinicore::ops
