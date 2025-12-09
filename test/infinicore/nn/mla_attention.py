import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import torch
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import convert_infinicore_to_torch

import infinicore
from infinicore.nn.modules import MLAAttention


_TEST_CASES_DATA = [
    # (B, L, H, Dh)
    (1, 8, 4, 16),
    (2, 4, 8, 8),
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for B, L, H, Dh in _TEST_CASES_DATA:
        shape = (B, L, H, Dh)
        for dtype in _TENSOR_DTYPES:
            tol = {"atol": 1e-3, "rtol": 1e-3} if dtype is infinicore.float16 else {"atol": 1e-5, "rtol": 1e-5}

            q_spec = TensorSpec.from_tensor(shape, None, dtype)
            k_spec = TensorSpec.from_tensor(shape, None, dtype)
            v_spec = TensorSpec.from_tensor(shape, None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[q_spec, k_spec, v_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="MLAAttention - core",
                )
            )

    return test_cases


def torch_mla_core(q, k, v):
    # q/k/v: [B, L, H, Dh]
    B, L, H, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)

    q_2d = q.reshape(B * H, L, Dh)
    k_2d = k.reshape(B * H, L, Dh)
    v_2d = v.reshape(B * H, L, Dh)

    scores = torch.matmul(q_2d, k_2d.transpose(-1, -2)) * scale  # [BH, L, L]
    attn = torch.softmax(scores, dim=-1)
    ctx_2d = torch.matmul(attn, v_2d)  # [BH, L, Dh]
    ctx = ctx_2d.reshape(B, H, L, Dh).permute(0, 2, 1, 3).contiguous()
    return ctx


class OpTest(BaseOperatorTest):
    """MLA Attention core test (no RoPE, no projections, just Q/K/V attention)."""

    def __init__(self):
        super().__init__("MLAAttention")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, q, k, v):
        return torch_mla_core(q, k, v)

    def infinicore_operator(self, q, k, v):
        # 构造一个 MLA 模块，仅用于核心 attention 测试（不启用 RoPE）
        _, _, H, Dh = q.shape
        mla = MLAAttention(num_heads=H, head_dim=Dh, use_rope=False, device=q.device, dtype=q.dtype)
        out = mla(q, k, v)
        # 转回 torch 做比较
        return convert_infinicore_to_torch(out)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


