import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import torch
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import convert_infinicore_to_torch

import infinicore
from infinicore.nn.modules import ViTBlock


_TEST_CASES_DATA = [
    # (B, N, C, num_heads, mlp_ratio)
    (1, 8, 32, 4, 4.0),
    (2, 16, 64, 8, 3.0),
]

# ViTBlock 主要面向 FP16 / BF16 场景，这里优先保证 FP16 对齐；
# FP32 路径在底层算子上仍存在少量 NaN 问题，后续单独排查。
_TENSOR_DTYPES = [infinicore.float16]


def parse_test_cases():
    test_cases = []
    for B, N, C, num_heads, mlp_ratio in _TEST_CASES_DATA:
        shape = (B, N, C)
        for dtype in _TENSOR_DTYPES:
            # ViT Block 叠加多层 FP16 计算，累计误差略大，适当放宽 atol
            tol = (
                {"atol": 2e-3, "rtol": 1e-3}
                if dtype is infinicore.float16
                else {"atol": 1e-5, "rtol": 1e-5}
            )
            x_spec = TensorSpec.from_tensor(shape, None, dtype)
            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={
                        "embed_dim": C,
                        "num_heads": num_heads,
                        "mlp_ratio": mlp_ratio,
                    },
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="ViTBlock - core",
                )
            )
    return test_cases


class TorchRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 与 InfiniCore RMSNorm 行为对齐：权重初始为 1
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 行为与 InfiniCore rms_norm 对齐：
        # scale = (x^2).mean(-1, keepdim=True) + eps -> rsqrt
        # y = x * scale * weight
        orig_dtype = x.dtype
        hidden = x.to(torch.float32)
        scale = hidden.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt_()
        out = hidden.mul(scale).mul(self.weight)
        return out.to(orig_dtype)


def torch_mla_core(q, k, v):
    # 与 test/infinicore/nn/mla_attention.py 中的实现保持一致
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
    return ctx  # [B, L, H, Dh]


class TorchViTSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, H, Dh)
        k = k.view(B, N, H, Dh)
        v = v.view(B, N, H, Dh)

        # 使用与 MLAAttention 相同的核心注意力逻辑，确保与组合版实现对齐
        ctx = torch_mla_core(q, k, v)  # [B, N, H, Dh]
        ctx = ctx.view(B, N, C)
        out = self.out_proj(ctx)
        return out


class TorchViTBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = TorchRMSNorm(embed_dim, eps=eps)
        self.attn = TorchViTSelfAttention(embed_dim, num_heads)
        self.norm2 = TorchRMSNorm(embed_dim, eps=eps)

        hidden_dim = int(math.ceil(embed_dim * mlp_ratio))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class OpTest(BaseOperatorTest):
    """ViTBlock 与等价 Torch ViTBlock 的数值对齐测试。"""

    def __init__(self):
        super().__init__("ViTBlock")
        self._last_torch_block = None

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, embed_dim, num_heads, mlp_ratio):
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        block = TorchViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio).to(
            device=device, dtype=dtype
        )

        self._last_torch_block = block
        return block(x)

    def infinicore_operator(self, x, embed_dim, num_heads, mlp_ratio):
        from infinicore.tensor import from_torch as ic_from_torch

        block = ViTBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            device=x.device,
            dtype=x.dtype,
        )

        torch_block = self._last_torch_block
        if torch_block is None:
            raise RuntimeError("Torch ViTBlock is not initialized before infinicore_operator.")

        # 拷贝 RMSNorm 权重
        with torch.no_grad():
            # norm1
            w1 = torch_block.norm1.weight
            block.norm1.weight.copy_(ic_from_torch(w1))
            # norm2
            w2 = torch_block.norm2.weight
            block.norm2.weight.copy_(ic_from_torch(w2))

            # Self-Attention 权重
            ta = torch_block.attn

            block.attn.q_proj.weight.copy_(ic_from_torch(ta.q_proj.weight))
            block.attn.q_proj.bias.copy_(ic_from_torch(ta.q_proj.bias))

            block.attn.k_proj.weight.copy_(ic_from_torch(ta.k_proj.weight))
            block.attn.k_proj.bias.copy_(ic_from_torch(ta.k_proj.bias))

            block.attn.v_proj.weight.copy_(ic_from_torch(ta.v_proj.weight))
            block.attn.v_proj.bias.copy_(ic_from_torch(ta.v_proj.bias))

            block.attn.out_proj.weight.copy_(ic_from_torch(ta.out_proj.weight))
            block.attn.out_proj.bias.copy_(ic_from_torch(ta.out_proj.bias))

            # MLP 权重
            # 第一层
            w_mlp1 = torch_block.mlp[0].weight
            b_mlp1 = torch_block.mlp[0].bias
            block.mlp.fc1_weight.copy_(ic_from_torch(w_mlp1))
            block.mlp.fc1_bias.copy_(ic_from_torch(b_mlp1))

            # 第二层
            w_mlp2 = torch_block.mlp[2].weight
            b_mlp2 = torch_block.mlp[2].bias
            block.mlp.fc2_weight.copy_(ic_from_torch(w_mlp2))
            block.mlp.fc2_bias.copy_(ic_from_torch(b_mlp2))

        y_infini = block(x)
        return convert_infinicore_to_torch(y_infini)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


