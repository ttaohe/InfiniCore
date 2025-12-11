from __future__ import annotations

import math

import infinicore
from infinicore.tensor import Tensor

from .. import functional as F
from .linear import Linear
from .module import InfiniCoreModule as Module
from .normalization import RMSNorm
from .mla import MLAAttention
from .projector import MlpProjector


class ViTSelfAttention(Module):
    r"""
    多头自注意力模块，适用于 ViT 形式的输入：

      - 输入:  x, shape = [B, N, C]，其中 C = num_heads * head_dim
      - 输出:  y, shape = [B, N, C]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"ViTSelfAttention: embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }

        # Q/K/V 投影和输出投影
        self.q_proj = Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.k_proj = Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        # 使用组合版 MLA Attention 作为核心注意力核（不启用 RoPE）
        self.mla = MLAAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            use_rope=False,
            device=factory_kwargs["device"],
            dtype=factory_kwargs["dtype"],
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, N, C]
        return: [B, N, C]
        """
        B, N, C = x.shape
        if C != self.embed_dim:
            raise ValueError(
                f"ViTSelfAttention: expected embed_dim={self.embed_dim}, got C={C}"
            )

        # 线性投影得到 Q/K/V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, N, C] -> [B, N, H, Dh]
        H = self.num_heads
        Dh = self.head_dim
        q = q.view([B, N, H, Dh])
        k = k.view([B, N, H, Dh])
        v = v.view([B, N, H, Dh])

        # MLA Attention 核（不使用 RoPE）
        ctx = self.mla(q, k, v)  # [B, N, H, Dh]
        # 由于 MLA 内部经过 permute，结果通常为非连续，需要先 contiguous 再 view
        ctx = ctx.contiguous().view([B, N, C])

        # 输出投影
        out = self.out_proj(ctx)
        return out


class ViTBlock(Module):
    r"""
    一个标准的 ViT Encoder Block（Pre-Norm 结构，RMSNorm + MSA + MLP）：

      - 输入:  x, shape = [B, N, C]
      - 输出:  y, shape = [B, N, C]

    结构：
      1) x = x + SelfAttention(RMSNorm(x))
      2) x = x + MLP(RMSNorm(x))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if mlp_ratio <= 0:
            raise ValueError(f"ViTBlock: mlp_ratio must be positive, got {mlp_ratio}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }

        # Pre-Norm
        self.norm1 = RMSNorm(embed_dim, eps=eps, **factory_kwargs)
        self.attn = ViTSelfAttention(embed_dim, num_heads, **factory_kwargs)
        self.norm2 = RMSNorm(embed_dim, eps=eps, **factory_kwargs)

        hidden_dim = int(math.ceil(embed_dim * mlp_ratio))

        # MLP 子层：结构等价于 [Linear(C, hidden) + SiLU + Linear(hidden, C)]
        self.mlp = MlpProjector(
            in_features=embed_dim,
            out_features=embed_dim,
            hidden_features=hidden_dim,
            activation="silu",
            device=factory_kwargs["device"],
            dtype=factory_kwargs["dtype"],
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, N, C]
        return: [B, N, C]
        """
        # Self-Attention 残差块
        x = x + self.attn(self.norm1(x))
        # MLP 残差块
        x = x + self.mlp(self.norm2(x))
        return x



