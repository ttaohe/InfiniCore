import math

import torch
import infinicore
from infinicore.tensor import Tensor

from ..functional import RopeAlgo, softmax
from infinicore.utils import to_torch
from .module import InfiniCoreModule as Module
from .rope import RoPE


class MLAAttention(Module):
    r"""
    Mixed Linear Attention（MLA）风格的多头自注意力“组合版”实现。

    约定：
      - 这里只实现核心的 Q/K/V 注意力部分，不包含 Q/K/V 的线性投影和输出投影；
      - 输入 Q/K/V 均为 `[B, seq_len, num_heads, head_dim]` 形状；
      - 返回同形状 `[B, seq_len, num_heads, head_dim]` 的上下文张量；
      - 可选地对 Q/K 应用 RoPE（GPT‑J / GPT‑NeoX 风格），便于与现有 `nn.RoPE`/`F.rope` 统一使用；
      - 目前不在内部管理 KV cache，增量 KV 由 `init_kv_cache / update_kv_cache / slice_kv_cache`
        等 helper 在上层组合。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        use_rope: bool = False,
        max_position_embeddings: int | None = None,
        rope_theta: float = 10000.0,
        rope_algo: RopeAlgo = RopeAlgo.GPT_NEOX,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError(f"MLAAttention: head_dim must be positive, got {head_dim}")
        if num_heads <= 0:
            raise ValueError(f"MLAAttention: num_heads must be positive, got {num_heads}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.use_rope = use_rope
        self.rope_algo = rope_algo

        if use_rope:
            if max_position_embeddings is None:
                raise ValueError(
                    "MLAAttention: max_position_embeddings must be provided when use_rope=True"
                )
            factory_kwargs = {
                "device": infinicore.device("cpu", 0) if device is None else device,
                "dtype": infinicore.float32 if dtype is None else dtype,
            }
            # 这里复用 Python 版 nn.RoPE（内部预计算 sin/cos cache）
            self.rope = RoPE(
                max_position_embeddings,
                rope_theta,
                head_dim,
                **factory_kwargs,
            )
        else:
            self.rope = None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            q/k/v: `[B, seq_len, num_heads, head_dim]` 的张量，通常是若干 Linear 投影后的结果。
            position_ids: `[B, seq_len]` 的位置索引，仅在 `use_rope=True` 时需要。

        Returns:
            context: `[B, seq_len, num_heads, head_dim]`
        """
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                f"MLAAttention: q/k/v shape mismatch, got q={q.shape}, k={k.shape}, v={v.shape}"
            )

        B, L, H, Dh = q.shape
        if H != self.num_heads or Dh != self.head_dim:
            raise ValueError(
                f"MLAAttention: expected num_heads={self.num_heads}, head_dim={self.head_dim}, "
                f"but got num_heads={H}, head_dim={Dh}"
            )

        x_q = q
        x_k = k
        x_v = v

        # 可选 RoPE：对 Q/K 应用旋转位置编码
        if self.use_rope:
            if position_ids is None:
                raise ValueError("MLAAttention: position_ids is required when use_rope=True")
            x_q = self.rope(x_q, position_ids, algo=self.rope_algo)
            x_k = self.rope(x_k, position_ids, algo=self.rope_algo)

        # 统一在连续内存上做后续 view
        x_q = x_q.contiguous()
        x_k = x_k.contiguous()
        x_v = x_v.contiguous()

        # [B, L, H, Dh] -> [B*H, L, Dh]
        q_2d = x_q.view([B * H, L, Dh])
        k_2d = x_k.view([B * H, L, Dh])
        v_2d = x_v.view([B * H, L, Dh])

        # 注意力分数: [B*H, L, L] = [B*H, L, Dh] @ [B*H, Dh, L]
        k_t = k_2d.permute([0, 2, 1])
        # 直接通过 matmul 的 alpha 参数做缩放，避免额外的逐元素乘法
        scores = infinicore.matmul(q_2d, k_t, alpha=self.scale)

        # softmax over last dim (L)
        if q.device.type == "cpu":
            # 当前 InfiniOP softmax 对 CPU 支持有限，这里在 CPU 上走一次 Torch fallback，
            # 再通过 from_torch 转回 InfiniCore Tensor，用于组合版 MLA 和单测场景即可。
            scores_t = to_torch(scores)
            attn_t = torch.softmax(scores_t, dim=-1)
            attn = infinicore.from_torch(attn_t)
        else:
            attn = softmax(scores, dim=-1)

        # 上下文: [B*H, L, Dh] = [B*H, L, L] @ [B*H, L, Dh]
        context_2d = infinicore.matmul(attn, v_2d)

        # 还原回 [B, L, H, Dh]
        context = context_2d.view([B, H, L, Dh]).permute([0, 2, 1, 3])
        return context



