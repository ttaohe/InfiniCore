from __future__ import annotations

from typing import Iterable, List

import torch

import infinicore
from infinicore.tensor import Tensor
from infinicore.utils import to_torch


def add_view_separator(
    tokens: Tensor,
    view_lengths: Iterable[int],
    view_separator_embed: Tensor,
) -> Tensor:
    """
    在一维 token 序列中，为多个“视图段”之间插入 view_separator token。

    典型场景（多视图图像 / patch 序列）：
      - tokens: [B, N, C]，其中 N = sum(view_lengths)
      - view_lengths: 可迭代整型，形如 [n_view0, n_view1, ...]，对应每个视图的 token 数；
      - view_separator_embed: 视图分隔 token 的 embedding，形状可以是：
          - [C]
          - [1, C]
          - [1, 1, C]

    操作：
      1. 将 tokens 沿 dim=1 按 view_lengths 切成若干段；
      2. 在相邻两段之间插入一个 view_separator token；
      3. 返回新的序列，形状为 [B, N + (num_views - 1), C]。

    注意：
      - 当前实现为了简单可靠，使用 Torch 来做拼接（split/cat），
        然后通过 from_torch 转回 InfiniCore Tensor；
      - 适合作为 DeepSeek‑OCR 中 view_separator 处理的高层 helper。
    """
    if tokens.ndim != 3:
        raise ValueError(f"add_view_separator: expect tokens.ndim==3 (B,N,C), got {tokens.ndim}")

    B, N, C = tokens.shape
    view_lengths = list(int(l) for l in view_lengths)
    if sum(view_lengths) != N:
        raise ValueError(
            f"add_view_separator: sum(view_lengths)={sum(view_lengths)} != N={N}"
        )

    num_views = len(view_lengths)
    if num_views <= 1:
        # 只有一个视图时，不需要插入 separator，直接返回原始 tokens
        return tokens

    # 转成 torch 做拼接
    tokens_t = to_torch(tokens)  # [B, N, C]
    sep_t = to_torch(view_separator_embed)

    # 规范 separator embedding 形状为 [1, 1, C]
    if sep_t.ndim == 1:
        if sep_t.shape[0] != C:
            raise ValueError(
                f"add_view_separator: view_separator_embed shape {sep_t.shape} incompatible with C={C}"
            )
        sep_t = sep_t.view(1, 1, C)
    elif sep_t.ndim == 2:
        if sep_t.shape[-1] != C:
            raise ValueError(
                f"add_view_separator: view_separator_embed shape {sep_t.shape} incompatible with C={C}"
            )
        sep_t = sep_t.view(1, 1, C)
    elif sep_t.ndim == 3:
        if sep_t.shape[-1] != C:
            raise ValueError(
                f"add_view_separator: view_separator_embed shape {sep_t.shape} incompatible with C={C}"
            )
        # 统一裁成 [1,1,C]，方便后续广播
        sep_t = sep_t[0:1, 0:1, :]
    else:
        raise ValueError(
            f"add_view_separator: unsupported view_separator_embed.ndim={sep_t.ndim}"
        )

    # [1,1,C] -> [B,1,C]
    sep_row = sep_t.view(1, 1, C).expand(B, 1, C)

    # 沿 dim=1 按 view_lengths 切分
    splits: List[torch.Tensor] = list(tokens_t.split(view_lengths, dim=1))

    # 拼接：view0, sep, view1, sep, ..., view_{k-1}, sep, view_k（最后一段后面不加）
    pieces: List[torch.Tensor] = []
    for i, chunk in enumerate(splits):
        pieces.append(chunk)
        if i != len(splits) - 1:
            pieces.append(sep_row)

    out_t = torch.cat(pieces, dim=1)  # [B, N + (num_views-1), C]
    return infinicore.from_torch(out_t)



