from __future__ import annotations

from typing import Tuple

import torch

import infinicore
from infinicore.tensor import Tensor
from infinicore.utils import to_torch


def _to_2tuple(val) -> Tuple[int, int]:
    if isinstance(val, (tuple, list)):
        assert len(val) == 2
        return int(val[0]), int(val[1])
    v = int(val)
    return v, v


def add_image_newline(
    tokens: Tensor,
    image_newline_embed: Tensor,
    grid_size,
) -> Tensor:
    """
    在 patch token 序列上按“每行末尾追加一个 image_newline” 的方式做拼接。

    典型场景（与 DeepSeek‑OCR 视觉 token 处理类似）：
      - tokens:           [B, N, C]，其中 N = H_grid * W_grid
      - grid_size:        (H_grid, W_grid)
      - image_newline_embed: 位置无关的 “行结束 token” embedding，形状可以是
          - [C]
          - [1, C]
          - [1, 1, C]（将广播到 [B, H_grid, 1, C]）

    操作：
      1. 将 tokens reshape 为 [B, H_grid, W_grid, C]；
      2. 构造一个 [B, H_grid, 1, C] 的 image_newline 行，拼接到 dim=2（列）末尾；
      3. 再展平为 [B, H_grid * (W_grid + 1), C] 返回。

    注意：
      - 当前实现使用 Torch 进行高层拼装（cat/reshape），再通过 from_torch 转回 InfiniCore，
        主要用于 **视觉 token 排布逻辑** 的组装和验证，对性能不敏感的场景足够。
    """
    if tokens.ndim != 3:
        raise ValueError(f"add_image_newline: expect tokens.ndim==3 (B,N,C), got {tokens.ndim}")

    B, N, C = tokens.shape
    H_grid, W_grid = _to_2tuple(grid_size)
    if H_grid * W_grid != N:
        raise ValueError(
            f"add_image_newline: grid_size ({H_grid},{W_grid}) not compatible with N={N}"
        )

    # 转成 torch 做拼接
    tokens_t = to_torch(tokens)  # [B, N, C]
    newline_t = to_torch(image_newline_embed)  # 支持 [C] / [1,C] / [1,1,C]

    # 规范 image_newline_embed 形状为 [1,1,C]
    if newline_t.ndim == 1:
        newline_t = newline_t.view(1, 1, C)
    elif newline_t.ndim == 2:
        if newline_t.shape[-1] != C:
            raise ValueError(
                f"add_image_newline: image_newline_embed shape {newline_t.shape} incompatible with C={C}"
            )
        newline_t = newline_t.view(1, 1, C)
    elif newline_t.ndim == 3:
        if newline_t.shape[-1] != C:
            raise ValueError(
                f"add_image_newline: image_newline_embed shape {newline_t.shape} incompatible with C={C}"
            )
        # 形如 [1,1,C] 或 [B,1,C]，统一处理成 [1,1,C] 以便广播
        newline_t = newline_t[0:1, 0:1, :]
    else:
        raise ValueError(
            f"add_image_newline: unsupported image_newline_embed.ndim={newline_t.ndim}"
        )

    # [B,N,C] -> [B,H,W,C]
    tokens_t = tokens_t.view(B, H_grid, W_grid, C)

    # [1,1,C] -> [B,H,1,C]
    newline_row = newline_t.view(1, 1, 1, C).expand(B, H_grid, 1, C)

    # 在列维度拼接 -> [B,H,W+1,C]
    out_t = torch.cat([tokens_t, newline_row], dim=2)

    # 展平回 [B, H*(W+1), C]
    out_t = out_t.view(B, H_grid * (W_grid + 1), C)

    # 转回 InfiniCore Tensor
    return infinicore.from_torch(out_t)



