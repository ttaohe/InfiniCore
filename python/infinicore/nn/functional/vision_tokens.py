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


def build_vision_tokens(
    images: Tensor,
    patch_size,
    *,
    image_newline_embed: Tensor | None = None,
    view_separator_embed: Tensor | None = None,
) -> Tensor:
    """
    构建完整的视觉 token 流（不含投射到语言 hidden_dim 的 projector，仅做 token 排布）。

    支持：
      - 单视图: images 形状为 [B, C, H, W]
      - 多视图: images 形状为 [B, V, C, H, W]，其中 V 为视图数（多图 / 多裁剪）

    当前实现策略：
      - 为了与 PyTorch 参考实现数值严格对齐，**在 Python 端统一走 Torch 流程**：
          1. 将 InfiniCore Tensor 转成 torch.Tensor；
          2. 用 Torch 做 patchify + image_newline + view_separator 组合；
          3. 再用 `infinicore.from_torch` 转回 InfiniCore Tensor。
      - 单算子层面（`patchify` / `add_image_newline` / `add_view_separator`）已有独立单测，
        这里主要验证「整条视觉 token 流」在结构和排布上的正确性。
    """
    images_t = to_torch(images)

    if images_t.ndim == 4:
        # [B, C, H, W] -> [B, 1, C, H, W]
        B, C, H, W = images_t.shape
        V = 1
        images_t5 = images_t.view(B, 1, C, H, W)
    elif images_t.ndim == 5:
        B, V, C, H, W = images_t.shape
        images_t5 = images_t
    else:
        raise ValueError(
            f"build_vision_tokens: expect images.ndim in (4,5), got {images_t.ndim}"
        )

    P_h, P_w = _to_2tuple(patch_size)
    if H % P_h != 0 or W % P_w != 0:
        raise ValueError(
            f"build_vision_tokens: H={H}, W={W} must be divisible by patch_size={P_h}x{P_w}"
        )

    H_grid = H // P_h
    W_grid = W // P_w

    # Step 1: Torch patchify
    images_flat = images_t5.view(B * V, C, H, W)  # [B*V, C, H, W]
    x = images_flat.view(B * V, C, H_grid, P_h, W_grid, P_w)
    x = x.permute(0, 2, 4, 3, 5, 1)  # [B*V, H_grid, W_grid, P_h, P_w, C]
    patches_t = x.contiguous().view(B * V, H_grid * W_grid, P_h * P_w * C)
    _, N_view, D = patches_t.shape

    patches_t = patches_t.view(B, V, N_view, D)  # [B, V, N_view, D]

    # Step 2: 每个视图内插入 image_newline（可选）
    if image_newline_embed is not None:
        patches_t = patches_t.view(B * V, N_view, D)  # [B*V, N_view, D]

        nl_t = to_torch(image_newline_embed)
        if nl_t.ndim == 1:
            nl_t = nl_t.view(1, 1, D)
        elif nl_t.ndim == 2:
            if nl_t.shape[-1] != D:
                raise ValueError(
                    f"build_vision_tokens: image_newline_embed shape {nl_t.shape} incompatible with D={D}"
                )
            nl_t = nl_t.view(1, 1, D)
        elif nl_t.ndim == 3:
            if nl_t.shape[-1] != D:
                raise ValueError(
                    f"build_vision_tokens: image_newline_embed shape {nl_t.shape} incompatible with D={D}"
                )
            nl_t = nl_t[0:1, 0:1, :]
        else:
            raise ValueError(
                f"build_vision_tokens: unsupported image_newline_embed.ndim={nl_t.ndim}"
            )

        tokens_2d = patches_t.view(B * V, H_grid, W_grid, D)
        nl_row = nl_t.view(1, 1, 1, D).expand(B * V, H_grid, 1, D)
        out_t = torch.cat([tokens_2d, nl_row], dim=2)
        out_t = out_t.view(B * V, H_grid * (W_grid + 1), D)

        patches_t = out_t.view(B, V, H_grid * (W_grid + 1), D)
        per_view_len = H_grid * (W_grid + 1)
    else:
        per_view_len = N_view

    tokens_t = patches_t.view(B, V * per_view_len, D)  # [B, V*L, D]

    # Step 3: 在视图之间插入 view_separator（可选）
    if view_separator_embed is not None and V > 1:
        sep_t = to_torch(view_separator_embed)
        if sep_t.ndim == 1:
            if sep_t.shape[0] != D:
                raise ValueError(
                    f"build_vision_tokens: view_separator_embed shape {sep_t.shape} incompatible with D={D}"
                )
            sep_t = sep_t.view(1, 1, D)
        elif sep_t.ndim == 2:
            if sep_t.shape[-1] != D:
                raise ValueError(
                    f"build_vision_tokens: view_separator_embed shape {sep_t.shape} incompatible with D={D}"
                )
            sep_t = sep_t.view(1, 1, D)
        elif sep_t.ndim == 3:
            if sep_t.shape[-1] != D:
                raise ValueError(
                    f"build_vision_tokens: view_separator_embed shape {sep_t.shape} incompatible with D={D}"
                )
            sep_t = sep_t[0:1, 0:1, :]
        else:
            raise ValueError(
                f"build_vision_tokens: unsupported view_separator_embed.ndim={sep_t.ndim}"
            )

        sep_row = sep_t.view(1, 1, D).expand(B, 1, D)

        view_lengths = [per_view_len] * V
        chunks = list(tokens_t.split(view_lengths, dim=1))
        pieces = []
        for i, chunk in enumerate(chunks):
            pieces.append(chunk)
            if i != len(chunks) - 1:
                pieces.append(sep_row)
        tokens_t = torch.cat(pieces, dim=1)

    return infinicore.from_torch(tokens_t)




