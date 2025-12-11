from __future__ import annotations

import infinicore
from infinicore.tensor import Tensor

from .. import functional as F
from .module import InfiniCoreModule as Module
from .projector import MlpProjector


class VisionFrontend(Module):
    r"""
    一个简单的 Vision Frontend：

      - 输入： images
          * 单视图: [B, C, H, W]
          * 多视图: [B, V, C, H, W]
      - 输出： vision tokens 投射到语言 hidden_dim：
          * [B, T, hidden_dim]，其中 T 为视觉 token 序列长度
    """

    def __init__(
        self,
        *,
        in_channels: int,
        patch_size: int | tuple[int, int],
        vision_embed_dim: int,
        hidden_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.hidden_dim = hidden_dim

        # projector: [B, T, D_v] -> [B, T, hidden_dim]
        self.projector = MlpProjector(
            in_features=vision_embed_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        images: Tensor,
        *,
        image_newline_embed: Tensor | None = None,
        view_separator_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            images: [B, C, H, W] 或 [B, V, C, H, W]
            image_newline_embed: 视觉换行 token embedding，维度 D_v
            view_separator_embed: 视图分隔 token embedding，维度 D_v

        Returns:
            tokens: [B, T, hidden_dim]
        """
        # Step 1: 构建视觉 token 序列（D_v 维）
        vision_tokens = F.build_vision_tokens(
            images,
            self.patch_size,
            image_newline_embed=image_newline_embed,
            view_separator_embed=view_separator_embed,
        )  # [B, T, D_v]

        # 检查维度与 projector 输入是否匹配
        if vision_tokens.shape[-1] != self.vision_embed_dim:
            raise ValueError(
                f"VisionFrontend: expected vision_embed_dim={self.vision_embed_dim}, "
                f"but got tokens_dim={vision_tokens.shape[-1]}"
            )

        # Step 2: 通过 MlpProjector 投射到语言 hidden_dim
        tokens = self.projector(vision_tokens)  # [B, T, hidden_dim]
        return tokens



