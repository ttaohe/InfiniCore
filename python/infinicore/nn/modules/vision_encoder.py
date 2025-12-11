from __future__ import annotations

import infinicore
from infinicore.tensor import Tensor

from .. import functional as F
from .module import InfiniCoreModule as Module
from .container import InfiniCoreModuleList as ModuleList
from .vision_frontend import VisionFrontend
from .vit_block import ViTBlock
from .normalization import RMSNorm


class VisionEncoder(Module):
    r"""
    一个简化版的 Vision Encoder，用于将图像编码成语言侧可用的视觉 token 序列。

    结构示意：
      images -> VisionFrontend(patchify + image_newline/view_separator + projector)
             -> N 层 ViTBlock 堆叠 (Pre-Norm: RMSNorm + Self-Attn + MLP)
             -> (可选) 最后一层 RMSNorm

    输入：
      - 单视图: [B, C, H, W]
      - 多视图: [B, V, C, H, W]

    输出：
      - tokens: [B, T, hidden_dim]，其中 T 为视觉 token 序列长度。
    """

    def __init__(
        self,
        *,
        in_channels: int,
        patch_size: int | tuple[int, int],
        vision_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        final_norm: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"VisionEncoder: num_layers must be positive, got {num_layers}")

        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.eps = eps
        self.final_norm = final_norm

        # 前端：从图像构建视觉 token 序列并投射到 hidden_dim
        self.frontend = VisionFrontend(
            in_channels=in_channels,
            patch_size=patch_size,
            vision_embed_dim=vision_embed_dim,
            hidden_dim=hidden_dim,
            device=factory_kwargs["device"],
            dtype=factory_kwargs["dtype"],
        )

        # ViTBlock 堆叠
        blocks = []
        for _ in range(num_layers):
            blocks.append(
                ViTBlock(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    device=factory_kwargs["device"],
                    dtype=factory_kwargs["dtype"],
                )
            )
        self.blocks = ModuleList(blocks)

        # 可选的最终 RMSNorm
        self.norm = (
            RMSNorm(hidden_dim, eps=eps, device=factory_kwargs["device"], dtype=factory_kwargs["dtype"])
            if final_norm
            else None
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
        # Step 1: 图像 -> 视觉 token 序列（已投射到 hidden_dim）
        x = self.frontend(
            images,
            image_newline_embed=image_newline_embed,
            view_separator_embed=view_separator_embed,
        )  # [B, T, hidden_dim]

        # Step 2: ViTBlock 堆叠
        for blk in self.blocks:
            x = blk(x)

        # Step 3: 最终 RMSNorm（可选）
        if self.norm is not None:
            x = self.norm(x)

        return x




