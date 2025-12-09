import infinicore
from infinicore.tensor import Tensor

from .. import functional as F
from ..parameter import InfiniCoreParameter as Parameter
from .module import InfiniCoreModule as Module


class MlpProjector(Module):
    r"""
    一个简单的 MLP Projector，用于把视觉特征投射到语言侧 hidden_dim。

    典型用法（与 ViT / DeepSeek‑OCR 中的 projector 相似）：
      - 输入:  x,  shape = [B, N, C_in]（如 patch embedding）
      - 输出:  y,  shape = [B, N, C_out]

    结构：
      - Linear(C_in -> hidden_dim)
      - 激活（默认 SiLU）
      - Linear(hidden_dim -> C_out)
    """

    __constants__ = ["in_features", "hidden_features", "out_features"]
    in_features: int
    hidden_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        *,
        activation: str = "silu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        if hidden_features is None:
            hidden_features = out_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation

        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }

        # 第一层权重 [hidden_features, in_features]
        self.fc1_weight = Parameter(
            infinicore.empty([hidden_features, in_features], **factory_kwargs)
        )
        self.fc1_bias = Parameter(
            infinicore.empty([hidden_features], **factory_kwargs)
        )

        # 第二层权重 [out_features, hidden_features]
        self.fc2_weight = Parameter(
            infinicore.empty([out_features, hidden_features], **factory_kwargs)
        )
        self.fc2_bias = Parameter(
            infinicore.empty([out_features], **factory_kwargs)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, N, C_in]
        return: [B, N, C_out]
        """
        # 先做第一层 Linear
        x = F.linear(x, self.fc1_weight, self.fc1_bias)

        # 激活
        if self.activation == "silu":
            x = F.silu(x)
        elif self.activation == "gelu":
            # 目前没有单独的 GELU kernel，这里暂时用 PyTorch GELU fallback
            import torch
            from infinicore.utils import to_torch
            from infinicore.tensor import from_torch as ic_from_torch

            x_t = to_torch(x)
            x_t = torch.nn.functional.gelu(x_t)
            x = ic_from_torch(x_t)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        # 第二层 Linear
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, activation={self.activation}"
        )



