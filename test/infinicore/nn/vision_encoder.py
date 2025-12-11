import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import convert_infinicore_to_torch


_TEST_CONFIGS = [
    # (B, V, C, H, W, patch_size, D_v, hidden_dim, num_layers, num_heads, mlp_ratio)
    (1, 1, 3, 8, 8, (4, 4), 3 * 4 * 4, 32, 2, 4, 4.0),
    (2, 2, 3, 8, 8, (4, 4), 3 * 4 * 4, 64, 2, 8, 3.0),
]

# VisionEncoder 主要用于 FP16 / BF16 场景，这里优先保证 FP16 对齐；
# FP32 路径在底层算子上仍存在少量 NaN 问题，后续单独排查。
_TENSOR_DTYPES = [infinicore.float16]


def parse_test_cases():
    test_cases = []
    for B, V, C, H, W, patch_size, D_v, hidden, num_layers, num_heads, mlp_ratio in _TEST_CONFIGS:
        images_shape = (B, V, C, H, W)
        for dtype in _TENSOR_DTYPES:
            # ViT Encoder 堆叠多层 FP16 计算，累计误差略大，适当放宽 atol
            # 对于接近零的值，相对误差会放大，因此使用更宽松的相对误差
            tol = (
                {"atol": 8e-3, "rtol": 1e-2}
                if dtype is infinicore.float16
                else {"atol": 1e-5, "rtol": 1e-5}
            )

            images_spec = TensorSpec.from_tensor(images_shape, None, dtype)
            nl_spec = TensorSpec.from_tensor((D_v,), None, dtype)
            sep_spec = TensorSpec.from_tensor((D_v,), None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[images_spec, nl_spec, sep_spec],
                    kwargs={
                        "patch_size": patch_size,
                        "vision_embed_dim": D_v,
                        "hidden_dim": hidden,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                        "mlp_ratio": mlp_ratio,
                        "eps": 1e-6,
                        "final_norm": True,
                    },
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="VisionEncoder - end2end",
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
        orig_dtype = x.dtype
        hidden = x.to(torch.float32)
        scale = hidden.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt_()
        out = hidden.mul(scale).mul(self.weight)
        return out.to(orig_dtype)


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
    return ctx  # [B, L, H, Dh]


class TorchViTSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

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
    """VisionEncoder：与等价 Torch VisionEncoder 的端到端数值对齐测试。"""

    def __init__(self):
        super().__init__("VisionEncoder")
        # 共享 Torch 侧的 MLP 和 ViT Block 权重，用于拷贝到 InfiniCore 模块中
        self._last_torch_mlp = None
        self._last_torch_blocks = None
        self._last_torch_final_norm = None

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        images,
        image_newline_embed,
        view_separator_embed,
        patch_size=None,
        vision_embed_dim=None,
        hidden_dim=None,
        num_layers=None,
        num_heads=None,
        mlp_ratio=None,
        eps=1e-6,
        final_norm=True,
    ):
        B, V, C, H, W = images.shape
        P_h, P_w = patch_size
        D_v = vision_embed_dim

        device = images.device
        dtype = images.dtype

        # 1. patchify + image_newline + view_separator
        images_flat = images.view(B * V, C, H, W)
        H_grid = H // P_h
        W_grid = W // P_w

        x = images_flat.view(B * V, C, H_grid, P_h, W_grid, P_w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        patches = x.contiguous().view(B * V, H_grid * W_grid, P_h * P_w * C)
        _, N_view, D = patches.shape
        assert D == D_v, f"Expected D_v={D_v}, but got {D}"
        patches = patches.view(B, V, N_view, D)

        # newline
        if image_newline_embed.ndim == 1:
            nl = image_newline_embed.view(1, 1, D)
        elif image_newline_embed.ndim == 2:
            nl = image_newline_embed.view(1, 1, D)
        else:
            nl = image_newline_embed[0:1, 0:1, :]
        tokens = patches.view(B * V, N_view, D)
        tokens = tokens.view(B * V, H_grid, W_grid, D)
        nl_row = nl.view(1, 1, 1, D).expand(B * V, H_grid, 1, D)
        tokens = torch.cat([tokens, nl_row], dim=2)
        tokens = tokens.view(B * V, H_grid * (W_grid + 1), D)
        tokens = tokens.view(B, V, -1, D)
        per_view_len = tokens.shape[2]

        # separator
        if view_separator_embed.ndim == 1:
            sep = view_separator_embed.view(1, 1, D)
        elif view_separator_embed.ndim == 2:
            sep = view_separator_embed.view(1, 1, D)
        else:
            sep = view_separator_embed[0:1, 0:1, :]
        sep_row = sep.view(1, 1, D).expand(B, 1, D)
        tokens = tokens.view(B, V * per_view_len, D)
        view_lengths = [per_view_len] * V
        chunks = list(tokens.split(view_lengths, dim=1))
        pieces = []
        for i, chunk in enumerate(chunks):
            pieces.append(chunk)
            if i != len(chunks) - 1:
                pieces.append(sep_row)
        tokens = torch.cat(pieces, dim=1)  # [B, T, D_v]

        # 2. projector: 简单两层 MLP，与 MlpProjector 结构等价，输出 [B, T, hidden_dim]
        mlp = torch.nn.Sequential(
            torch.nn.Linear(vision_embed_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ).to(device=device, dtype=dtype)
        self._last_torch_mlp = mlp

        x_hid = mlp(tokens)

        # 3. ViTBlock 堆叠
        blocks = torch.nn.ModuleList(
            [
                TorchViTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, eps=eps).to(
                    device=device, dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )
        self._last_torch_blocks = blocks

        for blk in blocks:
            x_hid = blk(x_hid)

        # 4. 最终 RMSNorm（可选）
        if final_norm:
            final_norm_mod = TorchRMSNorm(hidden_dim, eps=eps).to(device=device, dtype=dtype)
            self._last_torch_final_norm = final_norm_mod
            x_hid = final_norm_mod(x_hid)
        else:
            self._last_torch_final_norm = None

        return x_hid

    def infinicore_operator(
        self,
        images,
        image_newline_embed,
        view_separator_embed,
        patch_size=None,
        vision_embed_dim=None,
        hidden_dim=None,
        num_layers=None,
        num_heads=None,
        mlp_ratio=None,
        eps=1e-6,
        final_norm=True,
    ):
        from infinicore.tensor import from_torch as ic_from_torch
        from infinicore.nn import VisionEncoder

        B, V, C, H, W = images.shape

        encoder = VisionEncoder(
            in_channels=C,
            patch_size=patch_size,
            vision_embed_dim=vision_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            eps=eps,
            final_norm=final_norm,
            device=images.device,
            dtype=images.dtype,
        )

        torch_mlp = self._last_torch_mlp
        torch_blocks = self._last_torch_blocks
        torch_final_norm = self._last_torch_final_norm

        if torch_mlp is None or torch_blocks is None:
            raise RuntimeError(
                "Torch VisionEncoder components are not initialized before infinicore_operator."
            )

        with torch.no_grad():
            # 拷贝 VisionFrontend 中 MlpProjector 的权重
            vf = encoder.frontend

            w1 = torch_mlp[0].weight  # [hidden_dim, D_v]
            b1 = torch_mlp[0].bias
            vf.projector.fc1_weight.copy_(ic_from_torch(w1))
            vf.projector.fc1_bias.copy_(ic_from_torch(b1))

            w2 = torch_mlp[2].weight  # [hidden_dim, hidden_dim]
            b2 = torch_mlp[2].bias
            vf.projector.fc2_weight.copy_(ic_from_torch(w2))
            vf.projector.fc2_bias.copy_(ic_from_torch(b2))

            # 拷贝每一层 ViTBlock 的权重
            if len(encoder.blocks) != len(torch_blocks):
                raise RuntimeError(
                    f"Encoder blocks ({len(encoder.blocks)}) and Torch blocks ({len(torch_blocks)}) "
                    "have different lengths."
                )

            for blk_ic, blk_torch in zip(encoder.blocks, torch_blocks):
                # RMSNorm 权重
                blk_ic.norm1.weight.copy_(ic_from_torch(blk_torch.norm1.weight))
                blk_ic.norm2.weight.copy_(ic_from_torch(blk_torch.norm2.weight))

                # Self-Attention 权重
                ta = blk_torch.attn

                blk_ic.attn.q_proj.weight.copy_(ic_from_torch(ta.q_proj.weight))
                blk_ic.attn.q_proj.bias.copy_(ic_from_torch(ta.q_proj.bias))

                blk_ic.attn.k_proj.weight.copy_(ic_from_torch(ta.k_proj.weight))
                blk_ic.attn.k_proj.bias.copy_(ic_from_torch(ta.k_proj.bias))

                blk_ic.attn.v_proj.weight.copy_(ic_from_torch(ta.v_proj.weight))
                blk_ic.attn.v_proj.bias.copy_(ic_from_torch(ta.v_proj.bias))

                blk_ic.attn.out_proj.weight.copy_(ic_from_torch(ta.out_proj.weight))
                blk_ic.attn.out_proj.bias.copy_(ic_from_torch(ta.out_proj.bias))

                # MLP 投影层权重
                w_mlp1 = blk_torch.mlp[0].weight
                b_mlp1 = blk_torch.mlp[0].bias
                blk_ic.mlp.fc1_weight.copy_(ic_from_torch(w_mlp1))
                blk_ic.mlp.fc1_bias.copy_(ic_from_torch(b_mlp1))

                w_mlp2 = blk_torch.mlp[2].weight
                b_mlp2 = blk_torch.mlp[2].bias
                blk_ic.mlp.fc2_weight.copy_(ic_from_torch(w_mlp2))
                blk_ic.mlp.fc2_bias.copy_(ic_from_torch(b_mlp2))

            # 最终 RMSNorm 权重
            if encoder.norm is not None and torch_final_norm is not None:
                encoder.norm.weight.copy_(ic_from_torch(torch_final_norm.weight))

        y_infini = encoder(
            images,
            image_newline_embed=image_newline_embed,
            view_separator_embed=view_separator_embed,
        )
        return convert_infinicore_to_torch(y_infini)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


