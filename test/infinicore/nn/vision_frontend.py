import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
)


class OpTest(BaseOperatorTest):
    """验证 VisionFrontend：与等价的 Torch 实现数值对齐。"""

    def __init__(self):
        super().__init__("VisionFrontend")
        # 在 torch_operator / infinicore_operator 之间共享同一份 Torch MLP 权重
        self._last_torch_mlp = None

    def get_test_cases(self):
        tests = []

        # B, V, C, H, W, patch_size, D_v, hidden
        configs = [
            (1, 1, 3, 8, 8, (4, 4), 3 * 4 * 4, 32),
            (2, 2, 3, 8, 8, (4, 4), 3 * 4 * 4, 64),
        ]

        dtypes = [infinicore.float16, infinicore.float32]

        for B, V, C, H, W, patch_size, D_v, hidden in configs:
            images_shape = (B, V, C, H, W)
            for dtype in dtypes:
                tol = (
                    {"atol": 1e-3, "rtol": 1e-3}
                    if dtype is infinicore.float16
                    else {"atol": 1e-5, "rtol": 1e-5}
                )

                images_spec = TensorSpec.from_tensor(images_shape, None, dtype)
                nl_spec = TensorSpec.from_tensor((D_v,), None, dtype)
                sep_spec = TensorSpec.from_tensor((D_v,), None, dtype)

                tests.append(
                    TestCase(
                        inputs=[images_spec, nl_spec, sep_spec],
                        kwargs={
                            "patch_size": patch_size,
                            "vision_embed_dim": D_v,
                            "hidden_dim": hidden,
                        },
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tol,
                        description="VisionFrontend - core",
                    )
                )

        return tests

    def torch_operator(
        self,
        images,
        image_newline_embed,
        view_separator_embed,
        patch_size=None,
        vision_embed_dim=None,
        hidden_dim=None,
    ):
        B, V, C, H, W = images.shape
        P_h, P_w = patch_size

        # 参考实现：直接用 torch patchify + newline + separator + torch MLP
        images_flat = images.view(B * V, C, H, W)
        H_grid = H // P_h
        W_grid = W // P_w

        x = images_flat.view(B * V, C, H_grid, P_h, W_grid, P_w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        patches = x.contiguous().view(B * V, H_grid * W_grid, P_h * P_w * C)
        _, N_view, D = patches.shape
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
        tokens = torch.cat(pieces, dim=1)  # [B, T, D]

        # projector: 简单两层 MLP，与 MlpProjector 结构等价
        mlp = torch.nn.Sequential(
            torch.nn.Linear(vision_embed_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ).to(tokens.device, dtype=tokens.dtype)

        # 记录 MLP，用于在 infinicore_operator 中对齐权重
        self._last_torch_mlp = mlp
        out = mlp(tokens)
        return out

    def infinicore_operator(
        self,
        images,
        image_newline_embed,
        view_separator_embed,
        patch_size=None,
        vision_embed_dim=None,
        hidden_dim=None,
    ):
        from infinicore.nn import VisionFrontend
        from framework.utils import convert_infinicore_to_torch

        B, V, C, H, W = images.shape

        # 构建 InfiniCore 侧 VisionFrontend
        vf = VisionFrontend(
            in_channels=C,
            patch_size=patch_size,
            vision_embed_dim=vision_embed_dim,
            hidden_dim=hidden_dim,
            device=images.device,
            dtype=images.dtype,
        )

        # 将 Torch MLP 的权重拷贝到 vf.projector 中，确保两边参数完全一致
        torch_mlp = self._last_torch_mlp
        if torch_mlp is None:
            raise RuntimeError(
                "Torch MLP is not initialized before infinicore_operator."
            )

        with torch.no_grad():
            # 第一层 Linear
            w1 = torch_mlp[0].weight  # [hidden_dim, D_v]
            b1 = torch_mlp[0].bias    # [hidden_dim]
            vf.projector.fc1_weight.copy_(infinicore.from_torch(w1))
            vf.projector.fc1_bias.copy_(infinicore.from_torch(b1))

            # 第二层 Linear
            w2 = torch_mlp[2].weight  # [hidden_dim, hidden_dim]
            b2 = torch_mlp[2].bias    # [hidden_dim]
            vf.projector.fc2_weight.copy_(infinicore.from_torch(w2))
            vf.projector.fc2_bias.copy_(infinicore.from_torch(b2))

        # 前向计算并转回 Torch 进行数值对比
        y_infini = vf(
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


