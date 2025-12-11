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


# 简单测试多视图 + image_newline + view_separator 的完整 token 流。
#
# 这里只覆盖结构与维度是否正确、以及与等价的 PyTorch 实现是否一致；
# projector / 2D PE 在各自单测中已经验证，这里主要关注「视觉 token 排布」本身。


def _torch_patchify(images, patch_size):
    # images: [B*V, C, H, W]
    from math import prod

    Bv, C, H, W = images.shape
    P_h, P_w = patch_size
    H_grid = H // P_h
    W_grid = W // P_w
    x = images.view(Bv, C, H_grid, P_h, W_grid, P_w)
    x = x.permute(0, 2, 4, 3, 5, 1)  # [Bv, H_grid, W_grid, P_h, P_w, C]
    patches = x.contiguous().view(Bv, H_grid * W_grid, P_h * P_w * C)
    return patches, H_grid, W_grid


def _torch_add_image_newline(tokens, grid_size, newline_embed):
    B, N, C = tokens.shape
    H_grid, W_grid = grid_size
    assert H_grid * W_grid == N

    if newline_embed.ndim == 1:
        newline_embed = newline_embed.view(1, 1, C)
    elif newline_embed.ndim == 2:
        newline_embed = newline_embed.view(1, 1, C)
    elif newline_embed.ndim == 3:
        newline_embed = newline_embed[0:1, 0:1, :]
    else:
        raise ValueError("unsupported newline_embed ndim")

    tokens = tokens.view(B, H_grid, W_grid, C)
    newline_row = newline_embed.view(1, 1, 1, C).expand(B, H_grid, 1, C)
    out = torch.cat([tokens, newline_row], dim=2)
    out = out.view(B, H_grid * (W_grid + 1), C)
    return out


def _torch_add_view_separator(tokens, view_lengths, sep_embed):
    B, N, C = tokens.shape
    assert sum(view_lengths) == N

    if sep_embed.ndim == 1:
        sep_embed = sep_embed.view(1, 1, C)
    elif sep_embed.ndim == 2:
        sep_embed = sep_embed.view(1, 1, C)
    elif sep_embed.ndim == 3:
        sep_embed = sep_embed[0:1, 0:1, :]
    else:
        raise ValueError("unsupported sep_embed ndim")

    sep_row = sep_embed.view(1, 1, C).expand(B, 1, C)
    chunks = list(tokens.split(view_lengths, dim=1))
    pieces = []
    for i, chunk in enumerate(chunks):
        pieces.append(chunk)
        if i != len(chunks) - 1:
            pieces.append(sep_row)
    return torch.cat(pieces, dim=1)


def parse_test_cases():
    tests = []

    # B, V, C, H, W, patch_size, use_newline, use_sep
    configs = [
        (1, 1, 3, 8, 8, (4, 4), True, False),
        (2, 2, 3, 8, 8, (4, 4), True, True),
        (1, 3, 2, 4, 4, (2, 2), False, True),
    ]

    dtypes = [infinicore.float16, infinicore.float32]

    for B, V, C, H, W, patch_size, use_nl, use_sep in configs:
        images_shape = (B, V, C, H, W)
        for dtype in dtypes:
            tol = (
                {"atol": 1e-3, "rtol": 1e-3}
                if dtype is infinicore.float16
                else {"atol": 1e-5, "rtol": 1e-5}
            )

            images_spec = TensorSpec.from_tensor(images_shape, None, dtype)

            # newline / separator embedding 维度等于 patch embedding 维度
            P_h, P_w = patch_size
            D = P_h * P_w * C
            nl_spec = (
                TensorSpec.from_tensor((D,), None, dtype) if use_nl else None
            )
            sep_spec = (
                TensorSpec.from_tensor((D,), None, dtype) if use_sep else None
            )

            inputs = [images_spec]
            if nl_spec is not None:
                inputs.append(nl_spec)
            if sep_spec is not None:
                inputs.append(sep_spec)

            kwargs = {
                "patch_size": patch_size,
            }
            kwargs["use_newline"] = use_nl
            kwargs["use_separator"] = use_sep

            tests.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="VisionTokens - full flow",
                )
            )

    return tests


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("VisionTokens")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        images,
        *extra,
        patch_size=None,
        use_newline=False,
        use_separator=False,
    ):
        # images: [B, V, C, H, W]
        B, V, C, H, W = images.shape
        P_h, P_w = patch_size

        images_flat = images.view(B * V, C, H, W)
        patches, H_grid, W_grid = _torch_patchify(images_flat, patch_size)
        _, N_view, D = patches.shape
        patches = patches.view(B, V, N_view, D)

        idx = 0
        newline_embed = None
        sep_embed = None
        if use_newline:
            newline_embed = extra[idx]
            idx += 1
        if use_separator:
            sep_embed = extra[idx]
            idx += 1

        if use_newline:
            patches = patches.view(B * V, N_view, D)
            patches = _torch_add_image_newline(
                patches, (H_grid, W_grid), newline_embed
            )
            _, N_with_nl, _ = patches.shape
            patches = patches.view(B, V, N_with_nl, D)
            per_view_len = N_with_nl
        else:
            per_view_len = N_view

        tokens = patches.view(B, V * per_view_len, D)

        if use_separator and V > 1:
            view_lengths = [per_view_len] * V
            tokens = _torch_add_view_separator(tokens, view_lengths, sep_embed)

        return tokens

    def infinicore_operator(
        self,
        images,
        *extra,
        patch_size=None,
        use_newline=False,
        use_separator=False,
    ):
        import infinicore.nn.functional as F

        idx = 0
        image_newline_embed = None
        view_separator_embed = None
        if use_newline:
            image_newline_embed = extra[idx]
            idx += 1
        if use_separator:
            view_separator_embed = extra[idx]
            idx += 1

        return F.build_vision_tokens(
            images,
            patch_size,
            image_newline_embed=image_newline_embed,
            view_separator_embed=view_separator_embed,
        )


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


