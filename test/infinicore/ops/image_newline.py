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


# Test cases: (B, H_grid, W_grid, C, newline_shape_type)
# newline_shape_type: 0 -> [C], 1 -> [1,C], 2 -> [1,1,C]
_TEST_CASES_DATA = [
    (1, 2, 3, 4, 0),
    (2, 3, 4, 8, 1),
    (1, 1, 5, 16, 2),
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    tests = []
    for B, H, W, C, newline_type in _TEST_CASES_DATA:
        N = H * W
        tokens_shape = (B, N, C)

        for dtype in _TENSOR_DTYPES:
            tol = {"atol": 1e-3, "rtol": 1e-3} if dtype is infinicore.float16 else {"atol": 1e-5, "rtol": 1e-5}

            tokens_spec = TensorSpec.from_tensor(tokens_shape, None, dtype)

            # newline embedding spec
            if newline_type == 0:
                nl_shape = (C,)
            elif newline_type == 1:
                nl_shape = (1, C)
            else:
                nl_shape = (1, 1, C)

            nl_spec = TensorSpec.from_tensor(nl_shape, None, dtype)

            tests.append(
                TestCase(
                    inputs=[tokens_spec, nl_spec],
                    kwargs={"grid_size": (H, W)},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="ImageNewline - ADD",
                )
            )

    return tests


def torch_add_image_newline(tokens, image_newline_embed, grid_size):
    B, N, C = tokens.shape
    H, W = grid_size
    assert H * W == N

    if image_newline_embed.ndim == 1:
        image_newline_embed = image_newline_embed.view(1, 1, C)
    elif image_newline_embed.ndim == 2:
        image_newline_embed = image_newline_embed.view(1, 1, C)
    elif image_newline_embed.ndim == 3:
        image_newline_embed = image_newline_embed[0:1, 0:1, :]
    else:
        raise ValueError("unsupported newline ndim")

    tokens = tokens.view(B, H, W, C)
    newline_row = image_newline_embed.view(1, 1, 1, C).expand(B, H, 1, C)
    out = torch.cat([tokens, newline_row], dim=2)
    out = out.view(B, H * (W + 1), C)
    return out


class OpTest(BaseOperatorTest):
    """image_newline 拼接 helper 测试"""

    def __init__(self):
        super().__init__("ImageNewline")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, tokens, image_newline_embed, grid_size):
        return torch_add_image_newline(tokens, image_newline_embed, grid_size)

    def infinicore_operator(self, tokens, image_newline_embed, grid_size):
        import infinicore.nn.functional as F

        return F.add_image_newline(tokens, image_newline_embed, grid_size)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


