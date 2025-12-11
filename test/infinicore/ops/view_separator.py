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


# Test cases: (B, view_lengths, C, sep_shape_type)
# sep_shape_type: 0 -> [C], 1 -> [1,C], 2 -> [1,1,C]
_TEST_CASES_DATA = [
    (1, [4, 3], 8, 0),
    (2, [2, 2, 3], 16, 1),
    (1, [5], 4, 2),  # 单视图，不应插入 separator
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    tests = []
    for B, view_lengths, C, sep_type in _TEST_CASES_DATA:
        N = sum(view_lengths)
        tokens_shape = (B, N, C)

        for dtype in _TENSOR_DTYPES:
            tol = {"atol": 1e-3, "rtol": 1e-3} if dtype is infinicore.float16 else {"atol": 1e-5, "rtol": 1e-5}

            tokens_spec = TensorSpec.from_tensor(tokens_shape, None, dtype)

            if sep_type == 0:
                sep_shape = (C,)
            elif sep_type == 1:
                sep_shape = (1, C)
            else:
                sep_shape = (1, 1, C)

            sep_spec = TensorSpec.from_tensor(sep_shape, None, dtype)

            tests.append(
                TestCase(
                    inputs=[tokens_spec, sep_spec],
                    kwargs={"view_lengths": view_lengths},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="ViewSeparator - ADD",
                )
            )

    return tests


def torch_add_view_separator(tokens, view_lengths, sep_embed):
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


class OpTest(BaseOperatorTest):
    """view_separator 拼接 helper 测试"""

    def __init__(self):
        super().__init__("ViewSeparator")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, tokens, sep_embed, view_lengths):
        return torch_add_view_separator(tokens, view_lengths, sep_embed)

    def infinicore_operator(self, tokens, sep_embed, view_lengths):
        import infinicore.nn.functional as F

        return F.add_view_separator(tokens, view_lengths, sep_embed)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


