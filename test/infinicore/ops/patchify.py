import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
)


# Test cases: (in_shape, patch_size)
_TEST_CASES_DATA = [
    ((1, 3, 16, 16), 4),
    ((2, 4, 32, 32), (4, 4)),
    ((1, 8, 24, 32), (4, 8)),
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    tests = []
    for in_shape, patch_size in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = {"atol": 1e-3, "rtol": 1e-3} if dtype is infinicore.float16 else {"atol": 1e-5, "rtol": 1e-5}
            in_spec = TensorSpec.from_tensor(in_shape, None, dtype)

            tests.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"patch_size": patch_size},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Patchify - OUT_OF_PLACE",
                )
            )
    return tests


def _to_2tuple(val):
    if isinstance(val, (tuple, list)):
        assert len(val) == 2
        return int(val[0]), int(val[1])
    v = int(val)
    return v, v


def torch_patchify(x, patch_size):
    B, C, H, W = x.shape
    P_h, P_w = _to_2tuple(patch_size)
    assert H % P_h == 0 and W % P_w == 0
    H_grid = H // P_h
    W_grid = W // P_w
    x = x.view(B, C, H_grid, P_h, W_grid, P_w)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, H_grid * W_grid, P_h * P_w * C)


class OpTest(BaseOperatorTest):
    """Patchify operator test (Python functional)"""

    def __init__(self):
        super().__init__("Patchify")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, patch_size=None):
        return torch_patchify(input, patch_size)

    def infinicore_operator(self, input, patch_size=None):
        import infinicore.nn.functional as F

        return F.patchify(input, patch_size)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


