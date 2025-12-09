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

# ==============================================================================
# Operator-specific configuration for test_mul
# ==============================================================================

# 仅测试简单的一维向量点积场景
_TEST_CASES_DATA = [
    # (shape,)
    ((8,),),
    ((32,),),
    ((1024,),),
]

# test_mul 示例 kernel 只实现了 float32
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    """
    构造 test_mul 的测例列表。
    这里为了简单，只构造连续一维向量的 out-of-place 调用。
    """
    test_cases: list[TestCase] = []

    for data in _TEST_CASES_DATA:
        shape = data[0]

        for dtype in _TENSOR_DTYPES:
            a_spec = TensorSpec.from_tensor(shape, None, dtype, name="a")
            b_spec = TensorSpec.from_tensor(shape, None, dtype, name="b")

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance={"atol": 0.0, "rtol": 1e-5},
                    description=f"test_mul - 1D dot product (shape={shape}, dtype={dtype})",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Example operator test for test_mul (vector dot product)."""

    def __init__(self):
        super().__init__("TestMul")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch reference implementation using torch.dot."""
        assert len(args) == 2
        return torch.dot(args[0].view(-1), args[1].view(-1))

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore implementation."""
        assert len(args) == 2
        return infinicore.test_mul(*args, **kwargs)


def main():
    """Main entry point."""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


