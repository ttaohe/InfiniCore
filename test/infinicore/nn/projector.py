import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

import infinicore
from infinicore.nn.modules import MlpProjector


_TEST_CASES_DATA = [
    # (B, N, C_in, C_out, hidden)
    (1, 8, 16, 32, 64),
    (2, 4, 32, 16, 32),
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for B, N, C_in, C_out, hidden in _TEST_CASES_DATA:
        shape = (B, N, C_in)
        for dtype in _TENSOR_DTYPES:
            tol = {"atol": 1e-3, "rtol": 1e-3} if dtype is infinicore.float16 else {"atol": 1e-5, "rtol": 1e-5}

            x_spec = TensorSpec.from_tensor(shape, None, dtype)

            test_cases.append(
                TestCase(
                    inputs=[x_spec],
                    kwargs={
                        "C_in": C_in,
                        "C_out": C_out,
                        "hidden": hidden,
                    },
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="MlpProjector - core",
                )
            )
    return test_cases


class OpTest(BaseOperatorTest):
    """MlpProjector 对齐 PyTorch 两层 MLP 的测试。"""

    def __init__(self):
        super().__init__("MlpProjector")
        # 用于在 torch_operator 和 infinicore_operator 之间共享同一份 PyTorch MLP 权重
        self._last_torch_mlp = None

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, C_in, C_out, hidden):
        B, N, _ = x.shape
        device = x.device
        dtype = x.dtype

        mlp = torch.nn.Sequential(
            torch.nn.Linear(C_in, hidden, device=device, dtype=dtype),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, C_out, device=device, dtype=dtype),
        )

        # 记录这次构建的 mlp，infinicore_operator 会用到同一份权重
        self._last_torch_mlp = mlp
        return mlp(x)

    def infinicore_operator(self, x, C_in, C_out, hidden):
        # 构造并初始化 InfiniCore 侧的 MlpProjector
        proj = MlpProjector(
            in_features=C_in,
            out_features=C_out,
            hidden_features=hidden,
            device=x.device,
            dtype=x.dtype,
        )

        # 将上一次 torch_operator 中构建的 torch_mlp 的参数写入 InfiniCore projector，
        # 确保两边权重完全一致。
        torch_mlp = self._last_torch_mlp
        if torch_mlp is None:
            raise RuntimeError("Torch MLP is not initialized before infinicore_operator.")

        with torch.no_grad():
            # 第一层
            w1 = torch_mlp[0].weight  # [hidden, C_in]
            b1 = torch_mlp[0].bias    # [hidden]
            proj.fc1_weight.copy_(infinicore.from_torch(w1))
            proj.fc1_bias.copy_(infinicore.from_torch(b1))

            # 第二层
            w2 = torch_mlp[2].weight  # [C_out, hidden]
            b2 = torch_mlp[2].bias    # [C_out]
            proj.fc2_weight.copy_(infinicore.from_torch(w2))
            proj.fc2_bias.copy_(infinicore.from_torch(b2))

        # 用 InfiniCore projector 计算输出，再转回 torch 做比较
        y_infini = proj(x)
        from framework.utils import convert_infinicore_to_torch

        return convert_infinicore_to_torch(y_infini)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


