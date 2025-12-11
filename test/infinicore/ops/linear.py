import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    TensorInitializer,
    is_broadcast,
)

import infinicore

# ==============================================================================
# Operator-specific configuration
# ==============================================================================
_TEST_CASES_DATA = [
    # bs, n, in_features, out_features, bias
    (1, 5, 2048, 5632, True, None, None, None),
    (1, 1, 2048, 32000, False, None, None, None),
    (2, 5, 2048, 5632, True, None, None, None),
    (2, 5, 256, 2048, False, None, None, None),
    (None, 5, 256, 2048, False, None, None, None),
    (None, 1, 2048, 5632, True, None, None, None),
    # Extra: 标准小 batch 线性场景 (B=1, T=13, D=32) -> (B=1, T=13, 32)，便于和 ViTBlock 内部对齐
    (1, 13, 32, 32, True, None, None, None),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for linear operation.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        bs = data[0]
        n, in_features, out_features = data[1], data[2], data[3]
        bias = data[4]
        input_strides = data[5] if len(data) > 5 else None
        weight_strides = data[6] if len(data) > 6 else None
        out_strides = data[7] if len(data) > 7 else None

        # Determine shapes based on batch dimension
        if bs is None:
            input_shape = (n, in_features)
            weight_shape = (out_features, in_features)
            out_shape = (n, out_features)
        else:
            input_shape = (bs, n, in_features)
            weight_shape = (out_features, in_features)
            out_shape = (bs, n, out_features)

        if bias is True:
            bias_shape = (out_features,)
        else:
            bias_shape = None

        # Check if tensors support in-place operations
        c_supports_inplace = not is_broadcast(out_shape)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            input_spec = TensorSpec.from_tensor(input_shape, input_strides, dtype)
            weight_spec = TensorSpec.from_tensor(weight_shape, weight_strides, dtype)
            out_spec = TensorSpec.from_tensor(out_shape, out_strides, dtype)

            if bias_shape is not None:
                bias_spec = TensorSpec.from_tensor(bias_shape, None, dtype)
            else:
                bias_spec = None

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[input_spec, weight_spec, bias_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Linear - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (Linear(a, b, out=c))
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, weight_spec, bias_spec],
                        kwargs=None,
                        output_spec=out_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Linear - INPLACE(out)",
                    )
                )

    # Extra case: 专门模拟 VisionEncoder 中 projector.fc1 的大幅度 FP16 场景：
    #   input:  [1, 13, 48] （patch+newline 后的 D_v）
    #   weight: [32, 48]
    #   bias:   [32]
    #
    # 通过 MANUAL + 大尺度初始化构造一个近似于 demo 中的“激活 ~[-3,3], 权重几百, bias 上万”的场景，
    # 用于验证 CPU/NVIDIA 线性核在大幅度 FP16 下的数值稳定性。
    large_bs, large_n, large_in, large_out = 1, 13, 48, 32
    large_dtype = infinicore.float16
    large_tol = _TOLERANCE_MAP.get(large_dtype, {"atol": 0, "rtol": 1e-2})

    # 激活：均值 0、方差 ~1 的随机数，再缩放到 [-3,3] 量级
    large_x_base = torch.randn(large_bs, large_n, large_in, dtype=torch.float32) * 3.0
    # 权重：缩放到 ~400 量级
    large_w_base = torch.randn(large_out, large_in, dtype=torch.float32) * 400.0
    # bias：缩放到几千量级
    large_b_base = torch.randn(large_out, dtype=torch.float32) * 8000.0

    large_input_spec = TensorSpec.from_tensor(
        (large_bs, large_n, large_in),
        None,
        large_dtype,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=large_x_base,
    )
    large_weight_spec = TensorSpec.from_tensor(
        (large_out, large_in),
        None,
        large_dtype,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=large_w_base,
    )
    large_bias_spec = TensorSpec.from_tensor(
        (large_out,),
        None,
        large_dtype,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=large_b_base,
    )

    test_cases.append(
        TestCase(
            inputs=[large_input_spec, large_weight_spec, large_bias_spec],
            kwargs={},
            output_spec=None,
            comparison_target=None,
            tolerance=large_tol,
            description="Linear - LARGE_FP16_PROJECTOR_FC1",
        )
    )

    return test_cases


class OpTest(BaseOperatorTest):
    """Linear operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Linear")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, weight, bias=None, *args, **kwargs):
        """PyTorch linear implementation"""
        # Debug: 小规模 case 打印 shape/stride 以对齐 VisionEncoder/projector 等场景
        if (
            isinstance(input, torch.Tensor)
            and input.dim() == 3
            and input.shape[0] == 1
            and input.shape[1] == 13
            and input.shape[2] in (32, 48)
        ):
            print(
                "[Linear OpTest][torch] input shape:",
                tuple(input.shape),
                "stride:",
                tuple(input.stride()),
                "dtype:",
                input.dtype,
                "device:",
                input.device,
            )
            print(
                "[Linear OpTest][torch] weight shape:",
                tuple(weight.shape),
                "stride:",
                tuple(weight.stride()),
                "dtype:",
                weight.dtype,
                "device:",
                weight.device,
            )
            if bias is not None:
                print(
                    "[Linear OpTest][torch] bias shape:",
                    tuple(bias.shape),
                    "stride:",
                    tuple(bias.stride()),
                    "dtype:",
                    bias.dtype,
                    "device:",
                    bias.device,
                )

        out = torch.nn.functional.linear(input, weight, bias, *args, **kwargs)

        # 对 LARGE_FP16_PROJECTOR_FC1 这类大幅度 FP16 场景，额外打印输出统计信息
        if (
            isinstance(input, torch.Tensor)
            and input.dtype == torch.float16
            and input.dim() == 3
            and input.shape == (1, 13, 48)
        ):
            flat = out.float().view(-1)
            nan = torch.isnan(flat).any().item()
            inf = torch.isinf(flat).any().item()
            print(
                "[Linear OpTest][torch] LARGE_FP16_PROJECTOR_FC1 out stats:",
                "nan=",
                nan,
                "inf=",
                inf,
                "min=",
                float(flat.min()),
                "max=",
                float(flat.max()),
                "mean=",
                float(flat.mean()),
            )

        return out

    def infinicore_operator(self, input, weight, bias=None, *args, **kwargs):
        """InfiniCore linear implementation"""
        from infinicore.utils import to_torch

        # Debug: 转回 torch 检查布局
        input_t = to_torch(input)
        weight_t = to_torch(weight)
        bias_t = to_torch(bias) if bias is not None else None

        if (
            input_t.dim() == 3
            and input_t.shape[0] == 1
            and input_t.shape[1] == 13
            and input_t.shape[2] in (32, 48)
        ):
            print(
                "[Linear OpTest][infini] input shape:",
                tuple(input_t.shape),
                "stride:",
                tuple(input_t.stride()),
                "dtype:",
                input_t.dtype,
                "device:",
                input_t.device,
            )
            print(
                "[Linear OpTest][infini] weight shape:",
                tuple(weight_t.shape),
                "stride:",
                tuple(weight_t.stride()),
                "dtype:",
                weight_t.dtype,
                "device:",
                weight_t.device,
            )
            if bias_t is not None:
                print(
                    "[Linear OpTest][infini] bias shape:",
                    tuple(bias_t.shape),
                    "stride:",
                    tuple(bias_t.stride()),
                    "dtype:",
                    bias_t.dtype,
                    "device:",
                    bias_t.device,
                )

        out_ic = infinicore.nn.functional.linear(input, weight, bias, *args, **kwargs)

        # 对 LARGE_FP16_PROJECTOR_FC1 场景，打印 InfiniCore 输出统计信息
        out_t = to_torch(out_ic)
        if (
            out_t.dim() == 3
            and out_t.shape[0] == 1
            and out_t.shape[1] == 13
            and out_t.shape[2] == 32
            and out_t.dtype == torch.float16
        ):
            flat = out_t.float().view(-1)
            nan = torch.isnan(flat).any().item()
            inf = torch.isinf(flat).any().item()
            print(
                "[Linear OpTest][infini] LARGE_FP16_PROJECTOR_FC1 out stats:",
                "nan=",
                nan,
                "inf=",
                inf,
                "min=",
                float(flat.min()),
                "max=",
                float(flat.max()),
                "mean=",
                float(flat.mean()),
            )

        return out_ic


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
