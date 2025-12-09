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


_TEST_CASES_DATA = [
    # (B, H, max_len, D, T, start_pos)
    (1, 2, 8, 4, 3, 0),
    (2, 4, 16, 8, 5, 4),
]

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    tests = []
    for B, H, L_max, D, T, start_pos in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = {"atol": 1e-3, "rtol": 1e-3} if dtype is infinicore.float16 else {"atol": 1e-5, "rtol": 1e-5}

            cache_shape = (B, H, L_max, D)
            new_shape = (B, H, T, D)

            k_cache_spec = TensorSpec.from_tensor(cache_shape, None, dtype)
            v_cache_spec = TensorSpec.from_tensor(cache_shape, None, dtype)
            new_k_spec = TensorSpec.from_tensor(new_shape, None, dtype)
            new_v_spec = TensorSpec.from_tensor(new_shape, None, dtype)

            tests.append(
                TestCase(
                    inputs=[k_cache_spec, v_cache_spec, new_k_spec, new_v_spec],
                    kwargs={"start_pos": start_pos},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="KVCache - UPDATE",
                )
            )

    return tests


def torch_update_kv_cache(k_cache, v_cache, new_k, new_v, start_pos):
    B, H, L_max, D = k_cache.shape
    _, _, T, _ = new_k.shape
    k_out = k_cache.clone()
    v_out = v_cache.clone()
    k_out[:, :, start_pos : start_pos + T, :] = new_k
    v_out[:, :, start_pos : start_pos + T, :] = new_v
    return k_out, v_out


class OpTest(BaseOperatorTest):
    """KV Cache helper tests (Python functional)"""

    def __init__(self):
        super().__init__("KVCache")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, k_cache, v_cache, new_k, new_v, start_pos=0):
        return torch_update_kv_cache(k_cache, v_cache, new_k, new_v, start_pos)

    def infinicore_operator(self, k_cache, v_cache, new_k, new_v, start_pos=0):
        import infinicore.nn.functional as F

        return F.update_kv_cache(k_cache, v_cache, new_k, new_v, start_pos)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()


