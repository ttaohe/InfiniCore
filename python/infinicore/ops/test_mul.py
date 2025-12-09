from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def test_mul(a: Tensor, b: Tensor, *, out: Tensor | None = None) -> Tensor:
    """
    简单示例算子：向量点积 test_mul。

    语义：给定两个同长度的一维向量 a, b，计算
          out = sum_i a[i] * b[i]

    当前实现仅支持：
      - 设备：CPU
      - dtype：float32
      - 连续一维张量
    """
    if out is None:
        return Tensor(_infinicore.test_mul(a._underlying, b._underlying))

    _infinicore.test_mul_(out._underlying, a._underlying, b._underlying)
    return out


