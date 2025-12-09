from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def topk(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
):
    """
    TopK 前端接口，语义对齐 `torch.topk` 的常见用法。

    当前实现：
      - 仅支持 CPU；
      - 支持 float16/bfloat16/float32；
      - 返回 (values, indices)，其中 indices 的 dtype 为 int64。
    """
    values, indices = _infinicore.topk(
        input._underlying,
        k,
        dim,
        largest,
        sorted,
    )
    return Tensor(values), Tensor(indices)


