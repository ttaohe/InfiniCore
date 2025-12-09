from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def scatter_add(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    `scatter_add` 前端接口，尽量对齐 `torch.scatter_add` 的常见用法。

    当前实现（CPU）语义：
      - 如果 out is None:
          返回一个新的张量 out，满足 out = input.scatter_add(dim, index, src)
          （input 自身不会被修改）
      - 如果 out 不为 None:
          将结果写入 out，并返回 out。
    """
    if out is None:
        return Tensor(
            _infinicore.scatter_add(
                input._underlying,
                dim,
                index._underlying,
                src._underlying,
            )
        )

    _infinicore.scatter_add_(
        out._underlying,
        input._underlying,
        dim,
        index._underlying,
        src._underlying,
    )
    return out


