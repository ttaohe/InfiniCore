from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def one_hot(indices: Tensor, num_classes: int | None = None) -> Tensor:
    """
    OneHot 前端接口，语义对齐 `torch.nn.functional.one_hot` 的常用用法。

    - indices: 整数张量（建议 dtype=int64），shape 为任意；
    - num_classes: 若为 None，则在 CPU 上根据 `max(indices) + 1` 自动推断。
    """
    nc = -1 if num_classes is None else int(num_classes)
    return Tensor(_infinicore.one_hot(indices._underlying, nc))


