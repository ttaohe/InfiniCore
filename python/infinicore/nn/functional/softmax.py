from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """
    通用 softmax 接口，语义对齐 `torch.nn.functional.softmax(input, dim=dim)`。
    当前实现依赖 InfiniOP softmax，支持 float16/bfloat16/float32。
    """
    ndim = input.ndim
    if dim < 0:
        dim = dim + ndim
    return Tensor(_infinicore.softmax(input._underlying, dim))


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    """
    通用 log_softmax 接口，语义对齐 `torch.nn.functional.log_softmax(input, dim=dim)`。
    当前实现依赖 InfiniOP logsoftmax，支持 float16/bfloat16/float32。
    """
    ndim = input.ndim
    if dim < 0:
        dim = dim + ndim
    return Tensor(_infinicore.log_softmax(input._underlying, dim))


