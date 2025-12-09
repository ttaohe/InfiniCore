from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _mean_single_dim(input: Tensor, dim: int, keepdim: bool) -> Tensor:
    ndim = input.ndim
    if dim < 0:
        dim = dim + ndim
    return Tensor(_infinicore.mean(input._underlying, dim, keepdim))


def _mean_no_out(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    """内部实现：不处理 out，只返回结果 Tensor。"""
    # dim=None: flatten 后在 dim=0 上做一次 mean
    if dim is None:
        flat = input.contiguous().view([input.numel()])
        return _mean_single_dim(flat, 0, False)

    # dim 为 tuple: 按照从大到小的顺序逐个 reduce
    if isinstance(dim, (tuple, list)):
        dims = [d if d >= 0 else input.ndim + d for d in dim]
        dims = sorted(dims, reverse=True)

        result = input
        for i, d in enumerate(dims):
            kd = keepdim if i == len(dims) - 1 else False
            result = _mean_single_dim(result, d, kd)
        return result

    # 单一维度
    return _mean_single_dim(input, dim, keepdim)


def mean(input: Tensor, dim=None, keepdim: bool = False, *, out: Tensor | None = None) -> Tensor:
    """
    Mean reduction, 对齐常用的 `torch.mean` 用法（不支持 dtype 参数）:
      - dim=None: 全局平均，返回标量；
      - dim=int: 沿指定维度求平均；
      - dim=tuple: 按 dim 中的多个维度依次求平均；
      - out: 若提供，则将结果写入 out（支持 dim=None/int/tuple）。
    """
    if out is None:
        return _mean_no_out(input, dim, keepdim)

    # 有 out 时，先通过通用路径算出结果，再 copy_ 到 out 中
    result = _mean_no_out(input, dim, keepdim)
    out.copy_(result)
    return out


