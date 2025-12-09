from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _sum_single_dim(input: Tensor, dim: int, keepdim: bool) -> Tensor:
    ndim = input.ndim
    if dim < 0:
        dim = dim + ndim
    return Tensor(_infinicore.sum(input._underlying, dim, keepdim))


def sum(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    """
    Sum reduction,对齐常用的 `torch.sum` 用法:
      - dim=None: 全局求和，返回标量；
      - dim=int: 沿指定维度求和；
      - dim=tuple: 按 dim 中的多个维度依次求和。

    当前实现暂不支持 dtype 参数重定类型。
    """
    # dim=None: flatten 后在 dim=0 上做一次 sum
    if dim is None:
        flat = input.contiguous().view([input.numel()])
        return _sum_single_dim(flat, 0, False)

    # dim 为 tuple: 按照从大到小的顺序逐个 reduce，避免维度重新编号问题
    if isinstance(dim, (tuple, list)):
        dims = [d if d >= 0 else input.ndim + d for d in dim]
        # 多维归约时，PyTorch 在 keepdim=False 下对所有 dim 一次性归约；
        # 顺序不影响最终结果，因此我们采用降序顺序依次 reduce。
        dims = sorted(dims, reverse=True)

        result = input
        for i, d in enumerate(dims):
            # 对于多维归约，最后一次使用调用者传入的 keepdim，其余次统一使用 keepdim=False
            kd = keepdim if i == len(dims) - 1 else False
            result = _sum_single_dim(result, d, kd)
        return result

    # 单一维度
    return _sum_single_dim(input, dim, keepdim)


