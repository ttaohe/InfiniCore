from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def argmax(input: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """
    ArgMax 前端接口，语义对齐 `torch.argmax` 的常用用法：

    - 当 dim 为 None 时：会在展平后的维度上做 ArgMax，相当于 `torch.argmax(input.view(-1))`；
    - 当 dim 为整数时：沿指定维度做 ArgMax，输出 shape 为去掉该维度后的形状；
    - keepdim=True 时，会在返回结果中重新插入长度为 1 的该维度。
    """
    if dim is None:
        # 全局 ArgMax：先展平成 1D，再在 dim=0 上做归约
        # 注意：必须先 contiguous()，否则对于带步长的张量 view 会报错
        flat = input.contiguous().view([input.numel()])
        out = Tensor(_infinicore.argmax(flat._underlying, 0))

        if keepdim:
            # 保留维度：将结果 reshape 为 [1, 1, ..., 1]
            return out.view([1] * input.ndim)

        return out

    # 规范化 dim，与 PyTorch 行为一致
    ndim = input.ndim
    if dim < 0:
        dim = dim + ndim

    out = Tensor(_infinicore.argmax(input._underlying, dim))

    if keepdim:
        # 在 dim 位置插入一个长度为 1 的维度
        out_shape = list(out.shape)
        out_shape.insert(dim, 1)
        return out.view(out_shape)

    return out


