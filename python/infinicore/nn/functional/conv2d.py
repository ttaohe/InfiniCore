import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _to_2tuple(val):
    if isinstance(val, (tuple, list)):
        assert len(val) == 2
        return int(val[0]), int(val[1])
    v = int(val)
    return v, v


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> Tensor:
    """
    2D 卷积，接口尽量对齐 `torch.nn.functional.conv2d` 的常见用法：
      - input:  [N, C_in, H, W]
      - weight: [C_out, C_in/groups, K_h, K_w]
      - bias:   [C_out] or None
      - stride/padding/dilation: int 或 2-tuple
      - groups: 目前仅支持 1
    """
    if groups != 1:
        raise NotImplementedError("infinicore.nn.functional.conv2d 当前仅支持 groups=1")

    stride_h, stride_w = _to_2tuple(stride)
    pad_h, pad_w = _to_2tuple(padding)
    dil_h, dil_w = _to_2tuple(dilation)

    bias_underlying = None if bias is None else bias._underlying

    return Tensor(
        _infinicore.conv2d(
            input._underlying,
            weight._underlying,
            bias_underlying,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            groups,
        )
    )



