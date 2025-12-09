import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def sigmoid(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    """
    Element-wise Sigmoid activation, 类似 `torch.sigmoid` 的常用用法。
      - 默认返回一个新的 Tensor；
      - 支持 `inplace=True` 原地写回；
      - 或通过 `out` 显式指定输出张量。
    """
    # 如果启用了 ntops 并且是 GPU 设备，直接走 ntops 的 torch.sigmoid
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.sigmoid(input, inplace=inplace)

    # inplace: 直接在 input 上做原地更新
    if inplace:
        _infinicore.sigmoid_(input._underlying, input._underlying)
        return input

    # out is None: 正常 out-of-place 返回新 Tensor
    if out is None:
        return Tensor(_infinicore.sigmoid(input._underlying))

    # 显式 out：写入到 out 中并返回 out
    _infinicore.sigmoid_(out._underlying, input._underlying)
    return out



