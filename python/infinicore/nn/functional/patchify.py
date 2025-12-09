from infinicore.tensor import Tensor


def _to_2tuple(val):
    if isinstance(val, (tuple, list)):
        assert len(val) == 2
        return int(val[0]), int(val[1])
    v = int(val)
    return v, v


def patchify(input: Tensor, patch_size) -> Tensor:
    """
    将 NCHW 图像张量切成 patch 序列，语义等价于常见 ViT 的 patchify：

      - 输入:  input[B, C, H, W]
      - 输出:  patches[B, N, P*P*C]
        其中 N = (H / P_h) * (W / P_w)，P = P_h * P_w，patch_size 可以是 int 或 (P_h, P_w)。

    要求:
      - H % P_h == 0 且 W % P_w == 0
      - 当前只支持 NCHW 布局
    """
    if input.ndim != 4:
        raise ValueError(f"patchify: expect 4D NCHW input, got ndim={input.ndim}")

    # 先保证是连续内存，再做 view/permute
    x = input.contiguous()
    B, C, H, W = x.shape
    P_h, P_w = _to_2tuple(patch_size)

    if H % P_h != 0 or W % P_w != 0:
        raise ValueError(
            f"patchify: H={H}, W={W} must be divisible by patch_size={P_h}x{P_w}"
        )

    H_grid = H // P_h
    W_grid = W // P_w

    # 按 ViT 标准变换：
    # [B, C, H, W]
    # -> [B, C, H_grid, P_h, W_grid, P_w]
    # -> [B, H_grid, W_grid, P_h, P_w, C]
    # -> [B, H_grid*W_grid, P_h*P_w*C]
    x = x.view([B, C, H_grid, P_h, W_grid, P_w])
    x = x.permute([0, 2, 4, 3, 5, 1])
    # permute 之后张量通常是非连续的，先 contiguous 再做最终的 view
    patches = x.contiguous().view([B, H_grid * W_grid, P_h * P_w * C])
    return patches



