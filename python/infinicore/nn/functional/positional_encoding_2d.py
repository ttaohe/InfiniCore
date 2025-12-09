from infinicore.tensor import Tensor


def add_2d_positional_encoding(
    x: Tensor,
    pe: Tensor,
    *,
    data_format: str = "BCHW",
) -> Tensor:
    """
    给 2D 特征加位置编码的简易 helper。

    Args:
        x: 输入特征。
            - 若 data_format == "BCHW": 形状 [B, C, H, W]
            - 若 data_format == "BNC":  形状 [B, N, C]，其中 N=H*W，要求 pe 也展平成 [1, N, C]
        pe: 位置编码：
            - 对于 "BCHW"：通常为 [1, C, H, W] 或 [C, H, W]，会在 batch 维上广播；
            - 对于 "BNC"：通常为 [1, N, C] 或 [N, C]。
        data_format: "BCHW" 或 "BNC"。

    Returns:
        与 x 同形状的 Tensor。
    """
    if data_format not in ("BCHW", "BNC"):
        raise ValueError(f"add_2d_positional_encoding: unsupported data_format={data_format}")

    if data_format == "BCHW":
        if x.ndim != 4:
            raise ValueError(f"add_2d_positional_encoding: expect 4D input for BCHW, got ndim={x.ndim}")
        B, C, H, W = x.shape

        # 规范化 pe 形状为 [1, C, H, W]，利用广播加法
        if pe.ndim == 3:
            # [C, H, W] -> [1, C, H, W]
            pe = pe.view([1] + list(pe.shape))
        elif pe.ndim != 4:
            raise ValueError(
                f"add_2d_positional_encoding: expect pe.ndim in (3,4) for BCHW, got {pe.ndim}"
            )

        _, C_pe, H_pe, W_pe = pe.shape
        if (C_pe, H_pe, W_pe) != (C, H, W):
            raise ValueError(
                f"add_2d_positional_encoding: shape mismatch, x={x.shape}, pe={pe.shape}"
            )

        return x + pe

    # data_format == "BNC"
    if x.ndim != 3:
        raise ValueError(f"add_2d_positional_encoding: expect 3D input for BNC, got ndim={x.ndim}")
    B, N, C = x.shape

    if pe.ndim == 2:
        # [N, C] -> [1, N, C]
        pe = pe.view([1, N, C])
    elif pe.ndim != 3:
        raise ValueError(
            f"add_2d_positional_encoding: expect pe.ndim in (2,3) for BNC, got {pe.ndim}"
        )

    _, N_pe, C_pe = pe.shape
    if (N_pe, C_pe) != (N, C):
        raise ValueError(
            f"add_2d_positional_encoding: shape mismatch, x={x.shape}, pe={pe.shape}"
        )

    return x + pe



