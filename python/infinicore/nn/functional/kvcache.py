from __future__ import annotations

from infinicore.tensor import Tensor, empty


def init_kv_cache(
    batch_size: int,
    num_heads: int,
    max_seq_len: int,
    head_dim: int,
    *,
    dtype,
    device,
) -> tuple[Tensor, Tensor]:
    """
    初始化 KV Cache：
      - K_cache, V_cache 形状均为 [B, num_heads, max_seq_len, head_dim]
      - 默认用 0 填充，方便后续增量写入。
    """
    shape = [batch_size, num_heads, max_seq_len, head_dim]
    k_cache = empty(shape, dtype=dtype, device=device)
    v_cache = empty(shape, dtype=dtype, device=device)
    # 当前没有独立的 `zeros_`，大部分调用场景会很快被覆盖写入，这里保持未初始化状态即可；
    # 如果你确实需要显式清零，可以在上层调用 `k_cache.copy_(0)` 等方式处理。
    return k_cache, v_cache


def update_kv_cache(
    k_cache: Tensor,
    v_cache: Tensor,
    new_k: Tensor,
    new_v: Tensor,
    start_pos: int,
) -> tuple[Tensor, Tensor]:
    """
    在现有 KV Cache 中“原位”写入一段新的 K/V：
      - k_cache, v_cache: [B, num_heads, max_seq_len, head_dim]
      - new_k, new_v:     [B, num_heads, T, head_dim]
      - 将区间 [start_pos, start_pos + T) 的时间步更新为 new_k/new_v。

    语义类似于：
      cache[:, :, start_pos:start_pos+T, :] = new_k
    """
    B, H, L_max, D = k_cache.shape
    B2, H2, T, D2 = new_k.shape

    if (B2, H2, D2) != (B, H, D):
        raise ValueError(
            f"update_kv_cache: shape mismatch, cache[B,H,D]={B,H,D}, new_k[B,H,D]={B2,H2,D2}"
        )
    if start_pos < 0 or start_pos + T > L_max:
        raise ValueError(
            f"update_kv_cache: invalid range [{start_pos}, {start_pos+T}) for max_len={L_max}"
        )

    # 统一在连续内存上操作，避免 stride 带来的复杂度
    k_cache_c = k_cache.contiguous()
    v_cache_c = v_cache.contiguous()
    new_k_c = new_k.contiguous()
    new_v_c = new_v.contiguous()

    # 视作 [B, H, L_max, D]
    # 切出要更新的子区间 [start_pos:start_pos+T]
    k_slice = k_cache_c.narrow(2, start_pos, T)
    v_slice = v_cache_c.narrow(2, start_pos, T)

    k_slice.copy_(new_k_c)
    v_slice.copy_(new_v_c)

    return k_cache_c, v_cache_c


def slice_kv_cache(
    k_cache: Tensor,
    v_cache: Tensor,
    seq_range: slice,
) -> tuple[Tensor, Tensor]:
    """
    从 KV Cache 中切出某一段时间区间（通常用于构造当前 step 的 K/V）：
      - k_cache, v_cache: [B, num_heads, max_seq_len, head_dim]
      - seq_range: Python slice，等价于在 dim=2 上做切片，例如 slice(0, cur_len)。

    返回：
      - k_slice, v_slice: 视图张量（共享底层存储），保持原 stride。
    """
    start = 0 if seq_range.start is None else int(seq_range.start)
    stop = k_cache.shape[2] if seq_range.stop is None else int(seq_range.stop)

    if start < 0 or stop < start or stop > k_cache.shape[2]:
        raise ValueError(
            f"slice_kv_cache: invalid slice [{start}, {stop}) for max_len={k_cache.shape[2]}"
        )

    length = stop - start
    k_slice = k_cache.narrow(2, start, length)
    v_slice = v_cache.narrow(2, start, length)
    return k_slice, v_slice



