import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import torch
import torch.nn as nn
import infinicore
from infinicore.utils import to_torch
from infinicore.nn.modules import VisionEncoder, ViTBlock
from infinicore.nn import functional as F
from infinicore.tensor import from_torch as ic_from_torch


"""
一个简单的 Vision→Language 串联 Demo：

- 随机生成一批图像 `images: [B, V, C, H, W]`
- 用 `VisionEncoder` 编码成视觉 token：`vision_tokens: [B, T_v, hidden_dim]`
- 构造一段假想的文本 embedding：`text_tokens: [B, T_t, hidden_dim]`
- 在序列维上拼接成 `[vision | text]`，得到 `joint_tokens: [B, T_v+T_t, hidden_dim]`
- 通过一层 `ViTBlock`（内部使用 MLA Self-Attention + MLP）做联合建模

运行方式：
  python test/infinicore/nn/vision_language_demo.py
"""


class TorchRMSNorm(torch.nn.Module):
    """与 InfiniCore RMSNorm 行为对齐的 Torch 版本（用于权重对齐）."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 与 InfiniCore RMSNorm 行为对齐：权重初始为 1
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        hidden = x.to(torch.float32)
        scale = hidden.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt_()
        out = hidden.mul(scale).mul(self.weight)
        return out.to(orig_dtype)


def torch_mla_core(q, k, v):
    # q/k/v: [B, L, H, Dh]
    B, L, H, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)

    q_2d = q.reshape(B * H, L, Dh)
    k_2d = k.reshape(B * H, L, Dh)
    v_2d = v.reshape(B * H, L, Dh)

    scores = torch.matmul(q_2d, k_2d.transpose(-1, -2)) * scale  # [BH, L, L]
    attn = torch.softmax(scores, dim=-1)
    ctx_2d = torch.matmul(attn, v_2d)  # [BH, L, Dh]
    ctx = ctx_2d.reshape(B, H, L, Dh).permute(0, 2, 1, 3).contiguous()
    return ctx  # [B, L, H, Dh]


class TorchViTSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        Dh = self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, H, Dh)
        k = k.view(B, N, H, Dh)
        v = v.view(B, N, H, Dh)

        ctx = torch_mla_core(q, k, v)  # [B, N, H, Dh]
        ctx = ctx.view(B, N, C)
        out = self.out_proj(ctx)
        return out


class TorchViTBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = TorchRMSNorm(embed_dim, eps=eps)
        self.attn = TorchViTSelfAttention(embed_dim, num_heads)
        self.norm2 = TorchRMSNorm(embed_dim, eps=eps)

        hidden_dim = int(math.ceil(embed_dim * mlp_ratio))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _print_stats(name: str, x_t: torch.Tensor):
    nan = torch.isnan(x_t).any().item()
    inf = torch.isinf(x_t).any().item()
    x_flat = x_t.float().view(-1)
    min_val = x_flat.min().item()
    max_val = x_flat.max().item()
    mean_val = x_flat.mean().item()
    print(
        f"{name}: nan={nan}, inf={inf}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}"
    )


def build_torch_vision_encoder(
    images: torch.Tensor,
    image_newline_embed: torch.Tensor,
    view_separator_embed: torch.Tensor,
    patch_size,
    vision_embed_dim,
    hidden_dim,
    num_layers,
    num_heads,
    mlp_ratio,
    eps: float = 1e-6,
    final_norm: bool = True,
):
    """完全复用 vision_encoder 单测里的 Torch 端实现逻辑，用于生成权重."""
    B, V, C, H, W = images.shape
    P_h, P_w = patch_size
    D_v = vision_embed_dim

    device = images.device
    dtype = images.dtype

    # 1. patchify + image_newline + view_separator
    images_flat = images.view(B * V, C, H, W)
    H_grid = H // P_h
    W_grid = W // P_w

    x = images_flat.view(B * V, C, H_grid, P_h, W_grid, P_w)
    x = x.permute(0, 2, 4, 3, 5, 1)
    patches = x.contiguous().view(B * V, H_grid * W_grid, P_h * P_w * C)
    _, N_view, D = patches.shape
    assert D == D_v, f"Expected D_v={D_v}, but got {D}"
    patches = patches.view(B, V, N_view, D)

    # newline
    if image_newline_embed.ndim == 1:
        nl = image_newline_embed.view(1, 1, D)
    elif image_newline_embed.ndim == 2:
        nl = image_newline_embed.view(1, 1, D)
    else:
        nl = image_newline_embed[0:1, 0:1, :]
    tokens = patches.view(B * V, N_view, D)
    tokens = tokens.view(B * V, H_grid, W_grid, D)
    nl_row = nl.view(1, 1, 1, D).expand(B * V, H_grid, 1, D)
    tokens = torch.cat([tokens, nl_row], dim=2)
    tokens = tokens.view(B * V, H_grid * (W_grid + 1), D)
    tokens = tokens.view(B, V, -1, D)
    per_view_len = tokens.shape[2]

    # separator
    if view_separator_embed.ndim == 1:
        sep = view_separator_embed.view(1, 1, D)
    elif view_separator_embed.ndim == 2:
        sep = view_separator_embed.view(1, 1, D)
    else:
        sep = view_separator_embed[0:1, 0:1, :]
    sep_row = sep.view(1, 1, D).expand(B, 1, D)
    tokens = tokens.view(B, V * per_view_len, D)
    view_lengths = [per_view_len] * V
    chunks = list(tokens.split(view_lengths, dim=1))
    pieces = []
    for i, chunk in enumerate(chunks):
        pieces.append(chunk)
        if i != len(chunks) - 1:
            pieces.append(sep_row)
    tokens = torch.cat(pieces, dim=1)  # [B, T, D_v]

    # 2. projector: 简单两层 MLP，与 MlpProjector 结构等价，输出 [B, T, hidden_dim]
    mlp = torch.nn.Sequential(
        torch.nn.Linear(vision_embed_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
    ).to(device=device, dtype=dtype)

    x_hid = mlp(tokens)

    # 3. ViTBlock 堆叠
    blocks = torch.nn.ModuleList(
        [
            TorchViTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, eps=eps).to(
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ]
    )

    for blk in blocks:
        x_hid = blk(x_hid)

    # 4. 最终 RMSNorm（可选）
    if final_norm:
        final_norm_mod = TorchRMSNorm(hidden_dim, eps=eps).to(device=device, dtype=dtype)
        x_hid = final_norm_mod(x_hid)
    else:
        final_norm_mod = None

    return x_hid, mlp, blocks, final_norm_mod


def copy_vision_encoder_weights(
    encoder: VisionEncoder,
    torch_mlp: torch.nn.Sequential,
    torch_blocks: torch.nn.ModuleList,
    torch_final_norm: torch.nn.Module | None,
):
    """把 Torch 版 VisionEncoder 里的各层权重拷贝到 InfiniCore VisionEncoder."""
    with torch.no_grad():
        vf = encoder.frontend

        # MlpProjector
        w1 = torch_mlp[0].weight  # [hidden_dim, D_v]
        b1 = torch_mlp[0].bias
        vf.projector.fc1_weight.copy_(ic_from_torch(w1))
        vf.projector.fc1_bias.copy_(ic_from_torch(b1))

        w2 = torch_mlp[2].weight  # [hidden_dim, hidden_dim]
        b2 = torch_mlp[2].bias
        vf.projector.fc2_weight.copy_(ic_from_torch(w2))
        vf.projector.fc2_bias.copy_(ic_from_torch(b2))

        # ViTBlock 堆叠
        if len(encoder.blocks) != len(torch_blocks):
            raise RuntimeError(
                f"Encoder blocks ({len(encoder.blocks)}) and Torch blocks ({len(torch_blocks)}) "
                "have different lengths."
            )

        for blk_ic, blk_torch in zip(encoder.blocks, torch_blocks):
            # RMSNorm
            blk_ic.norm1.weight.copy_(ic_from_torch(blk_torch.norm1.weight))
            blk_ic.norm2.weight.copy_(ic_from_torch(blk_torch.norm2.weight))

            # Self-Attention
            ta = blk_torch.attn

            blk_ic.attn.q_proj.weight.copy_(ic_from_torch(ta.q_proj.weight))
            blk_ic.attn.q_proj.bias.copy_(ic_from_torch(ta.q_proj.bias))

            blk_ic.attn.k_proj.weight.copy_(ic_from_torch(ta.k_proj.weight))
            blk_ic.attn.k_proj.bias.copy_(ic_from_torch(ta.k_proj.bias))

            blk_ic.attn.v_proj.weight.copy_(ic_from_torch(ta.v_proj.weight))
            blk_ic.attn.v_proj.bias.copy_(ic_from_torch(ta.v_proj.bias))

            blk_ic.attn.out_proj.weight.copy_(ic_from_torch(ta.out_proj.weight))
            blk_ic.attn.out_proj.bias.copy_(ic_from_torch(ta.out_proj.bias))

            # MLP
            w_mlp1 = blk_torch.mlp[0].weight
            b_mlp1 = blk_torch.mlp[0].bias
            blk_ic.mlp.fc1_weight.copy_(ic_from_torch(w_mlp1))
            blk_ic.mlp.fc1_bias.copy_(ic_from_torch(b_mlp1))

            w_mlp2 = blk_torch.mlp[2].weight
            b_mlp2 = blk_torch.mlp[2].bias
            blk_ic.mlp.fc2_weight.copy_(ic_from_torch(w_mlp2))
            blk_ic.mlp.fc2_bias.copy_(ic_from_torch(b_mlp2))

        # 最终 RMSNorm
        if encoder.norm is not None and torch_final_norm is not None:
            encoder.norm.weight.copy_(ic_from_torch(torch_final_norm.weight))


def copy_vitblock_weights(lang_block: ViTBlock, torch_block: TorchViTBlock):
    """把 Torch ViTBlock 的权重拷贝到语言侧 InfiniCore ViTBlock."""
    with torch.no_grad():
        # RMSNorm
        lang_block.norm1.weight.copy_(ic_from_torch(torch_block.norm1.weight))
        lang_block.norm2.weight.copy_(ic_from_torch(torch_block.norm2.weight))

        # Self-Attention
        ta = torch_block.attn

        lang_block.attn.q_proj.weight.copy_(ic_from_torch(ta.q_proj.weight))
        lang_block.attn.q_proj.bias.copy_(ic_from_torch(ta.q_proj.bias))

        lang_block.attn.k_proj.weight.copy_(ic_from_torch(ta.k_proj.weight))
        lang_block.attn.k_proj.bias.copy_(ic_from_torch(ta.k_proj.bias))

        lang_block.attn.v_proj.weight.copy_(ic_from_torch(ta.v_proj.weight))
        lang_block.attn.v_proj.bias.copy_(ic_from_torch(ta.v_proj.bias))

        lang_block.attn.out_proj.weight.copy_(ic_from_torch(ta.out_proj.weight))
        lang_block.attn.out_proj.bias.copy_(ic_from_torch(ta.out_proj.bias))

        # MLP
        w_mlp1 = torch_block.mlp[0].weight
        b_mlp1 = torch_block.mlp[0].bias
        lang_block.mlp.fc1_weight.copy_(ic_from_torch(w_mlp1))
        lang_block.mlp.fc1_bias.copy_(ic_from_torch(b_mlp1))

        w_mlp2 = torch_block.mlp[2].weight
        b_mlp2 = torch_block.mlp[2].bias
        lang_block.mlp.fc2_weight.copy_(ic_from_torch(w_mlp2))
        lang_block.mlp.fc2_bias.copy_(ic_from_torch(b_mlp2))


def main():
    # 一个很小的 toy 例子，GPU/CPU 上都很快
    B, V, C, H, W = 1, 2, 3, 8, 8
    patch_size = (4, 4)
    hidden_dim = 32
    num_layers = 2  # VisionEncoder 内部 ViTBlock 层数
    num_heads = 4
    text_len = 5

    device = infinicore.device("cuda", 0)
    dtype = infinicore.float16
    # 默认使用 Torch baseline 生成一套稳定权重并拷贝到 InfiniCore 模块，避免随机初始化导致的 NaN
    use_torch_baseline = True

    torch.manual_seed(0)

    # 1. 图像张量（用随机噪声替代真实图像），放在 CUDA 上以匹配 encoder 的 device
    images_t = torch.randn(B, V, C, H, W, dtype=torch.float16, device="cuda")
    images = infinicore.from_torch(images_t)

    # 2. image_newline / view_separator embedding（在 patch embedding 维度上），同样放在 CUDA 上
    P_h, P_w = patch_size
    D_v = C * P_h * P_w
    image_newline_t = torch.randn(D_v, dtype=torch.float16, device="cuda")
    view_separator_t = torch.randn(D_v, dtype=torch.float16, device="cuda")
    image_newline = infinicore.from_torch(image_newline_t)
    view_separator = infinicore.from_torch(view_separator_t)

    # 3. VisionEncoder：图像 -> 视觉 token 序列
    encoder = VisionEncoder(
        in_channels=C,
        patch_size=patch_size,
        vision_embed_dim=D_v,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
        dtype=dtype,
    )

    if use_torch_baseline:
        # 3.0 构造 Torch 版 VisionEncoder，并跑一遍获取权重（保持在 CUDA 上）
        images_t_fp16 = images_t  # 已是 float16 + cuda
        image_newline_t_fp16 = image_newline_t
        view_separator_t_fp16 = view_separator_t
        (
            vision_tokens_t_torch,
            torch_mlp,
            torch_blocks,
            torch_final_norm,
        ) = build_torch_vision_encoder(
            images_t_fp16,
            image_newline_t_fp16,
            view_separator_t_fp16,
            patch_size=patch_size,
            vision_embed_dim=D_v,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=4.0,
            eps=1e-6,
            final_norm=True,
        )

        # 3.1 把 Torch 权重拷贝到 InfiniCore VisionEncoder
        copy_vision_encoder_weights(encoder, torch_mlp, torch_blocks, torch_final_norm)

    # 3.2 用 InfiniCore VisionEncoder 前向
    vision_tokens = encoder(
        images,
        image_newline_embed=image_newline,
        view_separator_embed=view_separator,
    )  # [B, T_v, hidden_dim]
    vision_tokens_t = to_torch(vision_tokens)

    # 4. 构造一段假想的文本 token embedding（放在与 vision_tokens 相同的设备上）
    text_tokens_t = torch.randn(
        B, text_len, hidden_dim, dtype=torch.float16, device=vision_tokens_t.device
    )
    text_tokens = infinicore.from_torch(text_tokens_t)

    # 5. 在序列维度上拼接视觉 token 和文本 token
    joint_tokens_t = torch.cat([vision_tokens_t, text_tokens_t], dim=1)
    joint_tokens = infinicore.from_torch(joint_tokens_t)  # [B, T_v+T_t, hidden_dim]

    # 6. 通过一层 ViTBlock 做联合建模（相当于一个基于 MLA 的 Transformer Block）
    lang_block = ViTBlock(
        embed_dim=hidden_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        device=device,
        dtype=dtype,
    )

    if use_torch_baseline:
        # 6.1 构造 Torch 版 ViTBlock，并跑一遍，初始化权重
        torch_lang_block = TorchViTBlock(
            hidden_dim, num_heads, mlp_ratio=4.0, eps=1e-6
        ).to(device=joint_tokens_t.device, dtype=joint_tokens_t.dtype)
        _ = torch_lang_block(joint_tokens_t)  # 跑一遍只是为了确保状态完整

        # 6.2 把 TorchViTBlock 权重拷贝到 InfiniCore 语言侧 ViTBlock
        copy_vitblock_weights(lang_block, torch_lang_block)

    joint_out = lang_block(joint_tokens)  # [B, T_v+T_t, hidden_dim]
    joint_out_t = to_torch(joint_out)

    B_out, T_out, C_out = joint_out_t.shape

    # 7. 极简 vocab head：只看最后 text_len 个“文本”位置，映射到 vocab logits
    vocab_size = 16
    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False).to(
        dtype=joint_out_t.dtype, device=joint_out_t.device
    )

    # 模拟语言侧输出：取最后 text_len 个 token
    text_out_t = joint_out_t[:, -text_len:, :]  # [B, T_t, hidden_dim]
    logits_t = lm_head(text_out_t)  # [B, T_t, vocab_size]

    # 取最后一个 token 的 top-3 预测，做个可视化
    last_logits = logits_t[0, -1]  # [vocab_size]
    topk_vals, topk_idx = torch.topk(last_logits, k=3, dim=-1)

    print("=== Vision→Language Demo ===")
    print(f"Images shape        : {list(images_t.shape)}  (B, V, C, H, W)")
    print(f"Vision tokens shape : {list(vision_tokens_t.shape)}  (B, T_v, hidden_dim)")
    print(f"Text tokens shape   : {list(text_tokens_t.shape)}  (B, T_t, hidden_dim)")
    print(f"Joint tokens shape  : {list(joint_tokens_t.shape)}  (B, T_v+T_t, hidden_dim)")
    print(f"Output shape        : {[B_out, T_out, C_out]}  (B, T_v+T_t, hidden_dim)")
    print(f"Logits shape        : {list(logits_t.shape)}  (B, T_t, vocab_size)")
    print(f"Last token top-3 ids: {topk_idx.tolist()}")
    print(f"Last token top-3 val: {topk_vals.tolist()}")

    # 简单 sanity check：输出与 joint_tokens 形状一致
    assert B_out == B and C_out == hidden_dim
    assert T_out == joint_tokens_t.shape[1]


if __name__ == "__main__":
    main()