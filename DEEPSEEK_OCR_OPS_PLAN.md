## DeepSeek‑OCR 算子与功能开发规划表

本表用于跟踪在 InfiniCore 中支持 **DeepSeek‑OCR（含 DeepSeek‑V2 语言侧 + 视觉侧）** 所需的算子与功能开发情况。  
状态字段建议统一使用：`已有(复用)` / `待封装` / `待开发` / `进行中` / `已完成`。

---

### 一、语言侧（DeepSeek‑V2 解码器相关）

> 说明：短期可以只依赖 “已有(复用)” 的算子先跑起来；  
> 中长期再逐步补齐 MoE / MLA 等 DeepSeek‑V2 特有结构。

#### 1.1 基础线性 / Norm / 激活 / 张量操作

| 模块 | 算子/功能 | Torch 对应 | 形状/语义约定（简要） | 现有 InfiniCore 接口/替代 | API 差异（与 Torch 对齐情况） | 优先级 | 实现方式建议 | 状态 |
| ---- | --------- | ---------- | ---------------------- | ------------------------- | -------------------------------- | ------ | ------------ | ---- |
| NN.Linear | 线性层 | `torch.nn.Linear` / `F.linear` | `[B, L, Din] -> [B, L, Dout]`，支持多维 batch | `infinicore.nn.Linear` | 构造/forward 语义与 `nn.Linear` 类似，但目前仅支持部分 dtype/设备，参数命名略有差异 | 高 | 已由 `infinicore.nn.Linear` 提供，按需补充缺失接口 | 已有(复用) |
| RMSNorm | RMSNorm | DeepSeek/LLaMA 用 RMSNorm | `[B, L, D]` 最后一维归一化，支持多精度 | `infinicore.nn.RMSNorm` / `infinicore.ops.rms_norm` | 行为与 DeepSeek/LLaMA RMSNorm 对齐；暂不支持全部 PyTorch `LayerNorm` 选项 | 高 | 复用现有实现，如需与 DeepSeek 精确对齐可在构造参数上扩展 | 已有(复用) |
| 激活 | SiLU/GELU 等 | `F.silu` / `F.gelu` | 按元素非线性，支持 `[*, D]` | `infinicore.ops.silu` / `infinicore.ops.swiglu` | 目前只提供 silu/swiglu，尚无统一 `nn.functional.gelu`；调用方式为 `infinicore.silu(x)` 或封装在 MLP 中使用 | 中 | 复用现有 `silu`，如需 GELU 再按算子模板新增 | 部分已有 |
| 矩阵乘 | 通用 matmul | `torch.matmul` / `bmm` | 支持 `[B, ..., M, K] @ [B, ..., K, N]` | `infinicore.matmul` / `infinicore.ops.matmul` | 接口大致等价 `matmul(a, b, alpha=1.0, out=None)`；广播规则与 PyTorch 基本一致，暂不支持所有奇异情况 | 高 | 复用 `matmul`/`gemm`，注意 batch 维和 alpha 缩放 | 已有(复用) |
| Softmax | 通用 softmax | `F.softmax` | 默认沿指定维度（支持负 dim）做 softmax | `infinicore.nn.functional.softmax` + C++ Softmax 算子（InfiniOP backend） | 行为与 `torch.nn.functional.softmax(input, dim=-1)` 对齐：当前实现支持 CPU + F16/BF16/F32，内部对非最后一维 dim 通过 `permute` 适配 InfiniOP “最后一维” kernel | 中 | 已按 `causal_softmax` 模式封装通用 softmax/log_softmax，后续可扩展到更多设备 | 已完成 |
| 张量创建 | `empty/zeros/ones` | `torch.empty/zeros/ones` | 按 shape/dtype/device 创建 | `infinicore.empty/zeros/ones` | 语义与 Torch 接口基本一致，参数目前为位置参数为主 | 高 | 复用现有 Tensor 静态方法 | 已有(复用) |
| 视图操作 | `view/reshape/permute/narrow/as_strided/contiguous` | 同名 API | Llama/DeepSeek 所需基本重排操作 | `Tensor.view/permute/narrow/as_strided/contiguous`；`infinicore.narrow` | 大部分与 PyTorch 一致；`narrow` 在 Python 侧为 `infinicore.narrow(t, dim, start, length)`，等价于 `t.narrow(...)` | 高 | 已在 Tensor/ops 中实现，Python 层已提供 `narrow` 等封装 | 已有(复用) |

#### 1.2 DeepSeek‑V2 MoE Gate / Router 相关

| 模块 | 算子/功能 | Torch 对应 | 形状/语义约定（简要） | 现有 InfiniCore 接口/替代 | API 差异（与 Torch 对齐情况） | 优先级 | 实现方式建议 | 状态 |
| ---- | --------- | ---------- | ---------------------- | ------------------------- | -------------------------------- | ------ | ------------ | ---- |
| MoE Gate | Top‑K | `torch.topk` | 输入 `[*, E]`，按指定维度取前 k 值和索引 | `infinicore.topk` + C++ TopK 算子 | 语义与 `torch.topk(input, k, dim, largest, sorted=True)` 对齐：当前实现仅支持 CPU + F16/BF16/F32，返回 `(values, indices)`，indices 为 int64；`sorted` 参数目前总是返回有序结果 | 高 | 已实现 CPU 版本（任意维度、支持负 dim），后续可扩展到 GPU，并在需要时补充对 `sorted=False` 的更细节对齐 | 已完成 |
| MoE Gate | Softmax | `F.softmax` | 对 gate logits 做 softmax | `infinicore.nn.functional.softmax` | 统一走 `infinicore.nn.functional.softmax(input, dim=-1)` 接口，与 `F.softmax` 行为对齐 | 高 | 直接复用通用 softmax 实现 | 已完成 |
| MoE Gate | Sigmoid | `torch.sigmoid` | 用于某些 gate 辅助分支 | 暂无单独接口，可用 `torch.sigmoid` 先占位 | 计划提供 `infinicore.nn.functional.sigmoid(x)`，行为与 Torch 对齐 | 低 | 作为通用 elementwise 激活来实现 | 待开发 |
| MoE Gate | Scatter‑Add | `tensor.scatter_add` | 将专家输出根据路由索引累加回原位置 | `infinicore.scatter_add` + C++ ScatterAdd 算子 | 语义与 `torch.scatter_add(input, dim, index, src, *, out=None)` 对齐：当前实现仅支持 CPU + F16/BF16/F32，index=I64，支持连续与非连续张量 | 高 | 已实现 CPU 版本（任意维度/stride），后续可扩展到 GPU | 已完成 |
| MoE Gate | One‑Hot | `F.one_hot` | 将专家 id 转为 one‑hot，用于统计 | `infinicore.nn.functional.one_hot` + C++ OneHot 算子 | 行为与 `torch.nn.functional.one_hot(indices, num_classes=None/整数)` 对齐，当前实现仅支持 CPU + int64 indices | 中 | 已在 CPU 上实现 one_hot 算子，后续可扩展到 GPU 并支持更多整数 dtype | 已完成 |
| MoE Gate | Reduce Sum/Mean | `sum/mean` | 计算负载均衡 loss 所需统计量 | `infinicore.sum` / `infinicore.mean` + C++ Sum/Mean 算子 | 语义对齐常见 `torch.sum/torch.mean` 用法：支持 dim=None/int/tuple、keepdim；当前仅实现 CPU + F16/BF16/F32，单维度内核，tuple 与全局归约在 Python 层组合实现 | 中 | 已实现 CPU 版本（单维度 kernel + Python 组合多维归约），后续可根据需要扩展到 GPU | 已完成 |

#### 1.3 DeepSeek‑V2 MLA Attention 相关

| 模块 | 算子/功能 | Torch 对应 | 形状/语义约定（简要） | 现有 InfiniCore 接口/替代 | API 差异（与 Torch 对齐情况） | 优先级 | 实现方式建议 | 状态 |
| ---- | --------- | ---------- | ---------------------- | ------------------------- | -------------------------------- | ------ | ------------ | ---- |
| MLA | Q/K/V 投影 | 多个 `Linear` | `q_proj`、`kv_a_proj_with_mqa`、`kv_b_proj` | `infinicore.nn.Linear` 组合 | 与 Torch 中多头 Q/K/V 线性层用法类似，需要在模型构建层面手动拆分/合并头维 | 高 | 直接用现有 Linear 叠加实现 | 已有(复用) |
| MLA | RoPE | RoPE 系列函数 | 基础 RoPE（GPT‑J / GPT‑NeoX 风格），后续可扩展 rope_scaling / yarn 配置 | `infinicore.nn.functional.rope` + `infinicore.nn.RoPE` | 已支持 GPT‑J / GPT‑NeoX 两种算法、预计算 sin/cos cache，并通过 `nn.RoPE`/`F.rope` 在语言侧统一调用；DeepSeek 特定 rope_scaling/yarn 目前暂未实现 | 中 | 复用现有 RoPE 模块和算子，在需要时按 DeepSeek 需求新增 rope_scaling/yarn 变体或额外配置参数 | 已完成（基础版） |
| MLA | 注意力核 | 多步 matmul + softmax | 实现 DeepSeek MLA 的压缩 KV 流程 | `infinicore.attention`（标准 MHA）+ `infinicore.nn.MLAAttention` | 当前提供“组合版” MLA：`MLAAttention` 在 Python 侧基于 matmul+softmax 实现多头自注意力核心，可选与 `nn.RoPE` 组合；后续如需 DeepSeek 原版 Mixed Linear Attention 的压缩 KV kernel，可再新增专门算子 | 中 | 先用 `MLAAttention` + 现有 Linear/RoPE/TopK/Scatter 等算子拼装 DeepSeek‑V2 的 MLA 结构，性能足够验证模型逻辑；如有需要，再在 C++/InfiniOP 中实现专门 MLA kernel | 已完成（组合版） |
| MLA | KV Cache 管理 | `cache_kv` | 对 KV 做增量更新与重排 | `infinicore.nn.functional.init_kv_cache` / `update_kv_cache` / `slice_kv_cache` | 提供简单的 KV cache 初始化/更新/切片 helper：目前支持 `[B, num_heads, max_seq_len, head_dim]` 形状，在 Python 侧基于 `contiguous()+narrow+copy_` 组合实现，适合大多数 decoder 增量推理场景 | 中 | 纯 Python helper，后端依赖已有 `narrow`/`view`/`copy_`；若未来在热点路径上有性能瓶颈，可进一步下沉到 C++/kernel | 已完成 |

#### 1.4 采样 / 调试辅助（可选）

| 模块 | 算子/功能 | Torch 对应 | 形状/语义约定（简要） | 现有 InfiniCore 接口/替代 | API 差异（与 Torch 对齐情况） | 优先级 | 实现方式建议 | 状态 |
| ---- | --------- | ---------- | ---------------------- | ------------------------- | -------------------------------- | ------ | ------------ | ---- |
| Sampling | Argmax | `torch.argmax` | `[*, V]` -> `[*,]`，取最后一维 argmax | `infinicore.argmax` + C++ ArgMax 算子，含 `dim`/`keepdim` 封装 | 行为已与 `torch.argmax(x, dim=None/维度, keepdim=...)` 对齐，当前仅支持 CPU + F32/I32 输入 | 中 | 作为通用 reduction 算子实现，内部自动处理非连续张量（先 `contiguous()`） | 已完成 |
| Sampling | LogSoftmax | `F.log_softmax` | logits -> log‑probs | `infinicore.nn.functional.log_softmax` + C++ LogSoftmax 算子（InfiniOP backend） | 行为与 `torch.nn.functional.log_softmax(input, dim=-1)` 对齐：当前实现支持 CPU + F16/BF16/F32，内部对非最后一维 dim 通过 `permute` 适配 InfiniOP kernel | 低 | 已直接封装 InfiniOP logsoftmax，后续视需求扩展到更多设备 | 已完成 |
| Debug | Tensor→Host | `tensor.cpu().numpy()` | 小批量张量复制到 host 方便调试 | `infinicore.utils.to_torch(tensor)` | 通过 `to_torch` 返回一个与原 Tensor 形状/dtype/device 一致且连续的 PyTorch Tensor，便于进一步 `.cpu().numpy()` / 分析；当前实现主要用于调试场景 | 中 | 纯 Python helper，内部复用现有 `from_torch` + `copy_` 路径；如有需要可在 C++ 层补充更高效的 D2H 拷贝接口 | 已完成 |

---

### 二、视觉侧（DeepSeek‑OCR 视觉编码器 / SAM / CLIP）

> 目标是逐步把 Vision Encoder / MlpProjector 等从 PyTorch 迁移到 InfiniCore，  
> 初期可以只保留 feature extraction 在 PyTorch，中后期再下沉到 C++/Kernel。

#### 2.1 ViT / CLIP 基础算子

| 模块 | 算子/功能 | Torch 对应 | 形状/语义约定（简要） | 现有 InfiniCore 接口/替代 | API 差异（与 Torch 对齐情况） | 优先级 | 实现方式建议 | 状态 |
| ---- | --------- | ---------- | ---------------------- | ------------------------- | -------------------------------- | ------ | ------------ | ---- |
| Conv2d | 2D 卷积 | `torch.nn.Conv2d` / `F.conv2d` | `[B, C_in, H, W] -> [B, C_out, H_out, W_out]`，目前仅支持 groups=1 | `infinicore.nn.functional.conv2d` + C++ Conv2d 算子（InfiniOP conv backend） | 接口对齐常见 `F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)` 用法：当前仅支持 CPU + F16/F32、groups=1，支持非连续输入；groups>1 暂未实现 | 高 | 复用 InfiniOP `infiniopConv`，在 C++ 层做形状/参数校验与 workspace 管理，后续可按需扩展 BF16 / 多设备 / groups>1 | 已完成 |
| Norm | LayerNorm / RMSNorm | `nn.LayerNorm` | `[B, H*W, C]` 或 `[B, C, H, W]` | `infinicore.nn.RMSNorm` | 与 `LayerNorm` 不完全等价；可通过 reshape 成 `[B*H*W, C]` 先用 RMSNorm | 中 | 复用现有 RMSNorm 逻辑，做维度兼容/包装函数 | 待封装 |
| Attention | ViT Self‑Attention / ViT Block | `nn.MultiheadAttention` / 自行实现 | 输入 `[B, N, C]`，N=patch 数 | `infinicore.nn.ViTSelfAttention` / `infinicore.nn.ViTBlock` | 已提供基于 `MLAAttention` 的多头自注意力模块：`ViTSelfAttention`（内部 Q/K/V Linear + MLA attention + out Linear），以及标准 Pre‑Norm 结构的 `ViTBlock`（RMSNorm + Self‑Attn + MLP），当前实现为 Python 组合版，适配 DeepSeek‑OCR/CLIP 风格 ViT Block | 中 | Python 层基于 `Linear + MLAAttention + RMSNorm + MlpProjector` 组合实现 ViT Encoder Block；后续如需更高性能可在 C++/InfiniOP 中增加专用 ViT Self‑Attention kernel | 已完成（组合版） |
| MLP | Patch MLP | Linear+激活+Linear | `[B, N, C]` -> `[B, N, C]` | `infinicore.nn.Linear` + `silu/gelu` | 需要在模型代码中组合使用；接口层面与 Torch `nn.Sequential(Linear, Act, Linear)` 等价 | 中 | 直接用 Linear+激活 组合 | 已有(复用) |
| Positional Encoding | 2D/1D 位置编码 | ViT/SAM 中的 PE | 按模型具体实现定义 | `infinicore.nn.functional.add_2d_positional_encoding` | 提供简单 helper：支持 `x` 为 `[B,C,H,W]` (BCHW) 或 `[B,N,C]` (BNC)，`pe` 分别为 `[1,C,H,W]/[C,H,W]` 或 `[1,N,C]/[N,C]`，通过广播相加实现 2D PE 叠加 | 低 | 纯 Python 实现，内部仅用现有 `view`/广播规则组合，足够支撑 ViT/SAM 等场景；如需更复杂的 PE，可在模型层自定义 | 已完成 |

#### 2.2 图像 Token & Projector 相关

| 模块 | 算子/功能 | Torch 对应 | 形状/语义约定（简要） | 现有 InfiniCore 接口/替代 | API 差异（与 Torch 对齐情况） | 优先级 | 实现方式建议 | 状态 |
| ---- | --------- | ---------- | ---------------------- | ------------------------- | -------------------------------- | ------ | ------------ | ---- |
| Patchify | 图像切 patch | `unfold` / reshape | `[B, C, H, W] -> [B, N, P^2*C]`，其中 `N=(H/P_h)*(W/P_w)` | `infinicore.nn.functional.patchify`（Python 侧封装） | 行为与常见 ViT Patchify 一致：`patch_size` 支持 int 或 (P_h, P_w)，当前仅支持 NCHW、要求 H/W 可整除 patch_size | 中 | 纯 Python 实现，内部使用 `view/permute` 等基础算子组合；后续如有性能需求可下沉到 C++/kernel | 已完成 |
| Token 组合 | image_newline / view_separator / 整体 token 流 | 多步 `view/cat` | 按行在末尾插入 `image_newline`，并在不同视图之间插入 `view_separator`，再展平成一条序列 | `infinicore.nn.functional.add_image_newline` / `add_view_separator` / `build_vision_tokens` | 已提供三个 Python helper：`add_image_newline(tokens, image_newline_embed, grid_size)` 处理每行末尾新增换行 token；`add_view_separator(tokens, view_lengths, view_sep_embed)` 在多个视图段之间插入分隔 token；`build_vision_tokens(images, patch_size, image_newline_embed, view_separator_embed)` 一站式从 `[B,(V,)C,H,W]` 构建完整视觉 token 序列 `[B,T,D]` | 中 | Python 侧 helper 封装视觉 token 拼接逻辑，内部基于 `to_torch/from_torch` + `torch.view/cat/split` 组合；后续如需更高性能可下沉到 C++/kernel 实现 concat / 专用 kernel | 已完成 |
| Projector | MlpProjector / VisionFrontend / VisionEncoder | Linear+激活+Linear / ViT Encoder | 把 vision feature 投到 text hidden_dim，并叠加若干 ViT Block 得到最终视觉特征 | `infinicore.nn.MlpProjector` / `VisionFrontend` / `ViTBlock` / `VisionEncoder` | `MlpProjector` 等价两层 MLP：`Linear(C_in->hidden)` + SiLU + `Linear(hidden->C_out)`；`VisionFrontend` 负责 `patchify + image_newline/view_separator + projector`；`ViTBlock` 和 `VisionEncoder` 提供基于 MLA 的 ViT Encoder Block 堆叠，输出 `[B,T,hidden]` 供语言侧使用 | 高 | Python 模块封装，内部复用 `F.linear`/`F.silu`/`MLAAttention` 等算子，整体作为 DeepSeek‑OCR 视觉编码前端；后续如需性能可将关键路径下沉到 C++/InfiniOP | 已完成（组合版） |
| （备注）CPU FP16 数值稳定性 |  |  |  |  | 在 VisionEncoder + ViTBlock 深堆叠、随机初始化较大权重时，CPU + FP16 路径下更易出现数值放大甚至 NaN；算子级单测（RMSNorm / Linear 等）在大幅 FP16 场景下与 Torch 对齐，但端到端 demo 仍建议优先使用 GPU（cuda）或在 CPU 上使用 FP32/预训练权重，以减少排查成本 |  |  | 说明性备注 |
| Upsample | 插值/上采样 | `F.interpolate` | 用于 SAM 中 mask 相关处理 | 暂无对应算子，推荐继续在 PyTorch 端实现 | 未来可以设计 `infinicore.nn.functional.interpolate(x, scale_factor/size, mode)` 接口 | 低 | 可以先保留在 PyTorch，暂不在 InfiniCore 实现 | 暂缓 |

---

### 三、通用工具 / 基础设施

| 模块 | 功能 | 说明 | 优先级 | 实现方式建议 | 状态 |
| ---- | ---- | ---- | ------ | ------------ | ---- |
| Context | 多设备支持 | CPU / NVIDIA / 其他后端的上下文切换 | 中 | 复用现有 `infinicore.context`，为新算子接入 | 已有(复用) |
| 测试框架 | 算子单测 | 统一使用 `test/infinicore/ops` 下的 `BaseOperatorTest` | 高 | 新算子按 `add`/`test_mul` 模板补全单测 | 进行中 |
| 文档 | 开发指南 | C++/Python/测试 统一示例 | 高 | 已有 `src/infinicore/ops/README.md` + `test_mul` 示例；本表为 OCR 专用补充 | 已完成 |

---

### 四、迭代建议（里程碑划分）

- **里程碑 0：最小可用版本**
  - 仅依赖 “语言侧 1.1 + 现有 attention/matmul/embedding”；
  - DeepSeek‑OCR 语言 decoder 可以在 InfiniCore 上跑，Vision 仍在 PyTorch。

- **里程碑 1：语言侧结构对齐 DeepSeek‑V2**
  - 补齐 MoE Gate（Top‑K、scatter‑add 等）与 MLA 相关算子；
  - 在不追求极致性能的前提下，结构与 PyTorch 模型基本一致。

- **里程碑 2：视觉侧迁移**
  - 实现 Conv2d / ViT Attention / Projector 等核心算子；
  - DeepSeek‑OCR 的 Vision Encoder 可以在 InfiniCore 上独立跑通。

- **里程碑 3：性能与多后端优化**
  - 为关键算子补充 GPU/各厂商加速卡后端 kernel；
  - 优化 KV cache、MoE 路由等热点路径。

> 后续你可以直接在本表的 “状态” 列中维护进度，也可以为每一行增加 “Owner / Issue ID” 等列，配合你们内部的任务管理流程使用。


