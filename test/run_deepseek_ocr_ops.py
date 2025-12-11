#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path


"""
Run a curated set of operator / module tests that DeepSeek‑OCR 依赖的核心算子。

用法示例（在仓库根目录执行）：
  - 只跑 CPU：
        python test/run_deepseek_ocr_ops.py --cpu
  - 跑 CPU + NVIDIA：
        python test/run_deepseek_ocr_ops.py --cpu --nvidia
  - 开启 debug / verbose：
        python test/run_deepseek_ocr_ops.py --cpu --nvidia --debug --verbose

脚本会依次调用各个单测文件，退出码为 0 表示所有相关算子单测均无失败。
"""


ROOT = Path(__file__).resolve().parents[1]

# DeepSeek‑OCR 语言侧 / 视觉侧核心算子 & 组合模块对应的单测脚本
DEEPSEEK_OCR_TESTS = [
    # --- 语言侧基础算子（Linear / RMSNorm / SiLU / matmul / softmax/log_softmax 等） ---
    "test/infinicore/ops/linear.py",
    "test/infinicore/ops/rms_norm.py",
    "test/infinicore/ops/silu.py",
    "test/infinicore/ops/matmul.py",
    "test/infinicore/ops/log_softmax.py",
    "test/infinicore/ops/causal_softmax.py",
    # --- 语言侧 MoE / gate 相关 ---
    "test/infinicore/ops/topk.py",
    "test/infinicore/ops/scatter_add.py",
    "test/infinicore/ops/one_hot.py",
    "test/infinicore/ops/sum.py",
    "test/infinicore/ops/mean.py",
    "test/infinicore/ops/argmax.py",
    "test/infinicore/ops/kvcache.py",
    "test/infinicore/ops/rope.py",
    # --- 视觉侧基础算子 ---
    "test/infinicore/ops/conv2d.py",
    "test/infinicore/ops/patchify.py",
    "test/infinicore/ops/image_newline.py",
    "test/infinicore/ops/view_separator.py",
    # --- 视觉 token 组合 / ViT Block / VisionEncoder / MLA Attention ---
    "test/infinicore/nn/vision_tokens.py",
    "test/infinicore/nn/vit_block.py",
    "test/infinicore/nn/vision_encoder.py",
    "test/infinicore/nn/mla_attention.py",
]


def main(argv: list[str]) -> int:
    # 把所有硬件相关 flag（--cpu / --nvidia / --ascend 等）以及 debug/verbose 选项原样透传给各个单测
    test_args = argv[1:]

    failed: list[tuple[str, int]] = []

    for rel_path in DEEPSEEK_OCR_TESTS:
        test_path = ROOT / rel_path
        if not test_path.is_file():
            print(f"[DeepSeek‑OCR Ops] WARNING: test file not found: {rel_path}")
            continue

        cmd = [sys.executable, str(test_path), *test_args]
        banner = f"[DeepSeek‑OCR Ops] Running {rel_path} {' '.join(test_args)}"
        print("\n" + "=" * len(banner))
        print(banner)
        print("=" * len(banner))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append((rel_path, result.returncode))

    print("\n" + "=" * 60)
    if failed:
        print("[DeepSeek‑OCR Ops] Some tests FAILED:")
        for path, code in failed:
            print(f"  - {path} (exit code {code})")
        print("=" * 60)
        return 1

    print("[DeepSeek‑OCR Ops] All selected operator/module tests PASSED.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


