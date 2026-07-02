from __future__ import annotations

"""
Shared utilities for Transformer-based rankers.

This module centralizes device selection, tokenizer configuration, batch device
movement and common article-title pair encoding used by the cross-encoder style
models.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from config import FORCE_CPU


def get_device(force_cpu: bool = FORCE_CPU) -> str:
    """
    Return the torch device used for training and inference.
    """
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def enable_cuda_optimizations(device: str) -> None:
    """
    Enable safe CUDA matrix multiplication optimizations when running on GPU.
    """
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def should_use_slow_tokenizer(model_name_or_path: str | Path) -> bool:
    """
    Return True for models that are safer with the slow tokenizer.
    """
    name = str(model_name_or_path).lower()
    return "deberta" in name


def get_tokenizer_kwargs(model_name_or_path: str | Path) -> dict:
    """
    Return tokenizer keyword arguments for a model or checkpoint path.
    """
    if should_use_slow_tokenizer(model_name_or_path):
        return {"use_fast": False}
    return {}


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: str | torch.device,
    non_blocking: bool = True,
) -> dict:
    """
    Move all tensor values in a batch dictionary to the selected device.
    """
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def encode_article_title_pairs(
    tokenizer,
    articles: Sequence[str],
    titles: Sequence[str],
    max_length: int,
):
    """
    Tokenize article-title pairs using the standard cross-encoder settings.
    """
    return tokenizer(
        list(articles),
        list(titles),
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


def article_token_budget(tokenizer, title: str, max_length: int) -> int:
    """
    Compute how many tokens can be allocated to the article in a pair input.
    """
    title_tokens = tokenizer.tokenize("" if title is None else str(title))
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    return max(1, max_length - len(title_tokens) - special_tokens)

