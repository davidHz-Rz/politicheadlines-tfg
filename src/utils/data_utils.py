from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from config import REQUIRED_COLUMNS, TITLE_COLS, TOKENS_ALL
from utils.scoring import minmax_01  # Re-exported for backwards compatibility.


def validate_columns(df: pd.DataFrame) -> None:
    """
    Check that the dataframe contains all columns required by the pipeline.

    The project expects each row to contain an article identifier, the article
    body, the image hash and the ten candidate headline columns.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def tokens(x: Any) -> List[str]:
    """
    Split a value into whitespace-separated tokens.

    This helper is mainly useful for parsing ranking strings such as
    't3 t1 t5 ...' into token lists.
    """
    if x is None:
        return []

    s = str(x).strip()
    return s.split() if s else []


def stable_seed(global_seed: int, row_key: str) -> int:
    """
    Generate a deterministic seed from a global seed and a row identifier.

    Python's built-in hash is not stable across executions, so SHA-256 is used
    to obtain reproducible per-row seeds.
    """
    h = hashlib.sha256(f"{global_seed}|{row_key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def get_source_text_task1(row: pd.Series) -> str:
    """
    Return the textual input used for Task 1.

    Task 1 is text-only, so the source text is simply the article body.
    Missing values are converted to an empty string.
    """
    return str(row.get("article_body", "") or "").strip()


def get_source_text_task2(row: pd.Series) -> str:
    """
    Return the textual representation used by legacy Task 2 text pipelines.

    In the final multimodal experiments, images are handled separately by the
    VLM module. This helper is kept for compatibility with earlier experiments
    where the image hash was concatenated with the article body.
    """
    body = str(row.get("article_body", "") or "").strip()
    img = str(row.get("image_hash", "") or "").strip()

    return (img + "\n\n" + body).strip() if img else body


def get_titles(row: pd.Series) -> List[str]:
    """
    Extract the ten candidate headlines from a dataset row.
    """
    return [str(row.get(col, "") or "") for col in TITLE_COLS]


def is_valid_rank(rank_tokens: List[str]) -> bool:
    """
    Check whether a ranking contains exactly the ten expected title tokens.
    """
    return len(rank_tokens) == 10 and set(rank_tokens) == set(TOKENS_ALL)


def find_image_path(images_dir: Path, image_hash: Any) -> Optional[Path]:
    """
    Find the image file associated with an image hash.

    Returns None when the image hash is missing or the image file is not found.
    """
    if image_hash is None or (isinstance(image_hash, float) and np.isnan(image_hash)):
        return None

    h = str(image_hash).strip()
    if not h:
        return None

    exts = [".jpg", ".jpeg", ".png", ".webp"]

    for ext in exts:
        candidate = images_dir / f"{h}{ext}"
        if candidate.exists():
            return candidate

    candidate = images_dir / h
    if candidate.exists():
        return candidate

    return None
