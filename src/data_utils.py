from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from config import REQUIRED_COLUMNS, TITLE_COLS, TOKENS_ALL


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}\n"
            f"Columnas presentes: {list(df.columns)}"
        )


def tokens(x: Any) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    return s.split() if s else []


def stable_seed(global_seed: int, row_key: str) -> int:
    h = hashlib.sha256(f"{global_seed}|{row_key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def get_source_text_task1(row: pd.Series) -> str:
    return str(row.get("article_body", "") or "").strip()


def get_source_text_task2(row: pd.Series) -> str:
    body = str(row.get("article_body", "") or "").strip()
    img = str(row.get("image_hash", "") or "").strip()
    return (img + "\n\n" + body).strip() if img else body


def get_titles(row: pd.Series) -> List[str]:
    return [str(row.get(col, "") or "") for col in TITLE_COLS]


def is_valid_rank(rank_tokens: List[str]) -> bool:
    return len(rank_tokens) == 10 and set(rank_tokens) == set(TOKENS_ALL)


def find_image_path(images_dir: Path, image_hash: Any) -> Optional[Path]:
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


def minmax_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)