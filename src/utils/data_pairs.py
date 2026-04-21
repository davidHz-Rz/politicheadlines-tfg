from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))

from utils.data_utils import get_source_text_task1, get_titles
from utils.metrics import parse_rank_list


def build_pairs(df: pd.DataFrame, y_true_col: str = "y_true") -> List[Tuple[str, str, int]]:
    """
    Convierte el dataset original en pares (article, title, label),
    tomando como positivo el primer elemento de y_true.
    """
    pairs = []

    for _, row in df.iterrows():
        article = get_source_text_task1(row)
        titles = get_titles(row)

        y_true_tokens = parse_rank_list(row[y_true_col])
        if not y_true_tokens:
            raise ValueError("Fila sin y_true válido.")

        top_token = y_true_tokens[0]   # p. ej. "t9"
        correct_idx = int(top_token[1:]) - 1

        for i, title in enumerate(titles):
            label = 1 if i == correct_idx else 0
            pairs.append((article, title, label))

    return pairs