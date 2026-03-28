from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from config import ALPHA, N_COLS, NDCG_K


def parse_rank_list(x: Any) -> List[str]:
    """
    Acepta:
      - "t1 t7 t3" (espacios)
      - "t1,t7,t3" (comas)
      - '["t1","t7"]' (json list)
      - "t1" (top-1)
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    s = str(x).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(t).strip() for t in arr if str(t).strip()]
        except Exception:
            pass

    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ").replace(";", " ")
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]

    return [p for p in parts if p]


def token_to_col(tok: Any) -> Optional[int]:
    """
    't7' -> 7
    """
    if tok is None or (isinstance(tok, float) and pd.isna(tok)):
        return None

    s = str(tok).strip()
    if len(s) < 2:
        return None

    prefix = s[0].lower()
    if prefix not in ("t", "d"):
        return None

    try:
        return int(s[1:])
    except Exception:
        return None


def unique_valid_pred_cols(pred: List[str], n_cols: int = N_COLS) -> List[int]:
    """
    Convierte tokens a columnas 1..n_cols, elimina duplicados preservando orden.
    """
    out: List[int] = []
    seen = set()

    for tok in pred:
        n = token_to_col(tok)
        if n is None or not (1 <= n <= n_cols):
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(n)

    return out


def ndcg_from_ideal(pred_cols: List[int], ideal_cols: List[int], k: int = NDCG_K) -> float:
    """
    nDCG@k donde el ideal es un ranking explícito ideal_cols.
    Ganancia lineal por posición ideal: top=10..1 (si len=10).
    """
    if not ideal_cols:
        return 0.0

    ideal_rank: Dict[int, int] = {c: i for i, c in enumerate(ideal_cols)}

    def gain_for_col(c: int) -> float:
        r = ideal_rank.get(c, None)
        if r is None:
            return 0.0
        return float(len(ideal_cols) - r)

    dcg = 0.0
    for i, c in enumerate(pred_cols[:k], start=1):
        dcg += gain_for_col(c) / math.log2(i + 1)

    idcg = 0.0
    for i, c in enumerate(ideal_cols[:k], start=1):
        idcg += gain_for_col(c) / math.log2(i + 1)

    if idcg <= 0.0:
        return 0.0

    return max(0.0, min(1.0, dcg / idcg))


def pa_ndcg(
    pred_tokens: List[str],
    true_tokens: List[str],
    k: int = NDCG_K,
    alpha: float = ALPHA,
) -> float:
    """
    PA-nDCG:
      - Si top-1 no coincide -> 0
      - Si coincide -> alpha + (1-alpha)*aux_nDCG(resto)
    """
    if not pred_tokens or not true_tokens:
        return 0.0

    ideal_cols = unique_valid_pred_cols(true_tokens, N_COLS)
    pred_cols = unique_valid_pred_cols(pred_tokens, N_COLS)

    if not ideal_cols or not pred_cols:
        return 0.0

    if pred_cols[0] != ideal_cols[0]:
        return 0.0

    primary = ideal_cols[0]
    pred_rest = [c for c in pred_cols if c != primary]
    ideal_rest = [c for c in ideal_cols if c != primary]

    aux = ndcg_from_ideal(pred_rest, ideal_rest, k=k)
    score = alpha + (1.0 - alpha) * aux

    return max(0.0, min(1.0, score))


def score_submission(
    validation_csv: str,
    results_csv: str,
    k: int = NDCG_K,
    alpha: float = ALPHA,
) -> Dict[str, float]:
    ref = pd.read_csv(validation_csv, dtype={"id": str})
    sub = pd.read_csv(results_csv, dtype={"id": str})

    ref = ref[["id", "y_true"]].copy()
    sub = sub[["id", "task_1", "task_2"]].copy()

    ref["id"] = ref["id"].astype(str).str.strip()
    sub["id"] = sub["id"].astype(str).str.strip()

    ref = ref.dropna(subset=["id"]).drop_duplicates(subset=["id"], keep="first")
    sub = sub.dropna(subset=["id"]).drop_duplicates(subset=["id"], keep="first")

    merged = ref.merge(sub, on="id", how="left")

    n_total = len(merged)
    has_any = merged[["task_1", "task_2"]].notna().any(axis=1)
    coverage = float(has_any.mean()) if n_total else 0.0

    t1_scores: List[float] = []
    t2_scores: List[float] = []

    for _, row in merged.iterrows():
        y_true = parse_rank_list(row["y_true"])
        pred_1 = parse_rank_list(row["task_1"])
        pred_2 = parse_rank_list(row["task_2"])

        t1_scores.append(pa_ndcg(pred_1, y_true, k=k, alpha=alpha) if pred_1 else 0.0)
        t2_scores.append(pa_ndcg(pred_2, y_true, k=k, alpha=alpha) if pred_2 else 0.0)

    task_1_score = float(sum(t1_scores) / len(t1_scores)) if t1_scores else 0.0
    task_2_score = float(sum(t2_scores) / len(t2_scores)) if t2_scores else 0.0
    mean_score = (task_1_score + task_2_score) / 2.0

    return {
        "task_1_pa_ndcg": task_1_score,
        "task_2_pa_ndcg": task_2_score,
        "mean_pa_ndcg": mean_score,
        "coverage": coverage,
        "k": k,
        "alpha": alpha,
    }