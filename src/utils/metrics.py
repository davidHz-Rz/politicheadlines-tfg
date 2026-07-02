from __future__ import annotations

"""
Evaluation utilities for the PoliticHeadlinES project.

This module provides helper functions to parse ranking strings, normalize
predicted title tokens, compute nDCG-based scores, and evaluate complete
submission files against local validation/test labels.
"""

import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from config import ALPHA, N_COLS, NDCG_K


def parse_rank_list(x: Any) -> List[str]:
    """
    Parse a ranking representation into a list of title tokens.

    Supported formats include:
    - "t1 t7 t3"
    - "t1,t7,t3"
    - '["t1", "t7", "t3"]'
    - "t1"

    Empty or missing values return an empty list.
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

    s = (
        s.replace("\t", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace(";", " ")
    )

    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]

    return [p for p in parts if p]


def token_to_col(tok: Any) -> Optional[int]:
    """
    Convert a title token into its numeric candidate index.

    Examples
    --------
    "t7" -> 7
    "d3" -> 3

    Returns None when the token is missing, malformed or unsupported.
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
    Convert ranking tokens to valid candidate indices.

    Invalid tokens are ignored. Duplicated candidates are removed while
    preserving the original prediction order.
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


def ndcg_from_ideal(
    pred_cols: List[int],
    ideal_cols: List[int],
    k: int = NDCG_K,
) -> float:
    """
    Compute nDCG@k using an explicit ideal ranking.

    Gains are assigned according to the position in the ideal ranking. For a
    list of ten candidates, the first ideal item receives the highest gain and
    the last one receives the lowest gain.
    """
    if not ideal_cols:
        return 0.0

    ideal_rank: Dict[int, int] = {c: i for i, c in enumerate(ideal_cols)}

    def gain_for_col(c: int) -> float:
        rank = ideal_rank.get(c)

        if rank is None:
            return 0.0

        return float(len(ideal_cols) - rank)

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
    Compute the PA-nDCG score for a single prediction.

    The first predicted headline is treated as mandatory. If the predicted
    top-1 headline does not match the reference top-1 headline, the score is 0

    If the top-1 headline is correct, the score is computed as:

        alpha + (1 - alpha) * nDCG(rest_of_ranking)
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
    """
    Evaluate a complete submission file against a labelled CSV file.

    The validation file must contain columns "id" and "y_true".
    The results file must contain columns "id", "task_1" and "task_2".

    Predictions are matched to references by article id. Missing predictions
    are scored as 0. The function returns Pa-nDCG for both tasks, their mean,
    prediction coverage, and the metric parameters used.
    """
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


def score_task1_predictions_df(
    df_val: pd.DataFrame,
    task_1_preds: List[str],
    y_true_col: str = "y_true",
) -> Dict[str, float]:
    """
    Evaluate Task 1 predictions directly from a dataframe.

    This helper is useful during experiments, when predictions are already
    available in memory and there is no need to create a full submission file.

    Returns PA-nDCG and top-1 accuracy. The alias "top1_acc" is kept for
    compatibility with older experiment scripts.
    """
    if y_true_col not in df_val.columns:
        raise ValueError(f"Column '{y_true_col}' was not found for evaluation.")

    if len(df_val) != len(task_1_preds):
        raise ValueError(
            "The number of predictions does not match the number of dataframe rows."
        )

    scores: List[float] = []
    top1_hits = 0

    for (_, row), pred in zip(df_val.iterrows(), task_1_preds):
        y_true = parse_rank_list(row[y_true_col])
        pred_tokens = parse_rank_list(pred)

        scores.append(pa_ndcg(pred_tokens, y_true, k=NDCG_K, alpha=ALPHA))

        if pred_tokens and y_true and pred_tokens[0] == y_true[0]:
            top1_hits += 1

    n = max(len(task_1_preds), 1)
    top1_accuracy = float(top1_hits / n)

    return {
        "task_1_pa_ndcg": float(sum(scores) / n),
        "top1_accuracy": top1_accuracy,
        "top1_acc": top1_accuracy,
    }