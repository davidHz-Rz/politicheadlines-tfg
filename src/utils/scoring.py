from __future__ import annotations

"""
Shared scoring and ranking helpers.

These utilities convert raw model scores into ranking tokens, validate score
vectors and normalize scores before late fusion or ensembling. Keeping them in
one place avoids duplicating the same argsort/min-max logic across rankers.
"""

from typing import List, Sequence

import numpy as np

from config import TOKENS_ALL


def validate_scores(
    scores: Sequence[float],
    expected_len: int,
    context: str = "ranker",
) -> np.ndarray:
    """
    Convert a score sequence to a finite numpy array with the expected length.
    """
    scores_array = np.asarray(scores, dtype=float)

    if len(scores_array) != expected_len:
        raise ValueError(
            f"{context} returned {len(scores_array)} scores for "
            f"{expected_len} titles."
        )

    if not np.all(np.isfinite(scores_array)):
        raise ValueError(f"{context} returned non-finite scores.")

    return scores_array


def rank_tokens_from_scores(scores: Sequence[float]) -> List[str]:
    """
    Return title tokens ordered by descending score.
    """
    scores_array = np.asarray(scores, dtype=float)
    order = np.argsort(-scores_array)
    return [TOKENS_ALL[i] for i in order]


def ranking_from_scores(scores: Sequence[float]) -> str:
    """
    Convert one score per title into a tokenized ranking string.
    """
    return " ".join(rank_tokens_from_scores(scores))


def minmax_01(scores: Sequence[float]) -> np.ndarray:
    """
    Normalize scores to the [0, 1] range using min-max normalization.

    If all values are equal, the returned vector is zero because the scores do
    not provide relative information for that row.
    """
    scores_array = np.asarray(scores, dtype=float)

    if scores_array.size == 0:
        return scores_array

    if not np.all(np.isfinite(scores_array)):
        raise ValueError("Cannot normalize non-finite scores.")

    lo = float(scores_array.min())
    hi = float(scores_array.max())

    if hi - lo < 1e-12:
        return np.zeros_like(scores_array, dtype=float)

    return (scores_array - lo) / (hi - lo)


def order_to_scores(order: Sequence[int], n_items: int) -> np.ndarray:
    """
    Convert an ordered list of candidate indices into descending ordinal scores.
    """
    order_list = list(order)

    if len(order_list) != n_items:
        raise ValueError(
            f"Final order has {len(order_list)} elements, but {n_items} were expected."
        )

    if sorted(order_list) != list(range(n_items)):
        raise ValueError("Final order is not a valid permutation of the candidates.")

    scores = np.zeros(n_items, dtype=float)
    for rank, idx in enumerate(order_list):
        scores[idx] = float(n_items - rank)

    return scores
