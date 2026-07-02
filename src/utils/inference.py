from __future__ import annotations

"""
Shared dataframe inference helpers.

The main pipeline and the experiment scripts both follow the same pattern:
extract article and candidate headlines, call ``score_titles()``, validate the
resulting scores, convert them into ranking strings and cache the raw scores.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.data_utils import get_source_text_task1, get_titles
from utils.scoring import ranking_from_scores, validate_scores


def score_dataframe_with_ranker(
    ranker,
    df_pred: pd.DataFrame,
    progress_every: int = 10,
    progress_label: str = "Predicted",
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Score all rows in a dataframe with a ranker implementing score_titles().

    Returns ranking strings and the raw score vectors, which can later be reused
    by the multimodal fusion step without recalculating the textual model.
    """
    preds: List[str] = []
    scores_cache: List[np.ndarray] = []

    for idx, (_, row) in enumerate(df_pred.iterrows(), start=1):
        article = get_source_text_task1(row)
        titles = get_titles(row)
        scores = validate_scores(
            ranker.score_titles(article, titles),
            expected_len=len(titles),
            context=f"ranker at row {idx}",
        )

        scores_cache.append(scores)
        preds.append(ranking_from_scores(scores))

        if progress_every and idx % progress_every == 0:
            print(f"{progress_label} {idx}/{len(df_pred)} rows...")

    return preds, scores_cache
