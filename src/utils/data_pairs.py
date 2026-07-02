from __future__ import annotations

"""
Utilities to convert labelled articles into training pairs.

The pointwise cross-encoder models are trained with binary article-headline
pairs. For each article, the headline ranked first in y_true is used as the
positive example and the remaining candidate headlines are used as negatives.
"""

from typing import List, Tuple

import pandas as pd

from utils.data_utils import get_source_text_task1, get_titles
from utils.metrics import parse_rank_list


TrainingPair = Tuple[str, str, int]


def extract_positive_index(row: pd.Series, y_true_col: str = "y_true") -> int:
    """
    Extract the 0-based index of the positive headline from a labelled row.

    The y_true column stores the ideal ranking as title tokens, for example
    "t9 t3 t1 ...". For binary pointwise training, only the first token is
    considered positive.

    Parameters
    ----------
    row:
        Dataset row containing candidate headlines and a y_true ranking.
    y_true_col:
        Name of the column containing the ideal ranking.

    Raises
    ------
    ValueError
        If y_true is missing, malformed or points outside the available title
        range.
        
    Returns
    -------
    int
        0-based index of the headline ranked first in y_true.
    """
    if y_true_col not in row:
        raise ValueError(f"Column '{y_true_col}' was not found in the row.")

    y_true_tokens = parse_rank_list(row[y_true_col])

    if not y_true_tokens:
        raise ValueError("Row does not contain a valid y_true ranking.")

    top_token = y_true_tokens[0]

    if not isinstance(top_token, str) or not top_token.startswith("t"):
        raise ValueError(f"Invalid y_true token: {top_token!r}")

    try:
        positive_idx = int(top_token[1:]) - 1
    except ValueError as exc:
        raise ValueError(f"Invalid y_true token: {top_token!r}") from exc

    titles = get_titles(row)

    if positive_idx < 0 or positive_idx >= len(titles):
        raise ValueError(
            f"Positive index out of range: {positive_idx}. "
            f"Number of available candidate headlines: {len(titles)}."
        )

    return positive_idx


def build_pairs(df: pd.DataFrame, y_true_col: str = "y_true") -> List[TrainingPair]:
    """
    Convert a labelled dataframe into binary article-headline training pairs.

    Each article produces one pair per candidate headline. The candidate in
    the first position of y_true receives label 1, while all other candidates
    receive label 0.

    Parameters
    ----------
    df:
        Labelled dataframe containing article text, candidate headlines and
        y_true rankings.
    y_true_col:
        Name of the column containing the ideal ranking.

    Returns
    -------
    list[TrainingPair]
        List of tuples with the form (article_text, candidate_headline, label).
    """
    pairs: List[TrainingPair] = []

    for _, row in df.iterrows():
        article = get_source_text_task1(row)
        titles = get_titles(row)
        positive_idx = extract_positive_index(row, y_true_col=y_true_col)

        for i, title in enumerate(titles):
            label = 1 if i == positive_idx else 0
            pairs.append((article, title, label))

    return pairs