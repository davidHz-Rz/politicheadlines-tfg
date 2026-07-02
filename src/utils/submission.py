from __future__ import annotations

"""
Submission utilities for the PoliticHeadlinES project.

This module builds, validates and saves submission files. A valid submission
contains one row per article and three columns: id, task_1 and task_2. The two
task columns must contain complete rankings of the ten candidate headline
tokens: t1, t2, ..., t10.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import TOKENS_ALL
from utils.metrics import parse_rank_list


REQUIRED_SUBMISSION_COLUMNS = ["id", "task_1", "task_2"]


def build_submission(
    ids: Iterable[str],
    task_1_preds: Iterable[str],
    task_2_preds: Iterable[str],
) -> pd.DataFrame:
    """
    Build the submission dataframe from article ids and task predictions.

    Parameters
    ----------
    ids:
        Article identifiers.
    task_1_preds:
        Rankings predicted for Task 1, encoded as strings such as "t3 t1 t7 ...".
    task_2_preds:
        Rankings predicted for Task 2, encoded with the same format.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns id, task_1 and task_2.
    """
    return pd.DataFrame(
        {
            "id": list(ids),
            "task_1": list(task_1_preds),
            "task_2": list(task_2_preds),
        }
    )


def validate_submission(submission: pd.DataFrame) -> None:
    """
    Validate that a submission has the expected columns and ranking format.

    Each task column must contain a complete permutation of TOKENS_ALL. This
    catches missing predictions, duplicated title tokens and malformed ranking
    strings before saving or evaluating the file.

    Raises
    ------
    ValueError
        If required columns are missing, if null values are found, or if any
        ranking is not a valid permutation of the expected title tokens.
    """
    missing = [c for c in REQUIRED_SUBMISSION_COLUMNS if c not in submission.columns]
    if missing:
        raise ValueError(f"Missing required submission columns: {missing}")

    for col in REQUIRED_SUBMISSION_COLUMNS:
        if submission[col].isna().any():
            raise ValueError(f"Column '{col}' contains null values.")

    expected = set(TOKENS_ALL)

    for col in ["task_1", "task_2"]:
        invalid_rows = []

        for idx, value in submission[col].items():
            tokens = parse_rank_list(value)

            if len(tokens) != len(TOKENS_ALL) or set(tokens) != expected:
                invalid_rows.append(idx)

        if invalid_rows:
            preview = invalid_rows[:5]
            raise ValueError(
                f"Column '{col}' contains invalid rankings in rows: {preview}."
            )


def save_submission(submission: pd.DataFrame, output_path: Path) -> None:
    """
    Save a submission dataframe to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
