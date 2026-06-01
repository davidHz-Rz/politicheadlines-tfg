from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import TOKENS_ALL
from utils.metrics import parse_rank_list

def build_submission(
    ids: Iterable[str],
    task_1_preds: Iterable[str],
    task_2_preds: Iterable[str],
) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "id": list(ids),
            "task_1": list(task_1_preds),
            "task_2": list(task_2_preds),
        }
    )
    return submission


def validate_submission(submission: pd.DataFrame) -> None:
    required_cols = ["id", "task_1", "task_2"]
    missing = [c for c in required_cols if c not in submission.columns]
    if missing:
        raise ValueError(f"Faltan columnas en la submission: {missing}")

    if submission["id"].isna().any():
        raise ValueError("La columna 'id' contiene valores nulos.")

    if submission["task_1"].isna().any():
        raise ValueError("La columna 'task_1' contiene valores nulos.")

    if submission["task_2"].isna().any():
        raise ValueError("La columna 'task_2' contiene valores nulos.")

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
                f"La columna '{col}' contiene rankings inválidos en filas: {preview}."
            )


def save_submission(submission: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission guardada en: {output_path}")