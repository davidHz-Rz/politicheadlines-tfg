from __future__ import annotations

"""
Compute descriptive statistics for the original PoliticHeadlinES splits.

The script reports the number of articles, available images, number of
candidate headlines per article, and token-length statistics for articles and
candidate headlines. It also prints a compact LaTeX table.

This is an analysis helper script. It does not modify the dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


DATA_ROOT = Path("data")

DATASETS = {
    "Training": DATA_ROOT / "train_corpora" / "train_public.csv",
    "Development": DATA_ROOT / "development_phase_initial" / "dev_public.csv",
    "Test": DATA_ROOT / "test_public" / "test_public.csv",
}

IMAGE_DIRS = {
    "Training": DATA_ROOT / "train_corpora" / "images",
    "Development": DATA_ROOT / "development_phase_initial" / "images",
    "Test": DATA_ROOT / "test_public" / "images",
}

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENIZER_NAME = "dccuchile/bert-base-spanish-wwm-cased"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def count_existing_images(df: pd.DataFrame, image_dir: Path) -> int:
    """
    Count rows whose image hash can be resolved to an existing image file.
    """
    count = 0

    for image_hash in df["image_hash"].fillna("").astype(str):
        image_hash = image_hash.strip()
        if not image_hash:
            continue

        for ext in IMAGE_EXTENSIONS:
            if (image_dir / f"{image_hash}{ext}").exists():
                count += 1
                break

    return count


def token_len(text: object, tokenizer: AutoTokenizer) -> int:
    """
    Count tokens using the same BETO tokenizer used by the main experiments.
    """
    return len(tokenizer.tokenize(str(text)))


def compute_split_stats(
    split: str,
    csv_path: Path,
    image_dir: Path,
    tokenizer: AutoTokenizer,
) -> dict:
    """
    Compute descriptive statistics for one dataset split.
    """
    df = pd.read_csv(csv_path)

    article_lengths = (
        df["article_body"]
        .fillna("")
        .apply(lambda text: token_len(text, tokenizer))
        .to_numpy()
    )

    headline_lengths: list[int] = []
    for col in TITLE_COLS:
        headline_lengths.extend(
            df[col]
            .fillna("")
            .apply(lambda text: token_len(text, tokenizer))
            .tolist()
        )

    headline_lengths = np.asarray(headline_lengths)

    return {
        "Split": split,
        "Articles": len(df),
        "Images": count_existing_images(df, image_dir),
        "Candidates/article": len(TITLE_COLS),
        "Avg article tokens": article_lengths.mean(),
        "Avg headline tokens": headline_lengths.mean(),
        "Max article tokens": article_lengths.max(),
        "Max headline tokens": headline_lengths.max(),
    }


def main() -> None:
    """
    Print dataset statistics and a LaTeX table.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    rows = [
        compute_split_stats(
            split=split,
            csv_path=csv_path,
            image_dir=IMAGE_DIRS[split],
            tokenizer=tokenizer,
        )
        for split, csv_path in DATASETS.items()
    ]

    stats = pd.DataFrame(rows)

    print(stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nLaTeX table:")
    print(
        stats[
            [
                "Split",
                "Articles",
                "Images",
                "Candidates/article",
                "Avg article tokens",
                "Avg headline tokens",
            ]
        ].to_latex(index=False, float_format="%.2f")
    )


if __name__ == "__main__":
    main()
