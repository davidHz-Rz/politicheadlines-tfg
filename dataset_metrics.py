from pathlib import Path
import pandas as pd
import numpy as np
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

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_existing_images(df, image_dir):
    count = 0
    for h in df["image_hash"].fillna("").astype(str):
        if not h.strip():
            continue
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            if (image_dir / f"{h}{ext}").exists():
                count += 1
                break
    return count


def token_len(text):
    return len(tokenizer.tokenize(str(text)))


rows = []

for split, csv_path in DATASETS.items():
    df = pd.read_csv(csv_path)

    article_lengths = df["article_body"].fillna("").apply(token_len).to_numpy()

    headline_lengths = []
    for col in TITLE_COLS:
        headline_lengths.extend(df[col].fillna("").apply(token_len).tolist())

    headline_lengths = np.array(headline_lengths)

    rows.append({
        "Split": split,
        "Articles": len(df),
        "Images": count_existing_images(df, IMAGE_DIRS[split]),
        "Candidates/article": len(TITLE_COLS),
        "Avg article tokens": article_lengths.mean(),
        "Avg headline tokens": headline_lengths.mean(),
        "Max article tokens": article_lengths.max(),
        "Max headline tokens": headline_lengths.max(),
    })

stats = pd.DataFrame(rows)

print(stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\nLaTeX table:")
print(stats[[
    "Split",
    "Articles",
    "Images",
    "Candidates/article",
    "Avg article tokens",
    "Avg headline tokens"
]].to_latex(index=False, float_format="%.2f"))