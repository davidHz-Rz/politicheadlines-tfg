from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import SEED, TITLE_COLS
from utils.data_utils import validate_columns


TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

INPUT_CSV = PROJECT_ROOT / "data" / "competition" / "test_public" / "train_public.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tfg_split"


def extract_top1(y_true: str) -> str:
    tokens = str(y_true).strip().split()

    if not tokens:
        raise ValueError("Encontrado y_true vacío.")

    top1 = tokens[0]

    valid_tokens = {f"t{i}" for i in range(1, 11)}
    if top1 not in valid_tokens:
        raise ValueError(f"Token top1 inválido en y_true: {top1}")

    return top1


def add_stratification_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["top1"] = df["y_true"].apply(extract_top1)

    df["article_length"] = (
        df["article_body"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["length_bin"] = pd.qcut(
        df["article_length"],
        q=4,
        labels=["short", "medium", "long", "very_long"],
        duplicates="drop",
    ).astype(str)

    df["strata"] = df["top1"].astype(str) + "_" + df["length_bin"].astype(str)

    return df


def choose_stratify_column(df: pd.DataFrame) -> str:
    strata_counts = df["strata"].value_counts()

    if strata_counts.min() >= 2:
        return "strata"

    top1_counts = df["top1"].value_counts()

    if top1_counts.min() >= 2:
        return "top1"

    return ""


def check_id_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    if train_ids & val_ids:
        raise ValueError("Hay IDs solapados entre train y validation.")

    if train_ids & test_ids:
        raise ValueError("Hay IDs solapados entre train y test.")

    if val_ids & test_ids:
        raise ValueError("Hay IDs solapados entre validation y test.")


def distribution_table(df: pd.DataFrame, column: str) -> dict:
    counts = df[column].value_counts(normalize=True).sort_index()
    return {
        str(index): round(float(value), 6)
        for index, value in counts.items()
    }


def print_distribution(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column: str,
) -> None:
    print(f"\nDistribución de {name}")

    all_values = sorted(
        set(train_df[column])
        | set(val_df[column])
        | set(test_df[column])
    )

    print(f"{'valor':<20} {'train':>10} {'val':>10} {'test':>10}")

    for value in all_values:
        train_pct = (train_df[column] == value).mean() * 100
        val_pct = (val_df[column] == value).mean() * 100
        test_pct = (test_df[column] == value).mean() * 100

        print(
            f"{str(value):<20} "
            f"{train_pct:>9.2f}% "
            f"{val_pct:>9.2f}% "
            f"{test_pct:>9.2f}%"
        )


def remove_auxiliary_columns(df: pd.DataFrame) -> pd.DataFrame:
    auxiliary_columns = [
        "top1",
        "article_length",
        "length_bin",
        "strata",
    ]

    return df.drop(
        columns=[col for col in auxiliary_columns if col in df.columns]
    )


def save_split_metadata(
    output_dir: Path,
    source_csv: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stratify_column: str,
) -> None:
    metadata = {
        "source_csv": str(source_csv),
        "output_dir": str(output_dir),
        "seed": SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "total": len(train_df) + len(val_df) + len(test_df),
        },
        "stratification": {
            "requested": "top1 + article_length_quartile",
            "used_column": stratify_column or None,
            "fallback": None if stratify_column == "strata" else stratify_column or "none",
        },
        "top1_distribution": {
            "train": distribution_table(train_df, "top1"),
            "validation": distribution_table(val_df, "top1"),
            "test": distribution_table(test_df, "top1"),
        },
        "length_bin_distribution": {
            "train": distribution_table(train_df, "length_bin"),
            "validation": distribution_table(val_df, "length_bin"),
            "test": distribution_table(test_df, "length_bin"),
        },
    }

    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


def main() -> None:
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO debe sumar 1.")

    print(f"Leyendo dataset original: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    validate_columns(df)

    if "y_true" not in df.columns:
        raise ValueError("El CSV de entrenamiento debe contener la columna y_true.")

    if df["id"].duplicated().any():
        duplicated_ids = df.loc[df["id"].duplicated(), "id"].tolist()[:10]
        raise ValueError(f"Hay IDs duplicados. Ejemplos: {duplicated_ids}")

    missing_title_rows = df[TITLE_COLS].isna().any(axis=1).sum()
    if missing_title_rows > 0:
        raise ValueError(
            f"Hay {missing_title_rows} filas con titulares title_1...title_10 vacíos."
        )

    df = add_stratification_columns(df)

    stratify_column = choose_stratify_column(df)
    stratify_values = df[stratify_column] if stratify_column else None

    print(f"Filas totales: {len(df)}")
    print(f"Estrategia de estratificación: {stratify_column or 'sin estratificación'}")

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True,
        stratify=stratify_values,
    )

    temp_val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    temp_stratify_column = choose_stratify_column(temp_df)
    temp_stratify_values = (
        temp_df[temp_stratify_column]
        if temp_stratify_column
        else None
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=temp_val_ratio,
        random_state=SEED,
        shuffle=True,
        stratify=temp_stratify_values,
    )

    check_id_integrity(train_df, val_df, test_df)

    print("\nTamaños finales")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")
    print(f"Total:      {len(train_df) + len(val_df) + len(test_df)}")

    print_distribution("top1", train_df, val_df, test_df, "top1")
    print_distribution("length_bin", train_df, val_df, test_df, "length_bin")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_split_metadata(
        output_dir=OUTPUT_DIR,
        source_csv=INPUT_CSV,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stratify_column=stratify_column,
    )

    train_out = remove_auxiliary_columns(train_df)
    val_out = remove_auxiliary_columns(val_df)
    test_out = remove_auxiliary_columns(test_df)

    train_out.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_out.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_out.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nArchivos generados:")
    print(f"- {OUTPUT_DIR / 'train.csv'}")
    print(f"- {OUTPUT_DIR / 'val.csv'}")
    print(f"- {OUTPUT_DIR / 'test.csv'}")
    print(f"- {OUTPUT_DIR / 'split_metadata.json'}")

    print("\nPara usar este split, cambia en config.py:")
    print('ACTIVE_DATASET = "tfg_split"')


if __name__ == "__main__":
    main()