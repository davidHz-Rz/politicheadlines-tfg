#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Ranking error analysis for PoliticHeadlinES-IberLEF 2026.

The script compares a prediction CSV with a labelled test split. It computes
global metrics, error distributions based on the final position of the correct
headline, length-based analysis, and exports error cases for manual inspection.

Recommended usage:
    python analyze_ranking_errors.py \
        --pred "base_ensemble(beto_headtail_0.40_beto_0.45_mdeberta_0.15)_tail_rank10.csv" \
        --test test.csv \
        --out_dir error_analysis_outputs

Expected prediction CSV format:
    id,prediction
    <id>,"t3 t9 t6 ..."

Expected test CSV format:
    id,article_body,title_1,...,title_10,y_true
    <id>,...,"t3 t7 t9 ..."
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_ranking(value: object) -> List[str]:
    """
    Parse a ranking into a list of tokens such as ["t3", "t9", ...].

    The parser accepts simple whitespace-separated rankings as well as rankings
    containing commas, brackets or quotes.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    return re.findall(r"t(?:10|[1-9])", text)


def count_words(text: object) -> int:
    """
    Count words using simple whitespace splitting.
    """
    if pd.isna(text):
        return 0

    return len(str(text).split())


def title_text(row: pd.Series, title_id: str, title_prefix: str = "title_") -> str:
    """
    Return the headline text associated with a token t1...t10.
    """
    if not title_id or not re.fullmatch(r"t(?:10|[1-9])", title_id):
        return ""

    idx = title_id[1:]
    col = f"{title_prefix}{idx}"

    return str(row.get(col, ""))


def error_bucket(rank: Optional[int]) -> str:
    """
    Map the correct-headline position to a human-readable error category.
    """
    if rank is None:
        return "Not found"
    if rank == 1:
        return "Top-1 correct"
    if 2 <= rank <= 3:
        return "Minor error"
    if 4 <= rank <= 5:
        return "Medium error"
    if 6 <= rank <= 10:
        return "Severe error"

    return "Not found"


def validate_rankings(df: pd.DataFrame, pred_col: str, truth_col: str) -> None:
    """
    Warn when parsed rankings do not contain exactly ten unique candidates.
    """
    for col in [pred_col, truth_col]:
        lengths = df[col].apply(len)
        unique_lengths = df[col].apply(lambda x: len(set(x)))
        invalid_length = (lengths != 10).sum()
        invalid_unique = (unique_lengths != 10).sum()

        if invalid_length or invalid_unique:
            print(
                f"[WARN] Column {col}: {invalid_length} rankings without 10 elements, "
                f"{invalid_unique} rankings with duplicated candidates."
            )


def build_cases(
    pred_path: Path,
    test_path: Path,
    id_col: str,
    prediction_col: str,
    truth_col: str,
    article_col: str,
    title_prefix: str,
) -> pd.DataFrame:
    """
    Load predictions and references, merge them and compute per-article fields.
    """
    pred = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)

    required_pred = {id_col, prediction_col}
    required_test = {id_col, truth_col, article_col}

    missing_pred = required_pred - set(pred.columns)
    missing_test = required_test - set(test.columns)

    if missing_pred:
        raise ValueError(f"Missing columns in prediction file: {sorted(missing_pred)}")

    if missing_test:
        raise ValueError(f"Missing columns in test file: {sorted(missing_test)}")

    df = test.merge(pred[[id_col, prediction_col]], on=id_col, how="inner")

    if len(df) != len(test) or len(df) != len(pred):
        print(
            f"[WARN] Row mismatch after merge: "
            f"test={len(test)}, pred={len(pred)}, common={len(df)}"
        )

    df["y_true_list"] = df[truth_col].apply(parse_ranking)
    df["prediction_list"] = df[prediction_col].apply(parse_ranking)

    validate_rankings(df, "prediction_list", "y_true_list")

    correct_ids = []
    predicted_top_ids = []
    correct_ranks = []
    exact_matches = []

    for _, row in df.iterrows():
        y_true = row["y_true_list"]
        pred_rank = row["prediction_list"]

        correct_id = y_true[0] if y_true else None
        predicted_top = pred_rank[0] if pred_rank else None

        if correct_id in pred_rank:
            rank = pred_rank.index(correct_id) + 1
        else:
            rank = None

        correct_ids.append(correct_id)
        predicted_top_ids.append(predicted_top)
        correct_ranks.append(rank)
        exact_matches.append(y_true == pred_rank)

    df["correct_title_id"] = correct_ids
    df["predicted_top_id"] = predicted_top_ids
    df["correct_rank"] = correct_ranks
    df["top1_correct"] = df["correct_rank"].eq(1)
    df["top3_correct"] = df["correct_rank"].le(3)
    df["top5_correct"] = df["correct_rank"].le(5)
    df["exact_ranking"] = exact_matches
    df["error_type"] = df["correct_rank"].apply(error_bucket)
    df["word_count"] = df[article_col].apply(count_words)

    df["correct_title_text"] = df.apply(
        lambda r: title_text(r, r["correct_title_id"], title_prefix),
        axis=1,
    )
    df["predicted_top_text"] = df.apply(
        lambda r: title_text(r, r["predicted_top_id"], title_prefix),
        axis=1,
    )
    df["article_excerpt"] = df[article_col].fillna("").astype(str).str.slice(0, 350)

    return df


def compute_global_metrics(df: pd.DataFrame) -> dict:
    """
    Compute global ranking metrics for the analysed predictions.
    """
    return {
        "n_rows": int(len(df)),
        "top1_accuracy": float(df["top1_correct"].mean()),
        "top3_hit_rate": float(df["top3_correct"].mean()),
        "top5_hit_rate": float(df["top5_correct"].mean()),
        "exact_ranking_matches": int(df["exact_ranking"].sum()),
        "mean_correct_rank": float(df["correct_rank"].mean()),
    }


def compute_error_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the distribution of error categories.
    """
    order = [
        "Top-1 correct",
        "Minor error",
        "Medium error",
        "Severe error",
        "Not found",
    ]

    position_desc = {
        "Top-1 correct": "1",
        "Minor error": "2--3",
        "Medium error": "4--5",
        "Severe error": "6--10",
        "Not found": "--",
    }

    dist = (
        df["error_type"]
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("case_type")
        .reset_index(name="articles")
    )

    dist = dist[dist["articles"] > 0].copy()
    dist["correct_headline_position"] = dist["case_type"].map(position_desc)
    dist["percentage"] = 100 * dist["articles"] / len(df)

    return dist[
        ["case_type", "correct_headline_position", "articles", "percentage"]
    ]


def compute_rank_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the exact distribution of the correct-headline position.
    """
    rank_dist = (
        df["correct_rank"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("correct_rank")
        .reset_index(name="articles")
    )

    rank_dist["percentage"] = 100 * rank_dist["articles"] / len(df)

    return rank_dist


def compute_length_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse performance by article-length quartiles measured in words.
    """
    labels = ["Short", "Medium", "Long", "Very long"]

    data = df.copy()
    data["length_bin"] = pd.qcut(
        data["word_count"],
        q=4,
        labels=labels,
        duplicates="drop",
    )

    summary = (
        data.groupby("length_bin", observed=True)
        .agg(
            articles=("id", "count"),
            words_min=("word_count", "min"),
            words_max=("word_count", "max"),
            words_mean=("word_count", "mean"),
            top1=("top1_correct", "mean"),
            top3=("top3_correct", "mean"),
            top5=("top5_correct", "mean"),
            mean_position=("correct_rank", "mean"),
        )
        .reset_index()
    )

    summary["top1_pct"] = 100 * summary["top1"]
    summary["top3_pct"] = 100 * summary["top3"]
    summary["top5_pct"] = 100 * summary["top5"]

    return summary


def export_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save metrics, distributions and representative error cases.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_global_metrics(df)
    error_dist = compute_error_distribution(df)
    rank_dist = compute_rank_distribution(df)
    length_analysis = compute_length_analysis(df)

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    error_dist.to_csv(out_dir / "error_distribution.csv", index=False, encoding="utf-8")
    rank_dist.to_csv(out_dir / "rank_distribution.csv", index=False, encoding="utf-8")
    length_analysis.to_csv(out_dir / "length_analysis.csv", index=False, encoding="utf-8")

    keep_cols = [
        "id",
        "word_count",
        "error_type",
        "correct_rank",
        "correct_title_id",
        "predicted_top_id",
        "correct_title_text",
        "predicted_top_text",
        "y_true",
        "prediction",
        "article_excerpt",
    ]

    existing = [c for c in keep_cols if c in df.columns]

    error_cases = (
        df.loc[~df["top1_correct"], existing]
        .sort_values(["correct_rank", "word_count"])
    )
    error_cases.to_csv(out_dir / "error_cases.csv", index=False, encoding="utf-8")

    # Small sample for manual inspection: several cases per error category.
    representative = (
        error_cases.groupby("error_type", group_keys=False)
        .head(8)
        .reset_index(drop=True)
    )
    representative.to_csv(
        out_dir / "representative_error_candidates.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n=== Global metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n=== Error distribution ===")
    print(
        error_dist.to_string(
            index=False,
            formatters={"percentage": "{:.2f}".format},
        )
    )

    print("\n=== Length analysis ===")
    cols = [
        "length_bin",
        "articles",
        "words_min",
        "words_max",
        "words_mean",
        "top1_pct",
        "top3_pct",
        "mean_position",
    ]
    print(
        length_analysis[cols].to_string(
            index=False,
            formatters={
                "words_mean": "{:.1f}".format,
                "top1_pct": "{:.2f}".format,
                "top3_pct": "{:.2f}".format,
                "mean_position": "{:.2f}".format,
            },
        )
    )

    print(f"\nFiles saved to: {out_dir.resolve()}")


def main() -> None:
    """
    Parse command-line arguments and run the error analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyse PoliticHeadlinES ranking errors.",
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="CSV with id and prediction columns.",
    )
    parser.add_argument(
        "--test",
        required=True,
        type=Path,
        help="Test CSV with articles, candidate headlines and y_true.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("error_analysis_outputs"),
        help="Output directory.",
    )
    parser.add_argument(
        "--id_col",
        default="id",
        help="Identifier column name.",
    )
    parser.add_argument(
        "--prediction_col",
        default="prediction",
        help="Predicted ranking column name.",
    )
    parser.add_argument(
        "--truth_col",
        default="y_true",
        help="Reference ranking column name.",
    )
    parser.add_argument(
        "--article_col",
        default="article_body",
        help="Article body column name.",
    )
    parser.add_argument(
        "--title_prefix",
        default="title_",
        help="Candidate headline column prefix: title_1...title_10.",
    )

    args = parser.parse_args()

    df = build_cases(
        pred_path=args.pred,
        test_path=args.test,
        id_col=args.id_col,
        prediction_col=args.prediction_col,
        truth_col=args.truth_col,
        article_col=args.article_col,
        title_prefix=args.title_prefix,
    )
    export_outputs(df, args.out_dir)


if __name__ == "__main__":
    main()


