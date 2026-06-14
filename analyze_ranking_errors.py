#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de errores para PoliticHeadlinES-IberLEF 2026.

El script compara un fichero de predicciones con el test de la partición experimental.
Calcula métricas globales, distribución de errores según la posición del titular correcto,
análisis por longitud del artículo y exporta casos de error para inspección manual.

Uso recomendado:
    python analyze_ranking_errors.py \
        --pred "base_ensemble(bert_headtail_0.40_bert_0.45_mdeberta_0.15)_tail_rank10.csv" \
        --test test.csv \
        --out_dir error_analysis_outputs

Formato esperado:
    Predicciones CSV:
        id,prediction
        <id>,"t3 t9 t6 ..."

    Test CSV:
        id,article_body,title_1,...,title_10,y_true
        <id>,...,"t3 t7 t9 ..."
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def parse_ranking(value: object) -> List[str]:
    """Convierte un ranking en lista de identificadores tipo ['t3', 't9', ...]."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    # Permite formatos simples: "t3 t9 t6" o con comas/corchetes/comillas.
    items = re.findall(r"t(?:10|[1-9])", text)
    return items


def count_words(text: object) -> int:
    """Cuenta palabras de forma simple mediante separación por espacios."""
    if pd.isna(text):
        return 0
    return len(str(text).split())


def title_text(row: pd.Series, title_id: str, title_prefix: str = "title_") -> str:
    """Devuelve el texto de un titular a partir de su identificador t1..t10."""
    if not title_id or not re.fullmatch(r"t(?:10|[1-9])", title_id):
        return ""
    idx = title_id[1:]
    col = f"{title_prefix}{idx}"
    return str(row.get(col, ""))


def error_bucket(rank: Optional[int]) -> str:
    """Clasifica el caso según la posición del titular correcto."""
    if rank is None:
        return "No encontrado"
    if rank == 1:
        return "Top-1 correcto"
    if 2 <= rank <= 3:
        return "Fallo leve"
    if 4 <= rank <= 5:
        return "Fallo medio"
    if 6 <= rank <= 10:
        return "Fallo grave"
    return "No encontrado"


def validate_rankings(df: pd.DataFrame, pred_col: str, truth_col: str) -> None:
    """Comprueba que los rankings tienen diez titulares y no repiten candidatos."""
    for col in [pred_col, truth_col]:
        lengths = df[col].apply(len)
        unique_lengths = df[col].apply(lambda x: len(set(x)))
        invalid_length = (lengths != 10).sum()
        invalid_unique = (unique_lengths != 10).sum()
        if invalid_length or invalid_unique:
            print(f"[AVISO] Columna {col}: {invalid_length} rankings sin 10 elementos, "
                  f"{invalid_unique} rankings con repetidos.")


def build_cases(
    pred_path: Path,
    test_path: Path,
    id_col: str,
    prediction_col: str,
    truth_col: str,
    article_col: str,
    title_prefix: str,
) -> pd.DataFrame:
    """Carga, une y calcula información por noticia."""
    pred = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)

    required_pred = {id_col, prediction_col}
    required_test = {id_col, truth_col, article_col}
    missing_pred = required_pred - set(pred.columns)
    missing_test = required_test - set(test.columns)
    if missing_pred:
        raise ValueError(f"Faltan columnas en predicciones: {sorted(missing_pred)}")
    if missing_test:
        raise ValueError(f"Faltan columnas en test: {sorted(missing_test)}")

    df = test.merge(pred[[id_col, prediction_col]], on=id_col, how="inner")
    if len(df) != len(test) or len(df) != len(pred):
        print(f"[AVISO] Filas test={len(test)}, pred={len(pred)}, comunes={len(df)}")

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

    df["correct_title_text"] = df.apply(lambda r: title_text(r, r["correct_title_id"], title_prefix), axis=1)
    df["predicted_top_text"] = df.apply(lambda r: title_text(r, r["predicted_top_id"], title_prefix), axis=1)
    df["article_excerpt"] = df[article_col].fillna("").astype(str).str.slice(0, 350)

    return df


def compute_global_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas globales del ranking."""
    return {
        "n_rows": int(len(df)),
        "top1_accuracy": float(df["top1_correct"].mean()),
        "top3_hit_rate": float(df["top3_correct"].mean()),
        "top5_hit_rate": float(df["top5_correct"].mean()),
        "exact_ranking_matches": int(df["exact_ranking"].sum()),
        "mean_correct_rank": float(df["correct_rank"].mean()),
    }


def compute_error_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de errores por gravedad."""
    order = ["Top-1 correcto", "Fallo leve", "Fallo medio", "Fallo grave", "No encontrado"]
    position_desc = {
        "Top-1 correcto": "1",
        "Fallo leve": "2--3",
        "Fallo medio": "4--5",
        "Fallo grave": "6--10",
        "No encontrado": "--",
    }
    dist = (
        df["error_type"]
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("tipo_caso")
        .reset_index(name="noticias")
    )
    dist = dist[dist["noticias"] > 0].copy()
    dist["posicion_titular_correcto"] = dist["tipo_caso"].map(position_desc)
    dist["porcentaje"] = 100 * dist["noticias"] / len(df)
    return dist[["tipo_caso", "posicion_titular_correcto", "noticias", "porcentaje"]]


def compute_rank_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución exacta por posición del titular correcto."""
    rank_dist = (
        df["correct_rank"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("correct_rank")
        .reset_index(name="noticias")
    )
    rank_dist["porcentaje"] = 100 * rank_dist["noticias"] / len(df)
    return rank_dist


def compute_length_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis por cuartiles de longitud del artículo medidos en palabras."""
    labels = ["Corto", "Medio", "Largo", "Muy largo"]
    data = df.copy()
    data["length_bin"] = pd.qcut(data["word_count"], q=4, labels=labels, duplicates="drop")

    summary = (
        data.groupby("length_bin", observed=True)
        .agg(
            noticias=("id", "count"),
            palabras_min=("word_count", "min"),
            palabras_max=("word_count", "max"),
            palabras_media=("word_count", "mean"),
            top1=("top1_correct", "mean"),
            top3=("top3_correct", "mean"),
            top5=("top5_correct", "mean"),
            posicion_media=("correct_rank", "mean"),
        )
        .reset_index()
    )
    summary["top1_pct"] = 100 * summary["top1"]
    summary["top3_pct"] = 100 * summary["top3"]
    summary["top5_pct"] = 100 * summary["top5"]
    return summary


def export_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    """Guarda métricas y casos de error en CSV/JSON."""
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
        "id", "word_count", "error_type", "correct_rank",
        "correct_title_id", "predicted_top_id",
        "correct_title_text", "predicted_top_text",
        "y_true", "prediction", "article_excerpt",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    error_cases = df.loc[~df["top1_correct"], existing].sort_values(["correct_rank", "word_count"])
    error_cases.to_csv(out_dir / "error_cases.csv", index=False, encoding="utf-8")

    # Muestra pequeña para inspección manual: varios casos por tipo de error.
    representative = (
        error_cases.groupby("error_type", group_keys=False)
        .head(8)
        .reset_index(drop=True)
    )
    representative.to_csv(out_dir / "representative_error_candidates.csv", index=False, encoding="utf-8")

    print("\n=== Métricas globales ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n=== Distribución de errores ===")
    print(error_dist.to_string(index=False, formatters={"porcentaje": "{:.2f}".format}))

    print("\n=== Análisis por longitud ===")
    cols = ["length_bin", "noticias", "palabras_min", "palabras_max", "palabras_media", "top1_pct", "top3_pct", "posicion_media"]
    print(length_analysis[cols].to_string(index=False, formatters={
        "palabras_media": "{:.1f}".format,
        "top1_pct": "{:.2f}".format,
        "top3_pct": "{:.2f}".format,
        "posicion_media": "{:.2f}".format,
    }))

    print(f"\nFicheros guardados en: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza errores de rankings PoliticHeadlinES.")
    parser.add_argument("--pred", required=True, type=Path, help="CSV con columnas id y prediction.")
    parser.add_argument("--test", required=True, type=Path, help="CSV de test con artículos, titulares y y_true.")
    parser.add_argument("--out_dir", type=Path, default=Path("error_analysis_outputs"), help="Directorio de salida.")
    parser.add_argument("--id_col", default="id", help="Nombre de la columna de identificador.")
    parser.add_argument("--prediction_col", default="prediction", help="Nombre de la columna de ranking predicho.")
    parser.add_argument("--truth_col", default="y_true", help="Nombre de la columna de ranking real.")
    parser.add_argument("--article_col", default="article_body", help="Nombre de la columna del artículo.")
    parser.add_argument("--title_prefix", default="title_", help="Prefijo de columnas de titulares: title_1...title_10.")
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
