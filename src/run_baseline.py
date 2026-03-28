from __future__ import annotations

import json

import pandas as pd

from config import (
    ALPHA,
    DEV_CSV,
    IMAGES_DIR,
    NDCG_K,
    OUTPUT_METRICS,
    TRAIN_CSV,
)
from data_utils import validate_columns
from metrics import score_submission
from models.clip_ranker import load_clip, predict_task2_clip_plus_tfidf
from models.tfidf_ranker import predict_tfidf
from models.semantic_ranker import predict_semantic
from submission import build_submission, save_submission, validate_submission


def main() -> None:
    print("Cargando datos...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(DEV_CSV)

    validate_columns(train_df)
    validate_columns(test_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Dev/Test rows: {len(test_df)}")

    # ---------------------------------------------------------
    # Task 1:
    # ---------------------------------------------------------
    print("Generando predicciones para Task 1 (Semantic Ranker)...")
    task_1_preds = predict_semantic(
        df_pred=test_df,
    )

    # ---------------------------------------------------------
    # Task 2: CLIP + TF-IDF
    # ---------------------------------------------------------
    print("Cargando modelo CLIP...")
    clip_model, clip_processor, device = load_clip()
    print(f"Dispositivo: {device}")

    print("Generando predicciones para Task 2 (CLIP + TF-IDF)...")
    task_2_preds = predict_task2_clip_plus_tfidf(
        df_train=train_df,
        df_pred=test_df,
        images_dir=IMAGES_DIR,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
    )

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    print("Construyendo submission...")
    submission = build_submission(
        ids=test_df["id"].astype(str),
        task_1_preds=task_1_preds,
        task_2_preds=task_2_preds,
    )

    validate_submission(submission)
    save_submission(submission)

    print(submission.head())

    # ---------------------------------------------------------
    # Evaluación local (si hay y_true)
    # ---------------------------------------------------------
    if "y_true" in test_df.columns:
        print("Evaluando submission...")
        scores = score_submission(
            validation_csv=str(DEV_CSV),
            results_csv="outputs/results.csv",
            k=NDCG_K,
            alpha=ALPHA,
        )

        print("\nResultados:")
        for key, value in scores.items():
            print(f"{key}: {value}")

        with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        print(f"Métricas guardadas en: {OUTPUT_METRICS}")
    else:
        print("No se encontró columna 'y_true'. Se omite la evaluación local.")


if __name__ == "__main__":
    main()