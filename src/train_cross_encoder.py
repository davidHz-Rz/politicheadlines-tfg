from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    ALPHA,
    DEV_CSV,
    NDCG_K,
    OUTPUTS_DIR,
    TRAIN_CSV,
)
from data_utils import validate_columns
from metrics import score_submission
from models.cross_encoder_ranker import (
    CrossEncoderConfig,
    CrossEncoderRanker,
    build_pair_examples,
)
from submission import build_submission, save_submission, validate_submission


CROSS_ENCODER_OUTPUT = OUTPUTS_DIR / "cross_encoder_results.csv"
CROSS_ENCODER_METRICS = OUTPUTS_DIR / "cross_encoder_results.metrics.json"


def main() -> None:
    print("Cargando datos...")
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)

    validate_columns(train_df)
    validate_columns(dev_df)

    if "y_true" not in train_df.columns:
        raise ValueError("TRAIN_CSV debe contener la columna 'y_true' para entrenar.")
    if "y_true" not in dev_df.columns:
        raise ValueError("DEV_CSV debe contener la columna 'y_true' para evaluar.")

    print(f"Train rows: {len(train_df)}")
    print(f"Dev rows: {len(dev_df)}")

    print("Construyendo pares de entrenamiento...")
    train_examples = build_pair_examples(train_df)
    dev_examples = build_pair_examples(dev_df)

    print(f"Número de pares train: {len(train_examples)}")
    print(f"Número de pares dev: {len(dev_examples)}")

    config = CrossEncoderConfig(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        max_length=512,
        batch_size=4,
        learning_rate=2e-5,
        epochs=2,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )

    print("Inicializando modelo...")
    ranker = CrossEncoderRanker(config)
    print(f"Dispositivo: {ranker.device}")

    print("Entrenando modelo...")
    ranker.fit(train_examples=train_examples, val_examples=dev_examples)

    print("Generando predicciones en dev...")
    task_1_preds = ranker.predict_dataframe(dev_df)

    # De momento duplicamos task_1 en task_2 solo para evaluar el modelo textual
    submission = build_submission(
        ids=dev_df["id"].astype(str),
        task_1_preds=task_1_preds,
        task_2_preds=task_1_preds,
    )

    validate_submission(submission)
    save_submission(submission, output_path=CROSS_ENCODER_OUTPUT)

    print(submission.head())

    print("Evaluando submission...")
    scores = score_submission(
        validation_csv=str(DEV_CSV),
        results_csv=str(CROSS_ENCODER_OUTPUT),
        k=NDCG_K,
        alpha=ALPHA,
    )

    print("\nResultados:")
    for key, value in scores.items():
        print(f"{key}: {value}")

    with open(CROSS_ENCODER_METRICS, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"Métricas guardadas en: {CROSS_ENCODER_METRICS}")


if __name__ == "__main__":
    main()