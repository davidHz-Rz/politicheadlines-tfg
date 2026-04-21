from __future__ import annotations

import json

import pandas as pd

from config import (
    ALPHA,
    TEST_CSV,
    IMAGES_DIR,
    NDCG_K,
    OUTPUTS_DIR,
    TRAIN_CSV,
)
from data_utils import validate_columns
from metrics import score_submission
from models.clip_ranker import load_clip, predict_task2_crossencoder_plus_clip
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker
from submission import build_submission, save_submission, validate_submission


MODEL_OUTPUT_DIR = OUTPUTS_DIR / "cross_encoder_model"
OUTPUT_SUBMISSION = OUTPUTS_DIR / "crossencoder_submission.csv"
OUTPUT_METRICS = OUTPUTS_DIR / "crossencoder_submission.metrics.json"


def main() -> None:
    print("Cargando datos...")
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(TEST_CSV)

    validate_columns(train_df)
    validate_columns(dev_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Dev/Test rows: {len(dev_df)}")

    config = CrossEncoderConfig(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        max_length=512,
        batch_size=4,
        learning_rate=2e-5,
        epochs=2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        use_amp=True,
    )

    print("Cargando cross-encoder entrenado...")
    ranker = CrossEncoderRanker.load(str(MODEL_OUTPUT_DIR), config)
    print(f"Dispositivo cross-encoder: {ranker.device}")

    print("Generando Task 1 (cross-encoder)...")
    task_1_preds = ranker.predict_dataframe(dev_df)

    print("Cargando CLIP...")
    clip_model, clip_processor, device = load_clip()
    print(f"Dispositivo CLIP: {device}")

    print("Generando Task 2 (cross-encoder + CLIP)...")
    task_2_preds = predict_task2_crossencoder_plus_clip(
        df_pred=dev_df,
        images_dir=IMAGES_DIR,
        cross_encoder_ranker=ranker,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
        w_text=0.96,
        w_img=0.04,
    )

    submission = build_submission(
        ids=dev_df["id"].astype(str),
        task_1_preds=task_1_preds,
        task_2_preds=task_2_preds,
    )

    validate_submission(submission)
    save_submission(submission, output_path=OUTPUT_SUBMISSION)

    print(submission.head())

    if "y_true" in dev_df.columns:
        print("Evaluando submission...")
        scores = score_submission(
            validation_csv=str(TEST_CSV),
            results_csv=str(OUTPUT_SUBMISSION),
            k=NDCG_K,
            alpha=ALPHA,
        )

        print("\nResultados:")
        for key, value in scores.items():
            print(f"{key}: {value}")

        with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        print(f"Métricas guardadas en: {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()