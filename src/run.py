from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (
    ALPHA,
    IMAGE_WEIGHT,
    IMAGES_DIR,
    MODEL_NAME,
    NDCG_K,
    OUTPUT_METRICS,
    OUTPUT_SUBMISSION,
    TEST_CSV,
    TEXT_WEIGHT,
    TRAIN_CSV,
    USE_CLIP_FOR_TASK2,
    get_cross_encoder_runtime_config,
)
from utils.data_utils import validate_columns
from utils.metrics import score_submission
from utils.submission import build_submission, save_submission, validate_submission

from models.clip_ranker import load_clip, predict_task2_clip_plus_tfidf, predict_task2_semantic_plus_clip, predict_task2_crossencoder_plus_clip
from models.tfidf_ranker import predict_tfidf
from models.semantic_ranker import predict_semantic, SemanticRanker
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker


def build_crossencoder_ranker(model_key: str) -> CrossEncoderRanker:
    cfg = get_cross_encoder_runtime_config(model_key)
    config = CrossEncoderConfig(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        use_amp=cfg["use_amp"],
    )
    return CrossEncoderRanker.load(str(cfg["model_dir"]), config)


def main() -> None:
    print("Cargando datos...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    validate_columns(train_df)
    validate_columns(test_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Modelo: {MODEL_NAME}")

    # ---------------------------------------------------------
    # Task 1
    # ---------------------------------------------------------
    if MODEL_NAME == "tfidf":
        print("Generando predicciones para Task 1 (TF-IDF)...")
        task_1_preds = predict_tfidf(
            df_train=train_df,
            df_pred=test_df,
        )

    elif MODEL_NAME == "semantic":
        print("Cargando Semantic Ranker...")
        semantic_ranker = SemanticRanker()
    
        print("Generando predicciones para Task 1 (Semantic Ranker)...")
        task_1_preds = []
        for _, row in test_df.iterrows():
            article = str(row.get("article_body", "") or "")
            titles = [str(row.get(f"title_{i}", "") or "") for i in range(1, 11)]
            ranked = semantic_ranker.rank_titles(article, titles)
            task_1_preds.append(" ".join(ranked))

    elif MODEL_NAME == "bert":
        print("Cargando BERT entrenado...")
        ranker = build_crossencoder_ranker("bert")
        print(f"Dispositivo BERT: {ranker.device}")

        print("Generando predicciones para Task 1 (BERT)...")
        task_1_preds = ranker.predict_dataframe(test_df)

    elif MODEL_NAME == "bertin":
        print("Cargando BERTIN entrenado...")
        ranker = build_crossencoder_ranker("bertin")
        print(f"Dispositivo BERTIN: {ranker.device}")

        print("Generando predicciones para Task 1 (BERTIN)...")
        task_1_preds = ranker.predict_dataframe(test_df)

    elif MODEL_NAME == "mdeberta":
        print("Cargando mDeBERTa entrenado...")
        ranker = build_crossencoder_ranker("mdeberta")
        print(f"Dispositivo mDeBERTa: {ranker.device}")

        print("Generando predicciones para Task 1 (mDeBERTa)...")
        task_1_preds = ranker.predict_dataframe(test_df)

    else:
        raise ValueError(f"MODEL_NAME no soportado: {MODEL_NAME}")

    # ---------------------------------------------------------
    # Task 2
    # ---------------------------------------------------------
    if USE_CLIP_FOR_TASK2:
        print("Cargando modelo CLIP...")
        clip_model, clip_processor, device = load_clip()
        print(f"Dispositivo CLIP: {device}")

        if MODEL_NAME == "tfidf":
            print("Generando predicciones para Task 2 (CLIP + TF-IDF)...")
            task_2_preds = predict_task2_clip_plus_tfidf(
                df_train=train_df,
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
            )

        elif MODEL_NAME == "semantic":
            print("Generando predicciones para Task 2 (Semantic + CLIP)...")
            task_2_preds = predict_task2_semantic_plus_clip(
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                semantic_ranker=semantic_ranker,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                w_text=TEXT_WEIGHT,
                w_img=IMAGE_WEIGHT,
            )
            
        elif MODEL_NAME == "bert":
            print("Generando predicciones para Task 2 (BERT + CLIP)...")
            task_2_preds = predict_task2_crossencoder_plus_clip(
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                cross_encoder_ranker=ranker,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                w_text=TEXT_WEIGHT,
                w_img=IMAGE_WEIGHT,
            )

        elif MODEL_NAME == "bertin":
            print("Generando predicciones para Task 2 (BERTIN + CLIP)...")
            task_2_preds = predict_task2_crossencoder_plus_clip(
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                cross_encoder_ranker=ranker,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                w_text=TEXT_WEIGHT,
                w_img=IMAGE_WEIGHT,
            )

        elif MODEL_NAME == "mdeberta":
            print("Generando predicciones para Task 2 (mDeBERTa + CLIP)...")
            task_2_preds = predict_task2_crossencoder_plus_clip(
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                cross_encoder_ranker=ranker,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                w_text=TEXT_WEIGHT,
                w_img=IMAGE_WEIGHT,
            )
        else:
            raise ValueError(f"MODEL_NAME no soportado: {MODEL_NAME}")

    else:
        print("USE_CLIP_FOR_TASK2 = False. Usando Task 1 como Task 2.")
        task_2_preds = task_1_preds

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

    OUTPUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    save_submission(submission, output_path=OUTPUT_SUBMISSION)

    print(submission.head())

    # ---------------------------------------------------------
    # Evaluación local
    # ---------------------------------------------------------
    if "y_true" in test_df.columns:
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
    else:
        print("No se encontró columna 'y_true'. Se omite la evaluación local.")


if __name__ == "__main__":
    main()