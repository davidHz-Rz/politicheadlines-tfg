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
    BM25_B,
    BM25_K1,
    BM25_QUERY_TERM_LIMIT,
    IMAGE_WEIGHT,
    IMAGES_DIR,
    MODEL_NAME,
    NDCG_K,
    OUTPUT_METRICS,
    OUTPUT_SUBMISSION,
    TEST_CSV,
    TEXT_WEIGHT,
    TRAIN_CSV,
    USE_VLM_FOR_TASK2,
    VLM_BACKEND,
    get_cross_encoder_runtime_config,
    get_vlm_model_name,
    CROSS_ENCODER_ENSEMBLE_MEMBERS
)
from utils.data_utils import validate_columns
from utils.metrics import score_submission
from utils.submission import build_submission, save_submission, validate_submission

from models.vlm_ranker import load_vlm, predict_task2_vlm_plus_tfidf, predict_task2_vlm_plus_bm25, predict_task2_semantic_plus_vlm, predict_task2_crossencoder_plus_vlm
from models.tfidf_ranker import predict_tfidf
from models.bm25_ranker import BM25Ranker, build_bm25_corpus
from models.semantic_ranker import SemanticRanker
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker
from models.cross_encoder_ensemble_ranker import CrossEncoderEnsembleRanker, EnsembleMember

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
        task_1_preds = predict_tfidf(df_train=train_df, df_pred=test_df)

    elif MODEL_NAME == "bm25":
        print("Cargando BM25...")
        bm25_ranker = BM25Ranker(
            k1=BM25_K1,
            b=BM25_B,
            query_term_limit=BM25_QUERY_TERM_LIMIT,
        )
        bm25_ranker.fit(build_bm25_corpus(train_df))

        print("Generando predicciones para Task 1 (BM25)...")
        task_1_preds = []
        for _, row in test_df.iterrows():
            article = str(row.get("article_body", "") or "")
            titles = [str(row.get(f"title_{i}", "") or "") for i in range(1, 11)]
            task_1_preds.append(" ".join(bm25_ranker.rank_titles(article, titles)))

    elif MODEL_NAME == "semantic":
        print("Cargando Semantic Ranker...")
        semantic_ranker = SemanticRanker()

        print("Generando predicciones para Task 1 (Semantic Ranker)...")
        task_1_preds = []
        for _, row in test_df.iterrows():
            article = str(row.get("article_body", "") or "")
            titles = [str(row.get(f"title_{i}", "") or "") for i in range(1, 11)]
            task_1_preds.append(" ".join(semantic_ranker.rank_titles(article, titles)))

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
        
    elif MODEL_NAME == "crossencoder_ensemble":
        print("Cargando ensemble de cross-encoders...")
        members = [
            EnsembleMember(model_key=name, weight=weight)
            for name, weight in CROSS_ENCODER_ENSEMBLE_MEMBERS
        ]
        ranker = CrossEncoderEnsembleRanker(members)
        print("Miembros ensemble:", CROSS_ENCODER_ENSEMBLE_MEMBERS)
        print("Generando predicciones para Task 1...")
        task_1_preds = ranker.predict_dataframe(test_df)

    else:
        raise ValueError(f"MODEL_NAME no soportado: {MODEL_NAME}")

    # ---------------------------------------------------------
    # Task 2
    # ---------------------------------------------------------
    if USE_VLM_FOR_TASK2:
        vlm_name = get_vlm_model_name()
        print(f"Cargando modelo {VLM_BACKEND.upper()}: {vlm_name}...")
        vlm_model, vlm_processor, device = load_vlm()
        print(f"Dispositivo {VLM_BACKEND.upper()}: {device}")

        if MODEL_NAME == "tfidf":
            print(f"Generando predicciones para Task 2 ({VLM_BACKEND.upper()} + TF-IDF)...")
            task_2_preds = predict_task2_vlm_plus_tfidf(
                df_train=train_df,
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                backend=VLM_BACKEND,
                device=device,
            )

        elif MODEL_NAME == "bm25":
            print(f"Generando predicciones para Task 2 (BM25 + {VLM_BACKEND.upper()})...")
            task_2_preds = predict_task2_vlm_plus_bm25(
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                bm25_ranker=bm25_ranker,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                backend=VLM_BACKEND,
                device=device,
                w_text=TEXT_WEIGHT,
                w_img=IMAGE_WEIGHT,
            )

        elif MODEL_NAME == "semantic":
            print(f"Generando predicciones para Task 2 (Semantic + {VLM_BACKEND.upper()})...")
            task_2_preds = predict_task2_semantic_plus_vlm(
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                semantic_ranker=semantic_ranker,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                backend=VLM_BACKEND,
                device=device,
                w_text=TEXT_WEIGHT,
                w_img=IMAGE_WEIGHT,
            )

        elif MODEL_NAME in {"bert", "bertin", "mdeberta", "crossencoder_ensemble"}:
            print(f"Generando predicciones para Task 2 ({MODEL_NAME} + {VLM_BACKEND.upper()})...")
            task_2_preds = predict_task2_crossencoder_plus_vlm(
                df_pred=test_df,
                images_dir=IMAGES_DIR,
                cross_encoder_ranker=ranker,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                backend=VLM_BACKEND,
                device=device,
                w_text=TEXT_WEIGHT,
                w_img=IMAGE_WEIGHT,
            )
        else:
            raise ValueError(f"MODEL_NAME no soportado: {MODEL_NAME}")

    else:
        print("USE_VLM_FOR_TASK2 = False. Usando Task 1 como Task 2.")
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
