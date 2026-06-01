from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import numpy as np
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
    CROSS_ENCODER_ENSEMBLE_MEMBERS,
    IMAGE_WEIGHT,
    IMAGES_DIR,
    LLM_BASE_RANKER,
    LLM_BASE_WEIGHT,
    LLM_DO_SAMPLE,
    LLM_LOAD_IN_4BIT,
    LLM_MAX_INPUT_CHARS,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_RANKER_MODE,
    LLM_RERANK_TOP_K,
    LLM_TEMPERATURE,
    LLM_TORCH_DTYPE,
    LLM_TRUST_REMOTE_CODE,
    LLM_WEIGHT,
    MODEL_NAME,
    MODERN_RERANKER_BASE_RANKER,
    MODERN_RERANKER_BASE_WEIGHT,
    MODERN_RERANKER_MODE,
    MODERN_RERANKER_MODEL_KEY,
    MODERN_RERANKER_TOP_K,
    MODERN_RERANKER_WEIGHT,
    TAIL_RERANKER_BASE_RANKER,
    TAIL_RERANKER_AUX_RANKER,
    TAIL_RERANKER_TOP_K,
    NDCG_K,
    OUTPUT_METRICS,
    OUTPUT_SUBMISSION,
    TEST_CSV,
    TEXT_WEIGHT,
    TRAIN_CSV,
    USE_VLM_FOR_TASK2,
    VLM_BACKEND,
    get_cross_encoder_runtime_config,
    get_cross_encoder_rank10_runtime_config,
    get_modern_reranker_runtime_config,
    get_vlm_model_name,
)
from utils.data_utils import get_source_text_task1, get_titles, validate_columns
from utils.metrics import score_submission
from utils.submission import build_submission, save_submission, validate_submission

from models.vlm_ranker import (
    load_vlm,
    predict_task2_vlm_plus_text_scores,
    ranking_from_scores,
)
from models.tfidf_ranker import TfidfRanker, build_tfidf_corpus
from models.bm25_ranker import BM25Ranker, build_bm25_corpus
from models.semantic_ranker import SemanticRanker
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker
from models.cross_encoder_rank10_ranker import CrossEncoderRank10Config, CrossEncoderRank10Ranker
from models.cross_encoder_ensemble_ranker import CrossEncoderEnsembleRanker, EnsembleMember
from models.llm_ranker import LLMEnsembleRanker, LLMRanker, LLMRankerConfig
from models.modern_reranker import ModernReranker, ModernRerankerPipeline
from models.tail_reranker import TailReranker


def build_crossencoder_ranker(model_key: str) -> CrossEncoderRanker:
    cfg = get_cross_encoder_runtime_config(model_key)
    config = CrossEncoderConfig(
        model_name=str(cfg["model_name"]),
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        use_amp=cfg["use_amp"],
        use_head_tail=cfg.get("use_head_tail", False),
        head_tokens=cfg.get("head_tokens", 384),
        tail_tokens=cfg.get("tail_tokens", 125),
    )
    return CrossEncoderRanker.load(str(cfg["model_dir"]), config)


def build_crossencoder_rank10_ranker(model_key: str = "bert_rank10") -> CrossEncoderRank10Ranker:
    cfg = get_cross_encoder_rank10_runtime_config(model_key)
    config = CrossEncoderRank10Config(
        model_name=str(cfg["model_name"]),
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        use_amp=cfg["use_amp"],
    )
    return CrossEncoderRank10Ranker.load(str(cfg["model_dir"]), config)


def build_crossencoder_ensemble_ranker() -> CrossEncoderEnsembleRanker:
    members = [
        EnsembleMember(model_key=name, weight=weight)
        for name, weight in CROSS_ENCODER_ENSEMBLE_MEMBERS
    ]
    return CrossEncoderEnsembleRanker(members)


def build_base_ranker(model_key: str, train_df: pd.DataFrame):
    """Construye un ranker textual base con interfaz score_titles(article, titles)."""
    model_key = model_key.lower().strip()

    if model_key == "tfidf":
        print("Cargando TF-IDF...")
        ranker = TfidfRanker()
        ranker.fit(build_tfidf_corpus(train_df))
        return ranker

    if model_key == "bm25":
        print("Cargando BM25...")
        ranker = BM25Ranker(
            k1=BM25_K1,
            b=BM25_B,
            query_term_limit=BM25_QUERY_TERM_LIMIT,
        )
        ranker.fit(build_bm25_corpus(train_df))
        return ranker

    if model_key == "semantic":
        print("Cargando Semantic Ranker...")
        return SemanticRanker()

    if model_key == "crossencoder_ensemble":
        print("Cargando ensemble de cross-encoders...")
        print("Miembros ensemble:", CROSS_ENCODER_ENSEMBLE_MEMBERS)
        return build_crossencoder_ensemble_ranker()

    if model_key in {"bert", "bert_headtail", "bertin", "mdeberta"}:
        print(f"Cargando cross-encoder entrenado: {model_key}...")
        ranker = build_crossencoder_ranker(model_key)
        print(f"Dispositivo {model_key}: {ranker.device}")
        return ranker

    if model_key == "bert_rank10":
        print("Cargando cross-encoder rank10 entrenado...")
        ranker = build_crossencoder_rank10_ranker("bert_rank10")
        print(f"Dispositivo bert_rank10: {ranker.device}")
        return ranker

    raise ValueError(f"Ranker base no soportado: {model_key}")


def build_llm_ranker(train_df: pd.DataFrame):
    print(f"Cargando LLM ranker: {LLM_MODEL_NAME}")
    print(f"Modo LLM: {LLM_RANKER_MODE}")

    llm_config = LLMRankerConfig(
        model_name=LLM_MODEL_NAME,
        max_input_chars=LLM_MAX_INPUT_CHARS,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        do_sample=LLM_DO_SAMPLE,
        load_in_4bit=LLM_LOAD_IN_4BIT,
        torch_dtype=LLM_TORCH_DTYPE,
        trust_remote_code=LLM_TRUST_REMOTE_CODE,
    )
    llm_ranker = LLMRanker(llm_config)

    if LLM_RANKER_MODE.lower().strip() == "solo":
        return LLMEnsembleRanker(base_ranker=None, llm_ranker=llm_ranker, mode="solo")

    base_ranker = build_base_ranker(LLM_BASE_RANKER, train_df=train_df)
    return LLMEnsembleRanker(
        base_ranker=base_ranker,
        llm_ranker=llm_ranker,
        mode=LLM_RANKER_MODE,
        base_weight=LLM_BASE_WEIGHT,
        llm_weight=LLM_WEIGHT,
        rerank_top_k=LLM_RERANK_TOP_K,
    )


def build_modern_reranker(train_df: pd.DataFrame):
    cfg = get_modern_reranker_runtime_config(MODERN_RERANKER_MODEL_KEY)
    print(f"Cargando reranker moderno: {MODERN_RERANKER_MODEL_KEY} ({cfg['model_name']})")
    print(f"Modo reranker moderno: {MODERN_RERANKER_MODE}")

    reranker = ModernReranker(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        use_fp16=cfg.get("use_fp16", True),
    )
    print(f"Dispositivo reranker moderno: {reranker.device}")

    if MODERN_RERANKER_MODE.lower().strip() == "solo":
        return ModernRerankerPipeline(
            reranker=reranker,
            base_ranker=None,
            mode="solo",
        )

    base_ranker = build_base_ranker(MODERN_RERANKER_BASE_RANKER, train_df=train_df)

    return ModernRerankerPipeline(
        reranker=reranker,
        base_ranker=base_ranker,
        mode=MODERN_RERANKER_MODE,
        base_weight=MODERN_RERANKER_BASE_WEIGHT,
        reranker_weight=MODERN_RERANKER_WEIGHT,
        rerank_top_k=MODERN_RERANKER_TOP_K,
    )


def build_tail_reranker(train_df: pd.DataFrame):
    print("Cargando tail reranker...")
    base_ranker = build_base_ranker(TAIL_RERANKER_BASE_RANKER, train_df=train_df)

    aux = TAIL_RERANKER_AUX_RANKER.lower().strip()
    if aux == "bge":
        cfg = get_modern_reranker_runtime_config(MODERN_RERANKER_MODEL_KEY)
        tail_ranker = ModernReranker(
            model_name=cfg["model_name"],
            max_length=cfg["max_length"],
            batch_size=cfg["batch_size"],
            use_fp16=cfg.get("use_fp16", True),
        )
    else:
        tail_ranker = build_base_ranker(aux, train_df=train_df)

    return TailReranker(base_ranker, tail_ranker, top_k=TAIL_RERANKER_TOP_K)

def build_ranker(model_name: str, train_df: pd.DataFrame):
    """Construye el ranker textual principal seleccionado en config.py."""
    model_name = model_name.lower().strip()

    if model_name == "llm_ranker":
        return build_llm_ranker(train_df=train_df)

    if model_name == "modern_reranker":
        return build_modern_reranker(train_df=train_df)

    if model_name == "tail_reranker":
        return build_tail_reranker(train_df=train_df)

    return build_base_ranker(model_name, train_df=train_df)


def score_dataframe_with_ranker(
    ranker,
    df_pred: pd.DataFrame,
    progress_every: int = 10,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Calcula scores textuales una sola vez.

    Devuelve:
    - preds: rankings para Task 1.
    - scores_cache: scores por fila para reutilizar en Task 2.
    """
    preds: List[str] = []
    scores_cache: List[np.ndarray] = []

    for idx, (_, row) in enumerate(df_pred.iterrows(), start=1):
        article = get_source_text_task1(row)
        titles = get_titles(row)
        scores = np.asarray(ranker.score_titles(article, titles), dtype=float)

        if len(scores) != len(titles):
            raise ValueError(
                f"El ranker devolvió {len(scores)} scores para {len(titles)} títulos "
                f"en la fila {idx}."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"El ranker devolvió scores no finitos en la fila {idx}.")

        scores_cache.append(scores)
        preds.append(ranking_from_scores(scores))

        if progress_every and idx % progress_every == 0:
            print(f"Predichas Task 1 {idx}/{len(df_pred)} filas...")

    return preds, scores_cache


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
    # Ranker textual: se calcula una vez y se reutiliza.
    # ---------------------------------------------------------
    ranker = build_ranker(MODEL_NAME, train_df=train_df)

    print(f"Generando predicciones para Task 1 ({MODEL_NAME})...")
    task_1_preds, text_scores_cache = score_dataframe_with_ranker(ranker, test_df)

    # ---------------------------------------------------------
    # Task 2: si hay VLM, solo calcula señal visual y fusiona con
    # scores textuales cacheados. No recalcula el ranker textual.
    # ---------------------------------------------------------
    if USE_VLM_FOR_TASK2:
        vlm_name = get_vlm_model_name()
        print(f"Cargando modelo {VLM_BACKEND.upper()}: {vlm_name}...")
        vlm_model, vlm_processor, device = load_vlm()
        print(f"Dispositivo {VLM_BACKEND.upper()}: {device}")

        print(
            f"Generando predicciones para Task 2 "
            f"({MODEL_NAME} cacheado + {VLM_BACKEND.upper()})..."
        )
        task_2_preds = predict_task2_vlm_plus_text_scores(
            df_pred=test_df,
            images_dir=IMAGES_DIR,
            text_scores_list=text_scores_cache,
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            backend=VLM_BACKEND,
            device=device,
            w_text=TEXT_WEIGHT,
            w_img=IMAGE_WEIGHT,
        )
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
