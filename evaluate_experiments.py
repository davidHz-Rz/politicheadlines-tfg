from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (  # noqa: E402
    ALPHA,
    BM25_B,
    BM25_K1,
    BM25_QUERY_TERM_LIMIT,
    CLIP_MODEL_NAME,
    IMAGES_DIR,
    LLM_DO_SAMPLE,
    LLM_LOAD_IN_4BIT,
    LLM_MAX_INPUT_CHARS,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TORCH_DTYPE,
    LLM_TRUST_REMOTE_CODE,
    MODERN_RERANKER_CONFIGS,
    NDCG_K,
    OUTPUTS_DIR,
    SIGLIP_MODEL_NAME,
    TEST_CSV,
    TOKENS_ALL,
    TRAIN_CSV,
)
from models.bm25_ranker import BM25Ranker, build_bm25_corpus  # noqa: E402
from models.cross_encoder_ensemble_ranker import (  # noqa: E402
    CrossEncoderEnsembleRanker,
    EnsembleMember,
)
from models.cross_encoder_rank10_ranker import (  # noqa: E402
    CrossEncoderRank10Config,
    CrossEncoderRank10Ranker,
)
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker  # noqa: E402
from models.llm_ranker import LLMEnsembleRanker, LLMRanker, LLMRankerConfig  # noqa: E402
from models.modern_reranker import ModernReranker, ModernRerankerPipeline  # noqa: E402
from models.semantic_ranker import SemanticRanker  # noqa: E402
from models.tail_reranker import TailReranker  # noqa: E402
from models.tfidf_ranker import TfidfRanker, build_tfidf_corpus  # noqa: E402
from models.vlm_ranker import (  # noqa: E402
    load_vlm,
    predict_task2_vlm,
    predict_task2_vlm_plus_text_scores,
    ranking_from_scores,
)
from run import (  # noqa: E402
    build_crossencoder_rank10_ranker,
    build_crossencoder_ranker,
)
from utils.data_utils import get_source_text_task1, get_titles, validate_columns  # noqa: E402
from utils.metrics import score_task1_predictions_df  # noqa: E402


EVAL_DIR = OUTPUTS_DIR / "evaluation"

INDIVIDUAL_MODELS = [
    "tfidf",
    "bm25",
    "semantic",
    "bert",
    "bert_headtail",
    "bertin",
    "mdeberta",
    "bert_rank10",
    "bge",
]

# Edita esta lista si quieres probar otros LLMs. Llama puede requerir HF_TOKEN y acceso aceptado.
LLM_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

VLM_MODELS = [
    ("clip", CLIP_MODEL_NAME),
    ("siglip", "google/siglip-base-patch16-224"),
    ("siglip", SIGLIP_MODEL_NAME),
]

PAIR_WEIGHTS = [
    (0.50, 0.50),
    (0.60, 0.40),
    (0.70, 0.30),
    (0.80, 0.20),
    (0.40, 0.60),
    (0.30, 0.70),
    (0.20, 0.80),
]

TRIPLE_WEIGHTS = [
    (1 / 3, 1 / 3, 1 / 3),
    (0.50, 0.25, 0.25),
    (0.40, 0.40, 0.20),
    (0.40, 0.30, 0.30),
]


@dataclass
class EvalResult:
    name: str
    stage: str
    task_1_pa_ndcg: float
    top1_accuracy: float
    top1_acc: float
    n_rows: int
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage,
            "task_1_pa_ndcg": self.task_1_pa_ndcg,
            "top1_accuracy": self.top1_accuracy,
            "top1_acc": self.top1_acc,
            "n_rows": self.n_rows,
            "notes": self.notes,
        }


def clear_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_data(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Cargando train: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Cargando test:  {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)

    validate_columns(train_df)
    validate_columns(test_df)
    if "y_true" not in test_df.columns:
        raise ValueError("TEST_CSV debe contener y_true para evaluación local.")

    if limit is not None:
        test_df = test_df.head(limit).copy()
        print(f"[DEBUG] Limitando evaluación a {len(test_df)} filas.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")
    return train_df, test_df


def build_tfidf_ranker(train_df: pd.DataFrame) -> TfidfRanker:
    ranker = TfidfRanker()
    ranker.fit(build_tfidf_corpus(train_df))
    return ranker


def build_bm25_ranker(train_df: pd.DataFrame) -> BM25Ranker:
    ranker = BM25Ranker(
        k1=BM25_K1,
        b=BM25_B,
        query_term_limit=BM25_QUERY_TERM_LIMIT,
    )
    ranker.fit(build_bm25_corpus(train_df))
    return ranker


def build_bge_ranker() -> ModernReranker:
    cfg = MODERN_RERANKER_CONFIGS["bge_reranker_v2_m3"]
    return ModernReranker(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        use_fp16=cfg.get("use_fp16", True),
    )


def build_named_ranker(name: str, train_df: pd.DataFrame):
    name = name.lower().strip()
    if name == "tfidf":
        return build_tfidf_ranker(train_df)
    if name == "bm25":
        return build_bm25_ranker(train_df)
    if name == "semantic":
        return SemanticRanker()
    if name in {"bert", "bert_headtail", "bertin", "mdeberta"}:
        return build_crossencoder_ranker(name)
    if name == "bert_rank10":
        return build_crossencoder_rank10_ranker("bert_rank10")
    if name == "bge":
        return build_bge_ranker()
    raise ValueError(f"Ranker no soportado: {name}")


def parse_weight_spec(spec: str) -> list[EnsembleMember]:
    """Convierte 'bert:0.7,bertin:0.3' en EnsembleMember(...)."""
    members: list[EnsembleMember] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Formato inválido en ensemble: {part!r}")
        name, weight = part.split(":", 1)
        members.append(EnsembleMember(name.strip(), float(weight)))
    if not members:
        raise ValueError("El ensemble no puede estar vacío.")
    return members


def make_ensemble_name(members: Sequence[EnsembleMember]) -> str:
    return "+".join(f"{m.model_key}_{m.weight:g}" for m in members)


def build_ensemble_from_spec(spec: str) -> CrossEncoderEnsembleRanker:
    return CrossEncoderEnsembleRanker(parse_weight_spec(spec))


def predict_with_ranker(
    ranker,
    df: pd.DataFrame,
    progress_every: int = 50,
) -> tuple[list[str], list[np.ndarray]]:
    preds: list[str] = []
    scores_cache: list[np.ndarray] = []

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        article = get_source_text_task1(row)
        titles = get_titles(row)
        scores = np.asarray(ranker.score_titles(article, titles), dtype=float)

        if len(scores) != len(titles):
            raise ValueError(
                f"Fila {idx}: el ranker devolvió {len(scores)} scores para {len(titles)} titulares."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Fila {idx}: scores no finitos.")

        scores_cache.append(scores)
        preds.append(ranking_from_scores(scores))

        if progress_every and idx % progress_every == 0:
            print(f"Predichas {idx}/{len(df)} filas...")

    return preds, scores_cache


def evaluate_predictions(
    name: str,
    stage: str,
    df: pd.DataFrame,
    preds: list[str],
    notes: str = "",
) -> EvalResult:
    metrics = score_task1_predictions_df(df, preds)
    result = EvalResult(
        name=name,
        stage=stage,
        task_1_pa_ndcg=float(metrics["task_1_pa_ndcg"]),
        top1_accuracy=float(metrics["top1_accuracy"]),
        top1_acc=float(metrics["top1_acc"]),
        n_rows=len(df),
        notes=notes,
    )
    print(
        f"[{stage}] {name}: PA-nDCG={result.task_1_pa_ndcg:.6f} | "
        f"top1={result.top1_accuracy:.6f}"
    )
    return result


def evaluate_ranker(
    name: str,
    stage: str,
    ranker,
    test_df: pd.DataFrame,
    predictions_dir: Path | None = None,
    notes: str = "",
) -> tuple[EvalResult, list[str], list[np.ndarray]]:
    print(f"\n=== Evaluando {stage}: {name} ===")
    preds, scores_cache = predict_with_ranker(ranker, test_df)
    result = evaluate_predictions(name, stage, test_df, preds, notes=notes)

    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": test_df["id"].astype(str), "prediction": preds}).to_csv(
            predictions_dir / f"{safe_name(name)}.csv",
            index=False,
        )

    return result, preds, scores_cache


def safe_name(name: str) -> str:
    return (
        name.replace("/", "__")
        .replace("+", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace(" ", "_")
    )


def save_results(stage: str, results: list[EvalResult]) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{stage}.csv"
    pd.DataFrame([r.as_dict() for r in results]).sort_values(
        by="task_1_pa_ndcg",
        ascending=False,
    ).to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")
    return out_path


def stage_individual(train_df: pd.DataFrame, test_df: pd.DataFrame, models: Sequence[str]) -> list[EvalResult]:
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "individual"

    for name in models:
        ranker = build_named_ranker(name, train_df)
        result, _, _ = evaluate_ranker(
            name=name,
            stage="individual",
            ranker=ranker,
            test_df=test_df,
            predictions_dir=predictions_dir,
        )
        results.append(result)
        del ranker
        clear_memory()

    save_results("individual_models", results)
    return results


def default_ensemble_specs(candidates: Sequence[str]) -> list[str]:
    specs: list[str] = []

    # Parejas con varios pesos.
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            for wa, wb in PAIR_WEIGHTS:
                specs.append(f"{a}:{wa},{b}:{wb}")

    # Ternas principales.
    if len(candidates) >= 3:
        from itertools import combinations

        for combo in combinations(candidates, 3):
            for weights in TRIPLE_WEIGHTS:
                specs.append(
                    ",".join(f"{model}:{weight}" for model, weight in zip(combo, weights))
                )

    # Todos con pesos uniformes.
    if len(candidates) >= 4:
        w = 1.0 / len(candidates)
        specs.append(",".join(f"{model}:{w}" for model in candidates))

    # El ensemble original como referencia.
    specs.append("bert:0.7,bertin:0.3")

    # Sin duplicados, preservando orden.
    return list(dict.fromkeys(specs))


def stage_ensembles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidates: Sequence[str],
    specs: Sequence[str] | None = None,
) -> list[EvalResult]:
    del train_df  # Los ensembles cargan checkpoints ya entrenados.
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "ensembles"

    specs = list(specs) if specs else default_ensemble_specs(candidates)
    print(f"Evaluando {len(specs)} ensembles...")

    for spec in specs:
        ranker = build_ensemble_from_spec(spec)
        members = parse_weight_spec(spec)
        name = make_ensemble_name(members)
        result, _, _ = evaluate_ranker(
            name=name,
            stage="ensembles",
            ranker=ranker,
            test_df=test_df,
            predictions_dir=predictions_dir,
            notes=spec,
        )
        results.append(result)
        del ranker
        clear_memory()

    save_results("ensembles", results)
    return results


def stage_rerankers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_ensemble: str,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "rerankers"

    base_ranker = build_ensemble_from_spec(base_ensemble)
    base_name = f"base_ensemble({base_ensemble})"
    result, _, _ = evaluate_ranker(base_name, "rerankers", base_ranker, test_df, predictions_dir)
    results.append(result)

    # Tail reranking con Rank10.
    rank10 = build_crossencoder_rank10_ranker("bert_rank10")
    tail_rank10 = TailReranker(base_ranker, rank10, top_k=10)
    result, _, _ = evaluate_ranker(
        name=f"{base_name}+tail_rank10",
        stage="rerankers",
        ranker=tail_rank10,
        test_df=test_df,
        predictions_dir=predictions_dir,
    )
    results.append(result)
    del rank10, tail_rank10
    clear_memory()

    # Tail reranking con BGE.
    bge = build_bge_ranker()
    tail_bge = TailReranker(base_ranker, bge, top_k=10)
    result, _, _ = evaluate_ranker(
        name=f"{base_name}+tail_bge",
        stage="rerankers",
        ranker=tail_bge,
        test_df=test_df,
        predictions_dir=predictions_dir,
    )
    results.append(result)

    # BGE como ensemble y rerank_tail sobre la base textual.
    for mode in ["ensemble", "rerank", "rerank_tail"]:
        pipe = ModernRerankerPipeline(
            reranker=bge,
            base_ranker=base_ranker,
            mode=mode,
            base_weight=0.90,
            reranker_weight=0.10,
            rerank_top_k=10,
        )
        result, _, _ = evaluate_ranker(
            name=f"{base_name}+bge_{mode}",
            stage="rerankers",
            ranker=pipe,
            test_df=test_df,
            predictions_dir=predictions_dir,
        )
        results.append(result)
        del pipe
        clear_memory()

    del bge, base_ranker
    clear_memory()

    # Tail reranking con baselines ligeros.
    for aux_name in ["bm25", "semantic"]:
        base_ranker = build_ensemble_from_spec(base_ensemble)
        aux_ranker = build_named_ranker(aux_name, train_df)
        tail = TailReranker(base_ranker, aux_ranker, top_k=10)
        result, _, _ = evaluate_ranker(
            name=f"{base_name}+tail_{aux_name}",
            stage="rerankers",
            ranker=tail,
            test_df=test_df,
            predictions_dir=predictions_dir,
        )
        results.append(result)
        del base_ranker, aux_ranker, tail
        clear_memory()

    save_results("rerankers", results)
    return results


def stage_llm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    llm_models: Sequence[str],
    mode: str = "solo",
    base_model: str = "crossencoder_ensemble",
    base_ensemble: str = "bert:0.7,bertin:0.3",
) -> list[EvalResult]:
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "llm"
    mode = mode.lower().strip()

    if mode != "solo":
        if base_model == "crossencoder_ensemble":
            base_ranker = build_ensemble_from_spec(base_ensemble)
        else:
            base_ranker = build_named_ranker(base_model, train_df)
    else:
        base_ranker = None

    for model_name in llm_models:
        llm_config = LLMRankerConfig(
            model_name=model_name,
            max_input_chars=LLM_MAX_INPUT_CHARS,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=LLM_DO_SAMPLE,
            load_in_4bit=LLM_LOAD_IN_4BIT,
            torch_dtype=LLM_TORCH_DTYPE,
            trust_remote_code=LLM_TRUST_REMOTE_CODE,
        )
        llm = LLMRanker(llm_config)
        ranker = LLMEnsembleRanker(
            base_ranker=base_ranker,
            llm_ranker=llm,
            mode=mode,
            base_weight=0.85,
            llm_weight=0.15,
            rerank_top_k=10,
        )
        result, _, _ = evaluate_ranker(
            name=f"llm_{model_name}_{mode}",
            stage="llm",
            ranker=ranker,
            test_df=test_df,
            predictions_dir=predictions_dir,
            notes=f"mode={mode}; base_model={base_model}; base_ensemble={base_ensemble}",
        )
        results.append(result)
        llm.unload()
        del llm, ranker
        clear_memory()

    del base_ranker
    clear_memory()
    save_results("llm", results)
    return results


def stage_vlm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_ensemble: str,
    images_dir: Path,
) -> list[EvalResult]:
    del train_df
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "vlm"

    if not images_dir.exists():
        print(f"[WARN] No existe images_dir: {images_dir}. La evaluación VLM usará fallback si no encuentra imágenes.")

    base_ranker = build_ensemble_from_spec(base_ensemble)
    base_result, _, base_scores = evaluate_ranker(
        name=f"text_base({base_ensemble})",
        stage="vlm",
        ranker=base_ranker,
        test_df=test_df,
        predictions_dir=predictions_dir,
    )
    results.append(base_result)

    for backend, model_name in VLM_MODELS:
        print(f"\n=== Cargando VLM {backend}: {model_name} ===")
        vlm_model, vlm_processor, device = load_vlm(backend=backend, model_name=model_name)

        # VLM solo.
        solo_preds = predict_task2_vlm(
            df_pred=test_df,
            images_dir=images_dir,
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            device=device,
            backend=backend,
        )
        result = evaluate_predictions(
            name=f"vlm_solo_{backend}_{model_name}",
            stage="vlm",
            df=test_df,
            preds=solo_preds,
            notes="image_only",
        )
        results.append(result)

        # Fusión con el mejor textual.
        for w_text, w_img in [(0.98, 0.02), (0.95, 0.05), (0.90, 0.10), (0.80, 0.20)]:
            fused_preds = predict_task2_vlm_plus_text_scores(
                df_pred=test_df,
                images_dir=images_dir,
                text_scores_list=base_scores,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                backend=backend,
                device=device,
                w_text=w_text,
                w_img=w_img,
            )
            result = evaluate_predictions(
                name=f"text_vlm_{backend}_{model_name}_{w_text:g}_{w_img:g}",
                stage="vlm",
                df=test_df,
                preds=fused_preds,
                notes=f"base={base_ensemble}; w_text={w_text}; w_img={w_img}",
            )
            results.append(result)

        del vlm_model, vlm_processor
        clear_memory()

    del base_ranker
    clear_memory()
    save_results("vlm", results)
    return results


def parse_list_arg(value: str | None, default: Sequence[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación masiva de modelos para los experimentos del TFG.",
    )
    parser.add_argument(
        "--stage",
        choices=["individual", "ensembles", "rerankers", "llm", "vlm", "all_text"],
        required=True,
        help="Fase de evaluación a ejecutar.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de filas de test para pruebas rápidas.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Lista separada por comas para individual. Ej: bert,bertin,bge",
    )
    parser.add_argument(
        "--ensemble-candidates",
        type=str,
        default="bert,bertin,mdeberta,bert_headtail",
        help="Modelos candidatos para grid de ensembles.",
    )
    parser.add_argument(
        "--ensemble-specs",
        type=str,
        default=None,
        help="Lista de ensembles separada por ';'. Ej: 'bert:0.7,bertin:0.3;bert:0.5,mdeberta:0.5'",
    )
    parser.add_argument(
        "--base-ensemble",
        type=str,
        default="bert:0.7,bertin:0.3",
        help="Ensemble base para rerankers/VLM/LLM rerank.",
    )
    parser.add_argument(
        "--llm-models",
        type=str,
        default=None,
        help="LLMs separados por coma. Por defecto: Qwen2.5-7B y Llama-3.1-8B.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["solo", "ensemble", "rerank"],
        default="solo",
        help="Modo de evaluación de LLM.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Directorio de imágenes para VLM.",
    )
    args = parser.parse_args()

    train_df, test_df = load_data(limit=args.limit)

    if args.stage == "individual":
        models = parse_list_arg(args.models, INDIVIDUAL_MODELS)
        stage_individual(train_df, test_df, models)
        return

    if args.stage == "ensembles":
        candidates = parse_list_arg(args.ensemble_candidates, [])
        specs = None
        if args.ensemble_specs:
            specs = [s.strip() for s in args.ensemble_specs.split(";") if s.strip()]
        stage_ensembles(train_df, test_df, candidates=candidates, specs=specs)
        return

    if args.stage == "rerankers":
        stage_rerankers(train_df, test_df, base_ensemble=args.base_ensemble)
        return

    if args.stage == "llm":
        llm_models = parse_list_arg(args.llm_models, LLM_MODELS)
        stage_llm(
            train_df,
            test_df,
            llm_models=llm_models,
            mode=args.llm_mode,
            base_ensemble=args.base_ensemble,
        )
        return

    if args.stage == "vlm":
        stage_vlm(train_df, test_df, base_ensemble=args.base_ensemble, images_dir=args.images_dir)
        return

    if args.stage == "all_text":
        models = parse_list_arg(args.models, INDIVIDUAL_MODELS)
        candidates = parse_list_arg(args.ensemble_candidates, [])
        stage_individual(train_df, test_df, models)
        stage_ensembles(train_df, test_df, candidates=candidates)
        stage_rerankers(train_df, test_df, base_ensemble=args.base_ensemble)
        print("\nall_text terminado. LLM y VLM se ejecutan aparte con --stage llm / --stage vlm.")
        return

    raise RuntimeError(f"Stage no alcanzable: {args.stage}")


if __name__ == "__main__":
    main()
