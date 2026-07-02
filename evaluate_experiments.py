"""
Batch evaluation script for PoliticHeadlinES TFG experiments.

This script evaluates groups of rankers and experimental variants over the
configured local test split. It can run individual baselines, cross-encoder
ensembles, rerankers, LLM variants and VLM fusion experiments, saving both
summary CSV files and per-model predictions under outputs/evaluation/.

This is an experimentation script. It intentionally keeps several model
builders and stages in one place to make large evaluation runs reproducible.
"""

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
)
from models.factory import (  # noqa: E402
    build_bge_ranker,
    build_crossencoder_rank10_ranker,
    build_crossencoder_ranker,
    build_ensemble_from_spec,
    build_named_ranker,
    make_ensemble_name,
    parse_weight_spec,
)
from utils.data_utils import validate_columns  # noqa: E402
from utils.metrics import score_task1_predictions_df  # noqa: E402
from utils.inference import score_dataframe_with_ranker as predict_with_ranker  # noqa: E402


EVAL_DIR = OUTPUTS_DIR / "evaluation"

INDIVIDUAL_MODELS = [
    "tfidf",
    "bm25",
    "semantic",
    "beto",
    "beto_headtail",
    "bertin",
    "mdeberta",
    "beto_rank10",
    "bge",
]

# Edit this list to test other LLMs. Llama may require HF_TOKEN and accepted access.
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
    """
    Run garbage collection and release cached CUDA memory when available.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_data(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the configured train and test CSV files used for local evaluation.
    """
    print(f"Loading train: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Loading test:  {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)

    validate_columns(train_df)
    validate_columns(test_df)
    if "y_true" not in test_df.columns:
        raise ValueError("TEST_CSV must contain y_true for local evaluation.")

    if limit is not None:
        test_df = test_df.head(limit).copy()
        print(f"[DEBUG] Limiting evaluation to {len(test_df)} rows.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows:  {len(test_df)}")
    return train_df, test_df


def evaluate_predictions(
    name: str,
    stage: str,
    df: pd.DataFrame,
    preds: list[str],
    notes: str = "",
) -> EvalResult:
    """
    Compute metrics for already generated predictions and wrap them in EvalResult.
    """
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
    """
    Run prediction, evaluation and optional prediction export for one ranker.
    """
    print(f"\n=== Evaluating {stage}: {name} ===")
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
    """
    Convert a model name into a safe filename component.
    """
    return (
        name.replace("/", "__")
        .replace("+", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace(" ", "_")
    )


def save_results(stage: str, results: list[EvalResult]) -> Path:
    """
    Save a sorted CSV summary for one evaluation stage.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{stage}.csv"
    pd.DataFrame([r.as_dict() for r in results]).sort_values(
        by="task_1_pa_ndcg",
        ascending=False,
    ).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    return out_path


def stage_individual(train_df: pd.DataFrame, test_df: pd.DataFrame, models: Sequence[str]) -> list[EvalResult]:
    """
    Evaluate individual baseline and neural rankers.
    """
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

    # Pairwise ensembles with multiple weight combinations
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            for wa, wb in PAIR_WEIGHTS:
                specs.append(f"{a}:{wa},{b}:{wb}")

    # Main three-model combinations
    if len(candidates) >= 3:
        from itertools import combinations

        for combo in combinations(candidates, 3):
            for weights in TRIPLE_WEIGHTS:
                specs.append(
                    ",".join(f"{model}:{weight}" for model, weight in zip(combo, weights))
                )

    # Uniform ensemble with all candidates
    if len(candidates) >= 4:
        w = 1.0 / len(candidates)
        specs.append(",".join(f"{model}:{w}" for model in candidates))

    # Original ensemble kept as a reference
    specs.append("beto:0.7,bertin:0.3")

    # Remove duplicates while preserving order
    return list(dict.fromkeys(specs))


def stage_ensembles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidates: Sequence[str],
    specs: Sequence[str] | None = None,
) -> list[EvalResult]:
    """
    Evaluate weighted cross-encoder ensemble specifications.
    """
    del train_df  # Ensembles load already trained checkpoints.
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "ensembles"

    specs = list(specs) if specs else default_ensemble_specs(candidates)
    print(f"Evaluating {len(specs)} ensembles...")

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
    """
    Evaluate tail and BGE reranking variants over a base ensemble.
    """
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "rerankers"

    base_ranker = build_ensemble_from_spec(base_ensemble)
    base_name = f"base_ensemble({base_ensemble})"
    result, _, _ = evaluate_ranker(base_name, "rerankers", base_ranker, test_df, predictions_dir)
    results.append(result)

    # Tail reranking with Rank10.
    rank10 = build_crossencoder_rank10_ranker("beto_rank10")
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

    # Tail reranking with BGE.
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

    # BGE as ensemble/reranker over the textual base.
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

    # Tail reranking with lightweight baselines
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
    base_ensemble: str = "beto:0.7,bertin:0.3",
) -> list[EvalResult]:
    """
    Evaluate LLM rankers in solo, ensemble or rerank mode.
    """
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


def build_vlm_text_ranker(base_ensemble: str, vlm_text_base: str):
    """
    Build the textual base that will be fused with visual scores.

    - ensemble: use the specified textual ensemble directly.
    - tail_rank10: apply TailReranker over that ensemble using beto_rank10.

    TailReranker returns ordinal scores consistent with its final ranking, so
    they can be reused as text_scores_list for VLM fusion.
    """
    vlm_text_base = vlm_text_base.lower().strip()
    base_ranker = build_ensemble_from_spec(base_ensemble)

    if vlm_text_base == "ensemble":
        name = f"text_base_ensemble({base_ensemble})"
        notes = f"vlm_text_base=ensemble; base={base_ensemble}"
        return base_ranker, name, notes

    if vlm_text_base == "tail_rank10":
        rank10 = build_crossencoder_rank10_ranker("beto_rank10")
        tail_rank10 = TailReranker(base_ranker, rank10, top_k=10)
        name = f"text_base_tail_rank10({base_ensemble})"
        notes = f"vlm_text_base=tail_rank10; base={base_ensemble}; tail=beto_rank10; top_k=10"
        return tail_rank10, name, notes

    raise ValueError(
        f"Unsupported vlm_text_base: {vlm_text_base!r}. "
        "Use 'ensemble' or 'tail_rank10'."
    )


def stage_vlm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_ensemble: str,
    images_dir: Path,
    vlm_text_base: str = "ensemble",
) -> list[EvalResult]:
    """
    Evaluate image-only VLMs and text-image fusion variants.
    """
    del train_df
    results: list[EvalResult] = []
    predictions_dir = EVAL_DIR / "predictions" / "vlm"

    if not images_dir.exists():
        print(f"[WARN] images_dir does not exist: {images_dir}. VLM evaluation will use fallbacks when images are missing.")

    text_ranker, text_base_name, text_base_notes = build_vlm_text_ranker(
        base_ensemble=base_ensemble,
        vlm_text_base=vlm_text_base,
    )
    base_result, _, base_scores = evaluate_ranker(
        name=text_base_name,
        stage="vlm",
        ranker=text_ranker,
        test_df=test_df,
        predictions_dir=predictions_dir,
        notes=text_base_notes,
    )
    results.append(base_result)

    for backend, model_name in VLM_MODELS:
        print(f"\n=== Loading VLM {backend}: {model_name} ===")
        vlm_model, vlm_processor, device = load_vlm(backend=backend, model_name=model_name)

        # Image-only VLM.
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

        # Fusion with the selected textual base.
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
                name=f"text_{vlm_text_base}_vlm_{backend}_{model_name}_{w_text:g}_{w_img:g}",
                stage="vlm",
                df=test_df,
                preds=fused_preds,
                notes=(
                    f"vlm_text_base={vlm_text_base}; base={base_ensemble}; "
                    f"w_text={w_text}; w_img={w_img}"
                ),
            )
            results.append(result)

        del vlm_model, vlm_processor
        clear_memory()

    del text_ranker
    clear_memory()
    save_results("vlm", results)
    return results


def parse_list_arg(value: str | None, default: Sequence[str]) -> list[str]:
    """
    Parse a comma-separated command-line list with a default fallback.
    """
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    """
    Parse command-line arguments and run the requested evaluation stage.
    """
    parser = argparse.ArgumentParser(
        description="Batch evaluation of PoliticHeadlinES TFG experiments.",
    )
    parser.add_argument(
        "--stage",
        choices=["individual", "ensembles", "rerankers", "llm", "vlm", "all_text"],
        required=True,
        help="Evaluation stage to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of test rows for quick checks.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list for the individual stage. Example: beto,bertin,bge",
    )
    parser.add_argument(
        "--ensemble-candidates",
        type=str,
        default="beto,bertin,mdeberta,beto_headtail",
        help="Candidate models for the ensemble grid.",
    )
    parser.add_argument(
        "--ensemble-specs",
        type=str,
        default=None,
        help="Semicolon-separated ensemble specs. Example: 'beto:0.7,bertin:0.3;beto:0.5,mdeberta:0.5'",
    )
    parser.add_argument(
        "--base-ensemble",
        type=str,
        default="beto:0.7,bertin:0.3",
        help="Base ensemble for rerankers, VLM and LLM rerank mode.",
    )
    parser.add_argument(
        "--llm-models",
        type=str,
        default=None,
        help="Comma-separated LLMs. Defaults to Qwen2.5-7B and Llama-3.1-8B.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["solo", "ensemble", "rerank"],
        default="solo",
        help="LLM evaluation mode.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Image directory for VLM evaluation.",
    )
    parser.add_argument(
        "--vlm-text-base",
        choices=["ensemble", "tail_rank10"],
        default="ensemble",
        help=(
            "Textual base to fuse with the VLM. "
            "'ensemble' uses the ensemble provided with --base-ensemble; "
            "'tail_rank10' applies TailReranker with beto_rank10 after that ensemble."
        ),
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
        stage_vlm(
            train_df,
            test_df,
            base_ensemble=args.base_ensemble,
            images_dir=args.images_dir,
            vlm_text_base=args.vlm_text_base,
        )
        return

    if args.stage == "all_text":
        models = parse_list_arg(args.models, INDIVIDUAL_MODELS)
        candidates = parse_list_arg(args.ensemble_candidates, [])
        stage_individual(train_df, test_df, models)
        stage_ensembles(train_df, test_df, candidates=candidates)
        stage_rerankers(train_df, test_df, base_ensemble=args.base_ensemble)
        print("\nall_text finished. LLM and VLM are run separately with --stage llm / --stage vlm.")
        return

    raise RuntimeError(f"Unreachable stage: {args.stage}")


if __name__ == "__main__":
    main()
