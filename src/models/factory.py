from __future__ import annotations

"""
Ranker factory functions used by the main pipeline and experiment scripts.

This module centralizes model construction so scripts do not need to duplicate
configuration-to-ranker logic.
"""

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from config import (
    BM25_B,
    BM25_K1,
    BM25_QUERY_TERM_LIMIT,
    CROSS_ENCODER_ENSEMBLE_MEMBERS,
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
    MODERN_RERANKER_BASE_RANKER,
    MODERN_RERANKER_BASE_WEIGHT,
    MODERN_RERANKER_CONFIGS,
    MODERN_RERANKER_MODE,
    MODERN_RERANKER_MODEL_KEY,
    MODERN_RERANKER_TOP_K,
    MODERN_RERANKER_WEIGHT,
    TAIL_RERANKER_AUX_RANKER,
    TAIL_RERANKER_BASE_RANKER,
    TAIL_RERANKER_TOP_K,
    get_cross_encoder_rank10_runtime_config,
    get_cross_encoder_runtime_config,
    get_modern_reranker_runtime_config,
)
from models.bm25_ranker import BM25Ranker, build_bm25_corpus
from models.cross_encoder_ensemble_ranker import CrossEncoderEnsembleRanker, EnsembleMember
from models.cross_encoder_rank10_ranker import CrossEncoderRank10Config, CrossEncoderRank10Ranker
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker
from models.llm_ranker import LLMEnsembleRanker, LLMRanker, LLMRankerConfig
from models.modern_reranker import ModernReranker, ModernRerankerPipeline
from models.semantic_ranker import SemanticRanker
from models.tail_reranker import TailReranker
from models.tfidf_ranker import TfidfRanker, build_tfidf_corpus


POINTWISE_CROSS_ENCODERS = {"beto", "beto_headtail", "bertin", "mdeberta"}


def build_crossencoder_ranker(model_key: str) -> CrossEncoderRanker:
    """
    Load a trained pointwise cross-encoder ranker.
    """
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


def build_crossencoder_rank10_ranker(model_key: str = "beto_rank10") -> CrossEncoderRank10Ranker:
    """
    Load the rank10 cross-encoder.
    """
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


def build_crossencoder_ensemble_ranker(
    members: Sequence[EnsembleMember] | None = None,
) -> CrossEncoderEnsembleRanker:
    """
    Build a weighted soft-voting ensemble of trained cross-encoders.
    """
    if members is None:
        members = [
            EnsembleMember(model_key=name, weight=weight)
            for name, weight in CROSS_ENCODER_ENSEMBLE_MEMBERS
        ]
    return CrossEncoderEnsembleRanker(list(members))


def build_tfidf_ranker(train_df: pd.DataFrame) -> TfidfRanker:
    """
    Build and fit the TF-IDF baseline ranker.
    """
    ranker = TfidfRanker()
    ranker.fit(build_tfidf_corpus(train_df))
    return ranker


def build_bm25_ranker(train_df: pd.DataFrame) -> BM25Ranker:
    """
    Build and fit the BM25 baseline ranker.
    """
    ranker = BM25Ranker(
        k1=BM25_K1,
        b=BM25_B,
        query_term_limit=BM25_QUERY_TERM_LIMIT,
    )
    ranker.fit(build_bm25_corpus(train_df))
    return ranker


def build_bge_ranker() -> ModernReranker:
    """
    Build the configured BGE reranker.
    """
    cfg = MODERN_RERANKER_CONFIGS["bge_reranker_v2_m3"]
    return ModernReranker(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        use_fp16=cfg.get("use_fp16", True),
    )


def build_base_ranker(model_key: str, train_df: pd.DataFrame):
    """
    Build one of the base textual rankers.
    """
    model_key = model_key.lower().strip()

    if model_key == "tfidf":
        print("Loading TF-IDF...")
        return build_tfidf_ranker(train_df)

    if model_key == "bm25":
        print("Loading BM25...")
        return build_bm25_ranker(train_df)

    if model_key == "semantic":
        print("Loading Semantic Ranker...")
        return SemanticRanker()

    if model_key == "crossencoder_ensemble":
        print("Loading cross-encoders ensemble...")
        print("Ensemble members:", CROSS_ENCODER_ENSEMBLE_MEMBERS)
        return build_crossencoder_ensemble_ranker()

    if model_key in POINTWISE_CROSS_ENCODERS:
        print(f"Loading trained cross-encoder: {model_key}...")
        ranker = build_crossencoder_ranker(model_key)
        print(f"Device {model_key}: {ranker.device}")
        return ranker

    if model_key == "beto_rank10":
        print("Loading trained rank10 cross-encoder...")
        ranker = build_crossencoder_rank10_ranker("beto_rank10")
        print(f"Device beto_rank10: {ranker.device}")
        return ranker

    if model_key == "bge":
        print("Loading BGE reranker...")
        return build_bge_ranker()

    raise ValueError(f"Base ranker not supported: {model_key}")
    
    
def build_named_ranker(model_key: str, train_df: pd.DataFrame):
    """
    Backwards-compatible alias for building a ranker by name.

    This function is used by evaluation scripts. Internally, it delegates to
    build_base_ranker(), which supports lexical baselines, semantic models,
    cross-encoders, rank10 and BGE.
    """
    return build_base_ranker(model_key, train_df=train_df)


def build_llm_ranker(train_df: pd.DataFrame):
    """
    Build the LLM-based ranker.
    """
    print(f"Loading LLM ranker: {LLM_MODEL_NAME}")
    print(f"LLM mode: {LLM_RANKER_MODE}")

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
    """
    Build the BGE reranker pipeline.
    """
    cfg = get_modern_reranker_runtime_config(MODERN_RERANKER_MODEL_KEY)
    print(f"Loading modern reranker: {MODERN_RERANKER_MODEL_KEY} ({cfg['model_name']})")
    print(f"Modern reranker mode: {MODERN_RERANKER_MODE}")

    reranker = ModernReranker(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        use_fp16=cfg.get("use_fp16", True),
    )
    print(f"Modern reranker device: {reranker.device}")

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
    """
    Build a tail reranking pipeline.
    """
    print("Loading tail reranker...")
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
    """
    Call the corresponding builder depending on the selected model.
    """
    model_name = model_name.lower().strip()

    if model_name == "llm_ranker":
        return build_llm_ranker(train_df=train_df)

    if model_name == "modern_reranker":
        return build_modern_reranker(train_df=train_df)

    if model_name == "tail_reranker":
        return build_tail_reranker(train_df=train_df)

    return build_base_ranker(model_name, train_df=train_df)


def parse_weight_spec(spec: str) -> list[EnsembleMember]:
    """
    Parse a string such as "beto:0.7,bertin:0.3" into ensemble members.
    """
    members: list[EnsembleMember] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid ensemble format: {part!r}")
        name, weight = part.split(":", 1)
        members.append(EnsembleMember(name.strip(), float(weight)))
    if not members:
        raise ValueError("The ensemble cannot be empty.")
    return members


def make_ensemble_name(members: Sequence[EnsembleMember]) -> str:
    """
    Build a compact name for an ensemble specification.
    """
    return "+".join(f"{member.model_key}_{member.weight:g}" for member in members)


def build_ensemble_from_spec(spec: str) -> CrossEncoderEnsembleRanker:
    """
    Build a cross-encoder ensemble from a command-line weight specification.
    """
    return CrossEncoderEnsembleRanker(parse_weight_spec(spec))
