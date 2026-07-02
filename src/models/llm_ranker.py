from __future__ import annotations

"""
LLM-based ranking utilities.

This module adapts instruction-tuned causal language models to the same
score_titles(article, titles) interface used by the rest of the ranking
pipeline.
"""

import gc
import json
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from config import FORCE_CPU
from utils.data_utils import get_source_text_task1, get_titles
from utils.scoring import minmax_01, rank_tokens_from_scores, order_to_scores


@dataclass
class LLMRankerConfig:
    """
    Default configuration values used to load and run an instruction-tuned LLM.
    """

    model_name: str
    max_input_chars: int = 3500
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    load_in_4bit: bool = True
    torch_dtype: str = "auto"
    trust_remote_code: bool = False


def get_device() -> str:
    """
    Return the execution device used for LLM inference.
    """

    if FORCE_CPU:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(dtype_name: str):
    """
    Resolve a string dtype name into the value expected by Transformers.
    """

    dtype_name = str(dtype_name).lower().strip()
    if dtype_name == "auto":
        return "auto"
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype for LLM: {dtype_name}")


def _repair_common_json_errors(text: str) -> str:
    """
    Repair a small set of common malformed JSON outputs produced by LLMs.
    """

    # Repair cases such as {"id": 2, 0} -> {"id": 2, "score": 0}.
    text = re.sub(
        r'(\{"id"\s*:\s*\d+\s*,\s*)(-?\d+(?:\.\d+)?)\s*\}',
        r'\1"score": \2}',
        text,
    )
    return text


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Extract the first valid JSON object from a generated text.
    """

    text = str(text or "").strip()
    text = _repair_common_json_errors(text)

    candidates = [text]

    fenced = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(fenced)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _scores_from_payload(payload: dict, n_titles: int) -> Optional[np.ndarray]:
    """
    Convert a parsed JSON payload into one score per candidate title.
    """

    if not isinstance(payload, dict):
        return None

    raw_scores = payload.get("scores")
    scores = np.full(n_titles, np.nan, dtype=float)

    if isinstance(raw_scores, list):
        # Expected format: [{"id": 1, "score": 8.5}, ...]
        for item in raw_scores:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id")) - 1
                value = float(item.get("score"))
            except (TypeError, ValueError):
                continue

            if 0 <= idx < n_titles:
                scores[idx] = value

        # Alternative format: [8.5, 3.0, ...]
        if np.isnan(scores).all() and len(raw_scores) == n_titles:
            try:
                scores = np.asarray([float(x) for x in raw_scores], dtype=float)
            except (TypeError, ValueError):
                return None

    if np.isnan(scores).all():
        return None

    # Partial parser: if an id is missing, keep the prediction and fill later.
    scores = np.nan_to_num(scores, nan=0.0)

    # Defensive clamp to the expected score range.
    scores = np.clip(scores, 0.0, 10.0)

    return scores


class LLMRanker:
    """
    Text ranker for instruction-tuned models such as Qwen or Llama.

    The class implements score_titles(article, titles), same to most other models.
    """

    def __init__(self, config: LLMRankerConfig):
        self.config = config
        self.device = get_device()

        tokenizer_kwargs = {"trust_remote_code": config.trust_remote_code}
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, **tokenizer_kwargs)

        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": _resolve_torch_dtype(config.torch_dtype),
        }

        if self.device == "cuda" and config.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "LLM_LOAD_IN_4BIT=True requires bitsandbytes and a Transformers "
                    "version with BitsAndBytesConfig. Install bitsandbytes or set "
                    "LLM_LOAD_IN_4BIT=False."
                )

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()


    def _build_messages(self, article: str, titles: List[str]) -> list[dict]:
        """
        Build chat messages for the LLM prompt. 
        
        Kept in spanish.
        """

        article = str(article or "").strip()

        if self.config.max_input_chars and len(article) > self.config.max_input_chars:
            article = article[: self.config.max_input_chars]

        titles_block = "\n".join(
            f'{i + 1}. "{str(title or "").strip()}"'
            for i, title in enumerate(titles)
        )

        n_titles = len(titles)

        system = (
            "Eres un sistema experto en ranking de titulares de noticias en español. "
            "Tu tarea es puntuar la relevancia semántica, factual y contextual de titulares candidatos. "
            "Debes seguir estrictamente el formato JSON solicitado. "
            "No expliques nada."
        )

        expected_items = ",\n".join(
            f'    {{"id": {i + 1}, "score": 0}}'
            for i in range(n_titles)
        )

        user = dedent(f"""
        Noticia de referencia:
        \"\"\"
        {article}
        \"\"\"

        Titulares candidatos:
        {titles_block}

        Instrucciones:
        - Evalúa TODOS los titulares candidatos.
        - Cada elemento debe tener SIEMPRE las dos claves: "id" y "score".
        - No omitas ningún id.
        - Los ids deben ser los mismos que aparecen en la lista de candidatos.
        - Nunca escribas objetos como {{"id": 2, 0}}; siempre debe ser {{"id": 2, "score": 0}}.
        - Puntúa cada titular de 0 a 10.
        - 10 = mismo evento/noticia, mismos actores principales y mismo contexto.
        - 7-9 = muy relacionado, aunque falte algún detalle.
        - 4-6 = relacionado por tema general, pero no necesariamente mismo evento.
        - 1-3 = relación débil.
        - 0 = no relacionado.
        - Usa decimales si ayuda a separar titulares parecidos.
        - No escribas explicación.
        - No escribas texto antes ni después del JSON.
        - Devuelve SOLO JSON válido.

        Formato obligatorio:
        {{
          "scores": [
        {expected_items}
          ]
        }}
        """).strip()

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


    @torch.inference_mode()
    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        """
        Generate LLM relevance scores for the candidate titles.
        """

        if not titles:
            return np.array([], dtype=float)

        messages = self._build_messages(article, titles)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device == "cuda":
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        payload = _extract_json_object(text)
        scores = _scores_from_payload(payload, len(titles)) if payload is not None else None

        if scores is None:
            print("[WARN] Could not parse LLM output; using the original ranking.")
            print("LLM output:", text[:500])
            return np.arange(len(titles), 0, -1, dtype=float)

        return scores.astype(float)


    def rank_titles(self, article: str, titles: List[str]) -> List[str]:
        """
        Return title tokens sorted by descending LLM score.
        """

        return rank_tokens_from_scores(self.score_titles(article, titles))



    def predict_dataframe(self, df_pred: pd.DataFrame) -> List[str]:
        """
        Used for isolated testing: predict rankings for a full dataframe.
        """

        preds = []

        for idx, (_, row) in enumerate(df_pred.iterrows(), start=1):
            article = get_source_text_task1(row)
            titles = get_titles(row)
            preds.append(" ".join(self.rank_titles(article, titles)))

            if idx % 10 == 0:
                print(f"Predicted {idx}/{len(df_pred)} rows...")

        return preds


    def unload(self) -> None:
        """
        Release model/tokenizer references and clear CUDA cache when possible.
        """

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()



class LLMEnsembleRanker:
    """
    Combine a base ranker and an LLM through ensemble or top-k reranking.
    """

    def __init__(
        self,
        base_ranker,
        llm_ranker: LLMRanker,
        mode: str = "rerank",
        base_weight: float = 0.85,
        llm_weight: float = 0.15,
        rerank_top_k: int = 10,
    ):
        self.base_ranker = base_ranker
        self.llm_ranker = llm_ranker
        self.mode = mode.lower().strip()
        self.base_weight = float(base_weight)
        self.llm_weight = float(llm_weight)
        self.rerank_top_k = int(rerank_top_k)

        if self.mode not in {"solo", "ensemble", "rerank"}:
            raise ValueError(f"Unsupported LLM_RANKER_MODE: {mode}")

    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        if self.mode == "solo" or self.base_ranker is None:
            return self.llm_ranker.score_titles(article, titles)

        base_scores = np.asarray(self.base_ranker.score_titles(article, titles), dtype=float)

        if self.mode == "ensemble":
            llm_scores = np.asarray(self.llm_ranker.score_titles(article, titles), dtype=float)

            total = self.base_weight + self.llm_weight
            w_base = self.base_weight / total
            w_llm = self.llm_weight / total

            return (w_base * minmax_01(base_scores)) + (w_llm * minmax_01(llm_scores))

        # mode == "rerank": keep candidates outside top-k in base order and
        # rerank only the top-k candidates with the LLM.
        n = len(titles)
        k = max(1, min(self.rerank_top_k, n))

        base_order = list(np.argsort(-base_scores))
        top_indices = base_order[:k]
        tail_indices = base_order[k:]

        top_titles = [titles[i] for i in top_indices]
        top_llm_scores = np.asarray(
            self.llm_ranker.score_titles(article, top_titles),
            dtype=float,
        )

        top_order_local = list(np.argsort(-top_llm_scores))
        top_order = [top_indices[i] for i in top_order_local]

        final_order = top_order + tail_indices

        return order_to_scores(final_order, n)

    def rank_titles(self, article: str, titles: List[str]) -> List[str]:
        """
        Return title tokens sorted by descending LLM score.
        """

        return rank_tokens_from_scores(self.score_titles(article, titles))

    def predict_dataframe(self, df_pred: pd.DataFrame) -> List[str]:
        """
        Used for isolated testing: predict rankings for a full dataframe.
        """

        preds = []

        for idx, (_, row) in enumerate(df_pred.iterrows(), start=1):
            article = get_source_text_task1(row)
            titles = get_titles(row)
            preds.append(" ".join(self.rank_titles(article, titles)))

            if idx % 10 == 0:
                print(f"Predicted {idx}/{len(df_pred)} rows...")

        return preds
    
    
    