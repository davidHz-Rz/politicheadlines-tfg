from __future__ import annotations

"""
Experimental BGE reranker utilities.

This module wraps a pretrained sequence-classification reranker and exposes the
same score_titles(article, titles) interface used by the rest of the project.
It also provides a small pipeline for solo scoring, weighted ensembling and
top-k reranking over a base ranker.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.scoring import minmax_01, order_to_scores


class ModernReranker:
    """
    Pretrained BGE reranker with the common score_titles interface.

    The model receives article-title pairs and returns one relevance score for
    each candidate title.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 4,
        max_length: int = 256,
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        dtype = torch.float16 if self.device.type == "cuda" and use_fp16 else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def score_titles(self, article: str, titles: list[str]) -> np.ndarray:
        """
        Score all candidate titles for a given article.
        """

        if not titles:
            return np.array([], dtype=float)

        scores: list[float] = []

        for start in range(0, len(titles), self.batch_size):
            batch_titles = titles[start : start + self.batch_size]

            encoded = self.tokenizer(
                [article] * len(batch_titles),
                batch_titles,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            logits = self.model(**encoded).logits

            if logits.ndim == 2:
                batch_scores = logits[:, 0]
            else:
                batch_scores = logits.reshape(-1)

            scores.extend(batch_scores.detach().float().cpu().tolist())

        return np.asarray(scores, dtype=float)


class ModernRerankerPipeline:
    """
    Pipeline wrapper around ModernReranker.

    Supported modes:
    - solo: BGE scores all titles directly.
    - ensemble: combine base-ranker scores with BGE scores.
    - rerank: the base ranker selects the top-k candidates, then BGE reranks them.
    - rerank_tail: keep the base top-1 fixed and rerank positions 2..k.
    """

    VALID_MODES = {"solo", "ensemble", "rerank", "rerank_tail"}

    def __init__(
        self,
        reranker: ModernReranker,
        mode: str = "ensemble",
        base_ranker=None,
        rerank_top_k: int = 10,
        base_weight: float = 0.95,
        reranker_weight: float = 0.05,
        normalize_scores: bool = True,
    ):
        self.mode = mode.lower().strip()

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported modern_reranker mode: {self.mode}. "
                f"Valid modes: {sorted(self.VALID_MODES)}"
            )

        if self.mode != "solo" and base_ranker is None:
            raise ValueError(f"Mode '{self.mode}' requires base_ranker.")

        self.reranker = reranker
        self.base_ranker = base_ranker
        self.rerank_top_k = rerank_top_k
        self.base_weight = base_weight
        self.reranker_weight = reranker_weight
        self.normalize_scores = normalize_scores


    def score_titles(self, article: str, titles: list[str]) -> np.ndarray:
        """
        Return final scores according to the selected reranking mode.
        """

        n_titles = len(titles)

        if n_titles == 0:
            return np.array([], dtype=float)

        if self.mode == "solo":
            return self.reranker.score_titles(article, titles)

        base_scores = np.asarray(
            self.base_ranker.score_titles(article, titles),
            dtype=float,
        )

        if self.mode == "ensemble":
            reranker_scores = np.asarray(
                self.reranker.score_titles(article, titles),
                dtype=float,
            )

            if self.normalize_scores:
                base_scores = minmax_01(base_scores)
                reranker_scores = minmax_01(reranker_scores)

            return (
                self.base_weight * base_scores
                + self.reranker_weight * reranker_scores
            )

        base_order = list(np.argsort(-base_scores))
        k = min(max(1, self.rerank_top_k), n_titles)

        if self.mode == "rerank":
            top_indices = base_order[:k]
            rest_indices = base_order[k:]

            reranked_top = self._rerank_subset(article, titles, top_indices)
            final_order = reranked_top + rest_indices

            return self._order_to_scores(final_order, n_titles)

        if self.mode == "rerank_tail":
            if n_titles <= 1:
                return base_scores

            k = min(max(2, self.rerank_top_k), n_titles)

            fixed_top1 = [base_order[0]]
            tail_indices = base_order[1:k]
            rest_indices = base_order[k:]

            reranked_tail = self._rerank_subset(article, titles, tail_indices)
            final_order = fixed_top1 + reranked_tail + rest_indices

            return self._order_to_scores(final_order, n_titles)

        raise RuntimeError(f"Unreachable mode: {self.mode}")

    def _rerank_subset(
        self,
        article: str,
        titles: list[str],
        indices: list[int],
    ) -> list[int]:
        """
        Rerank a subset of candidate indices with the BGE reranker.
        """

        if not indices:
            return []

        subset_titles = [titles[i] for i in indices]
        subset_scores = self.reranker.score_titles(article, subset_titles)
        subset_order = list(np.argsort(-subset_scores))

        return [indices[i] for i in subset_order]

    @staticmethod
    def _order_to_scores(order: list[int], n_titles: int) -> np.ndarray:
        """
        Convert a complete order of indices into descending pseudo-scores.
        """
        return order_to_scores(order, n_titles)
    
    