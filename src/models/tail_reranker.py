from __future__ import annotations

"""
Two-stage tail reranking utility.

The module implements a light wrapper that keeps the best candidate from a base
ranker fixed and reranks the remaining top-k candidates with an auxiliary
ranker. It exposes score_titles(article, titles) so it can be used by run.py.
"""

from typing import List

import numpy as np

from utils.scoring import order_to_scores, validate_scores


class TailReranker:
    """
    Two-stage reranker.

    The base ranker first produces an initial order. The first candidate is
    kept fixed, and only the tail of the top-k prediction is reordered with a
    second ranker. The final order is converted back into descending scores so
    the rest of the pipeline can keep using argsort(-scores).
    """

    def __init__(self, base_ranker, tail_ranker, top_k: int = 10):
        if base_ranker is None:
            raise ValueError("base_ranker cannot be None.")
        if tail_ranker is None:
            raise ValueError("tail_ranker cannot be None.")
        if top_k < 2:
            raise ValueError("top_k must be at least 2 to rerank the tail.")

        self.base_ranker = base_ranker
        self.tail_ranker = tail_ranker
        self.top_k = top_k


    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        """
        Return final scores after fixing top-1 and reranking the tail.
        """

        n = len(titles)
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.array([1.0], dtype=float)

        base_scores = self._score_with_ranker(self.base_ranker, article, titles, "base_ranker")
        base_order = list(np.argsort(-base_scores))

        rerank_limit = min(self.top_k, n)
        fixed_head = [base_order[0]]
        tail_indices = base_order[1:rerank_limit]
        untouched_rest = base_order[rerank_limit:]

        tail_titles = [titles[i] for i in tail_indices]
        tail_scores = self._score_with_ranker(
            self.tail_ranker,
            article,
            tail_titles,
            "tail_ranker",
        )

        reranked_tail = [tail_indices[i] for i in np.argsort(-tail_scores)]
        final_order = fixed_head + reranked_tail + untouched_rest

        return self._order_to_scores(final_order, n)


    @staticmethod
    def _score_with_ranker(ranker, article: str, titles: List[str], ranker_name: str) -> np.ndarray:
        """
        Run a ranker and validate the returned score vector.
        """

        return validate_scores(
            ranker.score_titles(article, titles),
            expected_len=len(titles),
            context=ranker_name,
        )


    @staticmethod
    def _order_to_scores(order: List[int], n: int) -> np.ndarray:
        """
        Convert a complete order of indices into descending pseudo-scores.
        """

        return order_to_scores(order, n)

