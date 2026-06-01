from __future__ import annotations

from typing import List

import numpy as np

class TailReranker:
    """
    Reranker de dos etapas.

    Primero obtiene una ordenación base. Después conserva fijo el primer
    candidato y reordena únicamente la cola de la predicción con un segundo
    ranker. El resultado se devuelve como puntuaciones descendentes para que
    el resto del pipeline pueda seguir usando argsort(-scores).
    """

    def __init__(self, base_ranker, tail_ranker, top_k: int = 10):
        if base_ranker is None:
            raise ValueError("base_ranker no puede ser None.")
        if tail_ranker is None:
            raise ValueError("tail_ranker no puede ser None.")
        if top_k < 2:
            raise ValueError("top_k debe ser al menos 2 para poder rerankear la cola.")

        self.base_ranker = base_ranker
        self.tail_ranker = tail_ranker
        self.top_k = top_k

    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
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
        scores = np.asarray(ranker.score_titles(article, titles), dtype=float)

        if len(scores) != len(titles):
            raise ValueError(
                f"{ranker_name} devolvió {len(scores)} scores para "
                f"{len(titles)} títulos."
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{ranker_name} devolvió scores no finitos.")

        return scores

    @staticmethod
    def _order_to_scores(order: List[int], n: int) -> np.ndarray:
        if len(order) != n:
            raise ValueError(
                f"La ordenación final tiene {len(order)} elementos, pero se esperaban {n}."
            )
        if sorted(order) != list(range(n)):
            raise ValueError("La ordenación final no es una permutación válida de los candidatos.")

        scores = np.zeros(n, dtype=float)
        for rank, idx in enumerate(order):
            scores[idx] = float(n - rank)
        return scores

