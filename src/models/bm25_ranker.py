from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, List

import numpy as np
import pandas as pd

from config import TOKENS_ALL
from utils.data_utils import get_source_text_task1, get_titles

_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize_bm25(text: str) -> List[str]:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _TOKEN_RE.findall(text)


class BM25Ranker:
    def __init__(self, k1: float = 1.5, b: float = 0.75, query_term_limit: int | None = 256):
        self.k1 = float(k1)
        self.b = float(b)
        self.query_term_limit = query_term_limit
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def fit(self, texts: Iterable[str]) -> None:
        tokenized_docs = [tokenize_bm25(text) for text in texts]
        self.n_docs = len(tokenized_docs)
        if self.n_docs == 0:
            raise ValueError("No se puede entrenar BM25 con un corpus vacío.")

        doc_freq: Counter[str] = Counter()
        total_len = 0
        for tokens in tokenized_docs:
            total_len += len(tokens)
            doc_freq.update(set(tokens))

        self.avgdl = total_len / max(self.n_docs, 1)
        self.idf = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def _query_terms(self, source_text: str) -> List[str]:
        counts = Counter(tokenize_bm25(source_text))
        terms = [term for term in counts if term in self.idf]
        if self.query_term_limit is not None and len(terms) > self.query_term_limit:
            terms = sorted(
                terms,
                key=lambda term: counts[term] * self.idf.get(term, 0.0),
                reverse=True,
            )[: self.query_term_limit]
        return terms

    def score_titles(self, source_text: str, titles: List[str]) -> np.ndarray:
        query_terms = self._query_terms(source_text)
        scores = []
        for title in titles:
            doc_tokens = tokenize_bm25(title)
            doc_len = len(doc_tokens)
            freqs = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-12))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            scores.append(score)
        return np.asarray(scores, dtype=float)

    def rank_titles(self, source_text: str, titles: List[str]) -> List[str]:
        scores = self.score_titles(source_text, titles)
        order = np.argsort(-scores)
        return [TOKENS_ALL[i] for i in order]


def build_bm25_corpus(df: pd.DataFrame) -> List[str]:
    corpus: List[str] = []
    for _, row in df.iterrows():
        corpus.append(get_source_text_task1(row))
        corpus.extend(get_titles(row))
    return corpus


def predict_bm25(
    df_train: pd.DataFrame,
    df_pred: pd.DataFrame,
    k1: float = 1.5,
    b: float = 0.75,
    query_term_limit: int | None = 256,
) -> List[str]:
    ranker = BM25Ranker(k1=k1, b=b, query_term_limit=query_term_limit)
    ranker.fit(build_bm25_corpus(df_train))
    preds = []
    for _, row in df_pred.iterrows():
        source_text = get_source_text_task1(row)
        titles = get_titles(row)
        preds.append(" ".join(ranker.rank_titles(source_text, titles)))
    return preds
