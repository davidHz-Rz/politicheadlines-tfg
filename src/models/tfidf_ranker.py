from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import TITLE_COLS, TOKENS_ALL
from utils.data_utils import get_source_text_task1, get_titles


class TfidfRanker:
    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=1,
        max_features=100000,
    ):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            norm="l2",
        )

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)

    def score_titles(self, source_text: str, titles: List[str]) -> np.ndarray:
        docs = [source_text] + titles
        X = self.vectorizer.transform(docs)
        q = X[0]
        T = X[1:]
        scores = (T @ q.T).toarray().ravel()
        return scores

    def rank_titles(self, source_text: str, titles: List[str]) -> List[str]:
        scores = self.score_titles(source_text, titles)
        order = np.argsort(-scores)
        return [TOKENS_ALL[i] for i in order]


def build_tfidf_corpus(df: pd.DataFrame) -> List[str]:
    corpus = []

    for _, row in df.iterrows():
        corpus.append(get_source_text_task1(row))
        corpus.extend(get_titles(row))

    return corpus


def predict_tfidf(
    df_train: pd.DataFrame,
    df_pred: pd.DataFrame,
) -> List[str]:
    ranker = TfidfRanker()

    corpus = build_tfidf_corpus(df_train)
    ranker.fit(corpus)

    preds = []
    for _, row in df_pred.iterrows():
        source_text = get_source_text_task1(row)
        titles = get_titles(row)
        preds.append(" ".join(ranker.rank_titles(source_text, titles)))

    return preds