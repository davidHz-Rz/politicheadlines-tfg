"""
TF-IDF lexical ranker for the PoliticHeadlinES project.

This module implements a simple lexical baseline based on TF-IDF vectors. The
article body and the ten candidate headlines are represented in the same
vector space, and candidate headlines are ranked by cosine similarity to the
article vector.

The class follows the same interface as the rest of the project rankers:

    score_titles(source_text, titles) -> np.ndarray
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.data_utils import get_source_text_task1, get_titles
from utils.scoring import rank_tokens_from_scores


class TfidfRanker:
    """
    TF-IDF baseline for ranking candidate headlines."""

    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=1,
        max_features=100000,
    ):
        """Create the underlying TF-IDF vectorizer.

        The default configuration uses unigrams and bigrams, lowercasing,
        Unicode accent stripping and L2-normalized vectors.
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            norm="l2",
        )

    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF vocabulary and IDF statistics."""
        self.vectorizer.fit(texts)

    def score_titles(self, source_text: str, titles: List[str]) -> np.ndarray:
        """Return one cosine-similarity score per candidate headline.

        Because TF-IDF vectors are L2-normalized, the dot product between each
        title vector and the article vector is equivalent to cosine similarity.
        """
        docs = [source_text] + titles
        X = self.vectorizer.transform(docs)

        article_vector = X[0]
        title_vectors = X[1:]

        scores = (title_vectors @ article_vector.T).toarray().ravel()
        return scores

    def rank_titles(self, source_text: str, titles: List[str]) -> List[str]:
        """Rank candidate headlines and return their tokens in descending order."""
        return rank_tokens_from_scores(self.score_titles(source_text, titles))


def build_tfidf_corpus(df: pd.DataFrame) -> List[str]:
    """Build the corpus used to fit the TF-IDF vectorizer.

    The corpus includes article bodies and candidate headlines from the training
    split so that the vectorizer sees vocabulary from both types of text.
    """
    corpus: List[str] = []

    for _, row in df.iterrows():
        corpus.append(get_source_text_task1(row))
        corpus.extend(get_titles(row))

    return corpus


def predict_tfidf(
    df_train: pd.DataFrame,
    df_pred: pd.DataFrame,
) -> List[str]:
    """Fit the TF-IDF ranker and predict rankings for a dataframe."""
    ranker = TfidfRanker()

    corpus = build_tfidf_corpus(df_train)
    ranker.fit(corpus)

    preds = []

    for _, row in df_pred.iterrows():
        source_text = get_source_text_task1(row)
        titles = get_titles(row)
        preds.append(" ".join(ranker.rank_titles(source_text, titles)))

    return preds
