"""Semantic embedding ranker for the PoliticHeadlinES project.

This module implements a sentence-transformer baseline. The article body and
the ten candidate headlines are encoded into the same embedding space, and
candidate headlines are ranked by cosine similarity to the article.

NOTE: The model is not fine-tuned in this project.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils.data_utils import get_source_text_task1, get_titles
from utils.scoring import rank_tokens_from_scores


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class SemanticRanker:
    """
    Rank headlines by semantic similarity to the article body.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """Load the sentence-transformer model used to compute embeddings."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)


    def score_titles(self, source_text: str, titles: List[str]) -> np.ndarray:
        """
        Return one cosine-similarity score per candidate headline.
        """
        source_emb = self.model.encode(
            [source_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        title_embs = self.model.encode(
            titles,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = np.dot(title_embs, source_emb[0])
        return scores


    def rank_titles(self, source_text: str, titles: List[str]) -> List[str]:
        """
        Rank candidate headlines and return their tokens in descending order.
        """
        return rank_tokens_from_scores(self.score_titles(source_text, titles))


def predict_semantic(
    df_pred: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[str]:
    """
    Predict semantic rankings for all rows in a dataframe.
    
    Used for isolated testing.
    """
    ranker = SemanticRanker(model_name=model_name)

    preds = []

    for _, row in df_pred.iterrows():
        source_text = get_source_text_task1(row)
        titles = get_titles(row)
        preds.append(" ".join(ranker.rank_titles(source_text, titles)))

    return preds
