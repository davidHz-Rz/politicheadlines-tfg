from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import TOKENS_ALL
from data_utils import get_source_text_task1, get_titles


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class SemanticRanker:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def score_titles(self, source_text: str, titles: List[str]) -> np.ndarray:
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

        # Con embeddings normalizados, el producto escalar equivale a cosine similarity
        scores = np.dot(title_embs, source_emb[0])
        return scores

    def rank_titles(self, source_text: str, titles: List[str]) -> List[str]:
        scores = self.score_titles(source_text, titles)
        order = np.argsort(-scores)
        return [TOKENS_ALL[i] for i in order]


def predict_semantic(
    df_pred: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[str]:
    ranker = SemanticRanker(model_name=model_name)

    preds = []
    for _, row in df_pred.iterrows():
        source_text = get_source_text_task1(row)
        titles = get_titles(row)
        preds.append(" ".join(ranker.rank_titles(source_text, titles)))

    return preds
