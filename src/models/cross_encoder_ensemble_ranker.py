from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

from config import TOKENS_ALL, get_cross_encoder_runtime_config
from utils.data_utils import get_source_text_task1, get_titles
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker


def minmax_01(scores):
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


@dataclass
class EnsembleMember:
    model_key: str
    weight: float


class CrossEncoderEnsembleRanker:
    def __init__(self, members: List[EnsembleMember]):
        total = sum(m.weight for m in members)
        self.members = [
            EnsembleMember(m.model_key, m.weight / total)
            for m in members
        ]

        self.rankers = []
        for member in self.members:
            cfg = get_cross_encoder_runtime_config(member.model_key)

            ce_config = CrossEncoderConfig(
                model_name=cfg["model_name"],
                max_length=cfg["max_length"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                epochs=cfg["epochs"],
                weight_decay=cfg["weight_decay"],
                warmup_ratio=cfg["warmup_ratio"],
                use_amp=cfg["use_amp"],
            )

            ranker = CrossEncoderRanker.load(str(cfg["model_dir"]), ce_config)
            self.rankers.append((member, ranker))

    def score_titles(self, article: str, titles: List[str]):
        final_scores = np.zeros(len(titles), dtype=float)

        for member, ranker in self.rankers:
            scores = ranker.score_titles(article, titles)
            final_scores += member.weight * minmax_01(scores)

        return final_scores

    def rank_titles(self, article: str, titles: List[str]):
        scores = self.score_titles(article, titles)
        order = np.argsort(-scores)
        return [TOKENS_ALL[i] for i in order]

    def predict_dataframe(self, df_pred: pd.DataFrame):
        preds = []

        for idx, (_, row) in enumerate(df_pred.iterrows(), start=1):
            article = get_source_text_task1(row)
            titles = get_titles(row)
            preds.append(" ".join(self.rank_titles(article, titles)))

            if idx % 10 == 0:
                print(f"Predichas {idx}/{len(df_pred)} filas...")

        return preds
