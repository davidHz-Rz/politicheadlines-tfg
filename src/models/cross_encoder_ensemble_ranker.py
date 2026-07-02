"""
Weighted ensemble for trained pointwise cross-encoder rankers.

Each member model produces one relevance score per candidate headline. Scores
are normalized per article and combined linearly with the configured weights.
The resulting scores can then be sorted to obtain the final ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from config import get_cross_encoder_runtime_config
from utils.data_utils import get_source_text_task1, get_titles
from utils.scoring import minmax_01, rank_tokens_from_scores, validate_scores
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker


@dataclass
class EnsembleMember:
    """
    Model identifier and normalized contribution weight for an ensemble member
    """

    model_key: str
    weight: float


class CrossEncoderEnsembleRanker:
    """
    Weighted soft-voting ensemble of trained cross-encoder rankers.

    All members must follow the same interface as CrossEncoderRanker:
    score_titles(article, titles) -> one score per candidate headline.
    """

    def __init__(self, members: List[EnsembleMember]):
        """
        Load all configured cross-encoder members and normalize their weights.
        """
        if not members:
            raise ValueError("The ensemble must contain at least one model.")

        for member in members:
            if member.weight < 0:
                raise ValueError(
                    f"Negative ensemble weight is not allowed: "
                    f"{member.model_key}={member.weight}"
                )

        total = sum(member.weight for member in members)
        if total <= 0:
            raise ValueError("The sum of ensemble weights must be greater than 0.")

        self.members = [
            EnsembleMember(member.model_key, member.weight / total)
            for member in members
        ]

        self.rankers = []
        for member in self.members:
            cfg = get_cross_encoder_runtime_config(member.model_key)

            ce_config = CrossEncoderConfig(
                model_name=str(cfg["model_name"]),
                max_length=cfg["max_length"],
                batch_size=cfg["batch_size"],
                gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
                learning_rate=cfg["learning_rate"],
                epochs=cfg["epochs"],
                weight_decay=cfg["weight_decay"],
                warmup_ratio=cfg["warmup_ratio"],
                use_amp=cfg["use_amp"],
                use_head_tail=cfg.get("use_head_tail", False),
                head_tokens=cfg.get("head_tokens", 384),
                tail_tokens=cfg.get("tail_tokens", 125),
            )

            ranker = CrossEncoderRanker.load(str(cfg["model_dir"]), ce_config)
            self.rankers.append((member, ranker))


    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        """
        Return combined ensemble scores for the candidate headlines.

        Scores from each member are min-max normalized per row and then summed
        using the normalized ensemble weights.
        """
        final_scores = np.zeros(len(titles), dtype=float)

        for member, ranker in self.rankers:
            scores = validate_scores(
                ranker.score_titles(article, titles),
                expected_len=len(titles),
                context=f"Model {member.model_key}",
            )

            final_scores += member.weight * minmax_01(scores)

        return final_scores


    def rank_titles(self, article: str, titles: List[str]) -> List[str]:
        """
        Return title tokens ordered by descending ensemble score.
        """
        return rank_tokens_from_scores(self.score_titles(article, titles))


    def predict_dataframe(self, df_pred: pd.DataFrame) -> List[str]:
        """
        Used for isolated testing.
        """
        preds = []

        for idx, (_, row) in enumerate(df_pred.iterrows(), start=1):
            article = get_source_text_task1(row)
            titles = get_titles(row)
            preds.append(" ".join(self.rank_titles(article, titles)))

            if idx % 10 == 0:
                print(f"Predicted {idx}/{len(df_pred)} rows...")

        return preds


