from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
import pandas as pd

from config import TOKENS_ALL, get_cross_encoder_runtime_config
from utils.data_utils import get_source_text_task1, get_titles
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker


def minmax_01(scores: Sequence[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        return scores
    if not np.all(np.isfinite(scores)):
        raise ValueError("El ensemble recibió scores no finitos.")

    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        # Si un modelo asigna el mismo score a todos los candidatos, no aporta
        # señal relativa para esa fila. Se devuelve un vector neutro.
        return np.zeros_like(scores)

    return (scores - lo) / (hi - lo)


@dataclass
class EnsembleMember:
    model_key: str
    weight: float


class CrossEncoderEnsembleRanker:
    def __init__(self, members: List[EnsembleMember]):
        if not members:
            raise ValueError("El ensemble debe contener al menos un modelo.")

        for member in members:
            if member.weight < 0:
                raise ValueError(
                    f"Peso negativo no permitido en el ensemble: "
                    f"{member.model_key}={member.weight}"
                )

        total = sum(m.weight for m in members)
        if total <= 0:
            raise ValueError("La suma de pesos del ensemble debe ser mayor que 0.")

        self.members = [
            EnsembleMember(m.model_key, m.weight / total)
            for m in members
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
        final_scores = np.zeros(len(titles), dtype=float)

        for member, ranker in self.rankers:
            scores = np.asarray(ranker.score_titles(article, titles), dtype=float)
            if len(scores) != len(titles):
                raise ValueError(
                    f"El modelo {member.model_key} devolvió {len(scores)} scores "
                    f"para {len(titles)} títulos."
                )
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
