from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from utils.data_utils import get_source_text_task1, get_titles
from utils.metrics import parse_rank_list


def extract_positive_index(row: pd.Series, y_true_col: str = "y_true") -> int:
    """Devuelve el indice 0-based del titular correcto a partir de y_true.

    y_true contiene una ordenacion de tokens tipo "t9 t3 ...". Para el
    entrenamiento binario se toma como positivo el primer token de esa lista.
    """
    y_true_tokens = parse_rank_list(row[y_true_col])

    if not y_true_tokens:
        raise ValueError("Fila sin y_true válido.")

    top_token = y_true_tokens[0]

    if not isinstance(top_token, str) or not top_token.startswith("t"):
        raise ValueError(f"Token y_true inválido: {top_token!r}")

    try:
        positive_idx = int(top_token[1:]) - 1
    except ValueError as exc:
        raise ValueError(f"Token y_true inválido: {top_token!r}") from exc

    titles = get_titles(row)

    if positive_idx < 0 or positive_idx >= len(titles):
        raise ValueError(
            f"Indice positivo fuera de rango: {positive_idx}. "
            f"Numero de titulares disponibles: {len(titles)}"
        )

    return positive_idx


def build_pairs(df: pd.DataFrame, y_true_col: str = "y_true") -> List[Tuple[str, str, int]]:
    """Convierte filas del dataset en pares (article, title, label).

    Cada articulo genera tantos pares como titulares candidatos. El titular
    situado en primera posicion dentro de y_true recibe etiqueta 1 y el resto
    etiqueta 0.
    """
    pairs: List[Tuple[str, str, int]] = []

    for _, row in df.iterrows():
        article = get_source_text_task1(row)
        titles = get_titles(row)
        positive_idx = extract_positive_index(row, y_true_col=y_true_col)

        for i, title in enumerate(titles):
            label = 1 if i == positive_idx else 0
            pairs.append((article, title, label))

    return pairs
