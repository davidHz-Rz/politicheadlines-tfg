from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from config import (
    CLIP_MODEL_NAME,
    IMAGE_WEIGHT,
    SIGLIP_MODEL_NAME,
    TEXT_WEIGHT,
    TOKENS_ALL,
    VLM_BACKEND,
    get_vlm_model_name,
)
from utils.data_utils import find_image_path, get_titles, minmax_01


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_vlm(
    backend: str = VLM_BACKEND,
    model_name: str | None = None,
):
    backend = backend.lower().strip()
    device = get_device()
    resolved_model_name = model_name or get_vlm_model_name()

    if backend == "clip":
        model = CLIPModel.from_pretrained(resolved_model_name).to(device)
        processor = CLIPProcessor.from_pretrained(resolved_model_name)
    elif backend == "siglip":
        model = AutoModel.from_pretrained(resolved_model_name).to(device)
        processor = AutoProcessor.from_pretrained(resolved_model_name)
    else:
        raise ValueError(f"Backend VLM no soportado: {backend}")

    model.eval()
    return model, processor, device


def load_clip(model_name: str = CLIP_MODEL_NAME):
    return load_vlm(backend="clip", model_name=model_name)


def load_siglip(model_name: str = SIGLIP_MODEL_NAME):
    return load_vlm(backend="siglip", model_name=model_name)


def ranking_from_scores(scores: Sequence[float]) -> str:
    """Convierte scores por título en ranking tokenizado t1 ... t10."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    return " ".join(TOKENS_ALL[i] for i in order)


@torch.inference_mode()
def vlm_logits_image_vs_titles(
    vlm_model,
    vlm_processor,
    device: str,
    image_path: Path,
    titles: List[str],
    backend: str = VLM_BACKEND,
) -> np.ndarray:
    backend = backend.lower().strip()
    image = Image.open(image_path).convert("RGB")

    processor_kwargs = dict(
        text=titles,
        images=image,
        return_tensors="pt",
        truncation=True,
    )

    if backend == "clip":
        processor_kwargs.update(padding=True, max_length=77)
    elif backend == "siglip":
        # La documentación oficial recomienda padding=max_length para SigLIP.
        processor_kwargs.update(padding="max_length")
    else:
        raise ValueError(f"Backend VLM no soportado: {backend}")

    inputs = vlm_processor(**processor_kwargs).to(device)
    outputs = vlm_model(**inputs)
    return outputs.logits_per_image[0].detach().float().cpu().numpy()


@torch.inference_mode()
def clip_logits_image_vs_titles(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    image_path: Path,
    titles: List[str],
) -> np.ndarray:
    return vlm_logits_image_vs_titles(
        vlm_model=clip_model,
        vlm_processor=clip_processor,
        device=device,
        image_path=image_path,
        titles=titles,
        backend="clip",
    )


@torch.inference_mode()
def siglip_logits_image_vs_titles(
    siglip_model,
    siglip_processor,
    device: str,
    image_path: Path,
    titles: List[str],
) -> np.ndarray:
    return vlm_logits_image_vs_titles(
        vlm_model=siglip_model,
        vlm_processor=siglip_processor,
        device=device,
        image_path=image_path,
        titles=titles,
        backend="siglip",
    )


@torch.inference_mode()
def vlm_rank_titles_for_row(
    vlm_model,
    vlm_processor,
    device: str,
    image_path: Path,
    titles: List[str],
    backend: str = VLM_BACKEND,
) -> List[str]:
    logits = vlm_logits_image_vs_titles(
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
        device=device,
        image_path=image_path,
        titles=titles,
        backend=backend,
    )
    order = np.argsort(-logits)
    return [TOKENS_ALL[i] for i in order]


def predict_task2_vlm(
    df_pred: pd.DataFrame,
    images_dir: Path,
    vlm_model,
    vlm_processor,
    device: str,
    backend: str = VLM_BACKEND,
) -> List[str]:
    """Ranking únicamente visual. Se conserva por compatibilidad."""
    preds = []
    missing_images = 0

    for _, row in df_pred.iterrows():
        titles = get_titles(row)
        img_path = find_image_path(images_dir, row.get("image_hash", ""))

        if img_path is None:
            missing_images += 1
            preds.append(" ".join(TOKENS_ALL))
            continue

        preds.append(
            " ".join(
                vlm_rank_titles_for_row(
                    vlm_model=vlm_model,
                    vlm_processor=vlm_processor,
                    device=device,
                    image_path=img_path,
                    titles=titles,
                    backend=backend,
                )
            )
        )

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used identity ranking.")

    return preds


def predict_task2_clip(
    df_pred: pd.DataFrame,
    images_dir: Path,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
) -> List[str]:
    return predict_task2_vlm(
        df_pred=df_pred,
        images_dir=images_dir,
        vlm_model=clip_model,
        vlm_processor=clip_processor,
        device=device,
        backend="clip",
    )


def predict_task2_vlm_plus_text_scores(
    df_pred: pd.DataFrame,
    images_dir: Path,
    text_scores_list: Sequence[Sequence[float]],
    vlm_model,
    vlm_processor,
    device: str,
    backend: str = VLM_BACKEND,
    w_text: float = TEXT_WEIGHT,
    w_img: float = IMAGE_WEIGHT,
) -> List[str]:
    """
    Función genérica para Task 2.

    Recibe los scores textuales ya calculados para Task 1, añade la señal
    imagen-titulares del VLM y devuelve el ranking multimodal. Así evitamos
    recalcular TF-IDF/BM25/Semantic/CrossEncoder/LLM para Task 2.
    """
    if len(text_scores_list) != len(df_pred):
        raise ValueError(
            "text_scores_list debe tener una entrada por fila de df_pred: "
            f"{len(text_scores_list)} != {len(df_pred)}"
        )

    preds: List[str] = []
    missing_images = 0

    for row_idx, (_, row) in enumerate(df_pred.iterrows()):
        titles = get_titles(row)
        text_scores = np.asarray(text_scores_list[row_idx], dtype=float)

        if len(text_scores) != len(titles):
            raise ValueError(
                f"Fila {row_idx}: número de scores textuales inválido "
                f"({len(text_scores)} scores para {len(titles)} títulos)."
            )

        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        if img_path is None:
            missing_images += 1
            preds.append(ranking_from_scores(text_scores))
            continue

        img_scores = vlm_logits_image_vs_titles(
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            device=device,
            image_path=img_path,
            titles=titles,
            backend=backend,
        )

        final_scores = (w_text * minmax_01(text_scores)) + (w_img * minmax_01(img_scores))
        preds.append(ranking_from_scores(final_scores))

        if (row_idx + 1) % 100 == 0:
            print(f"Predichas Task 2 {row_idx + 1}/{len(df_pred)} filas...")

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only fallback.")

    return preds


# Aliases de compatibilidad para código anterior. Internamente ya no recalculan texto;
# se recomienda usar predict_task2_vlm_plus_text_scores desde run.py.
def predict_task2_crossencoder_plus_vlm(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_crossencoder_plus_vlm está obsoleto. Usa "
        "predict_task2_vlm_plus_text_scores con scores textuales cacheados."
    )


def predict_task2_vlm_plus_bm25(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_vlm_plus_bm25 está obsoleto. Usa "
        "predict_task2_vlm_plus_text_scores con scores textuales cacheados."
    )


def predict_task2_semantic_plus_vlm(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_semantic_plus_vlm está obsoleto. Usa "
        "predict_task2_vlm_plus_text_scores con scores textuales cacheados."
    )


def predict_task2_vlm_plus_tfidf(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_vlm_plus_tfidf está obsoleto. Usa "
        "predict_task2_vlm_plus_text_scores con scores textuales cacheados."
    )


def predict_task2_crossencoder_plus_clip(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_crossencoder_plus_clip está obsoleto. Usa "
        "predict_task2_vlm_plus_text_scores con scores textuales cacheados."
    )


def predict_task2_semantic_plus_clip(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_semantic_plus_clip está obsoleto. Usa "
        "predict_task2_vlm_plus_text_scores con scores textuales cacheados."
    )
