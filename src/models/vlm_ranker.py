from __future__ import annotations

"""
Vision-language ranking utilities for Task 2.

This module loads CLIP/SigLIP-style models, scores image-title pairs and
combines visual scores with cached textual scores. It is used for the
multimodal experiments while preserving the same ranking-token output format
as the textual pipeline.
"""

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
from utils.data_utils import find_image_path, get_titles
from utils.scoring import minmax_01, ranking_from_scores, rank_tokens_from_scores


def get_device() -> str:
    """
    Return the device used for VLM inference.
    """

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_vlm(
    backend: str = VLM_BACKEND,
    model_name: str | None = None,
):
    """
    Load the configured CLIP or SigLIP vision-language model.
    """

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
        raise ValueError(f"Unsupported VLM backend: {backend}")

    model.eval()
    return model, processor, device


def load_clip(model_name: str = CLIP_MODEL_NAME):
    """
    Compatibility helper for loading CLIP.
    """

    return load_vlm(backend="clip", model_name=model_name)


def load_siglip(model_name: str = SIGLIP_MODEL_NAME):
    """
    Compatibility helper for loading SigLIP.
    """

    return load_vlm(backend="siglip", model_name=model_name)


@torch.inference_mode()
def vlm_logits_image_vs_titles(
    vlm_model,
    vlm_processor,
    device: str,
    image_path: Path,
    titles: List[str],
    backend: str = VLM_BACKEND,
) -> np.ndarray:
    """
    Compute image-vs-title logits for the selected VLM backend.
    """

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
        # Official SigLIP examples use padding=max_length.
        processor_kwargs.update(padding="max_length")
    else:
        raise ValueError(f"Unsupported VLM backend: {backend}")

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
    """
    Compute CLIP image-title logits for compatibility with older code.
    """

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
    """
    Compute SigLIP image-title logits for compatibility with older code.
    """

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
    """
    Rank candidate titles using only VLM image-title similarity.
    """

    logits = vlm_logits_image_vs_titles(
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
        device=device,
        image_path=image_path,
        titles=titles,
        backend=backend,
    )
    return rank_tokens_from_scores(logits)


def predict_task2_vlm(
    df_pred: pd.DataFrame,
    images_dir: Path,
    vlm_model,
    vlm_processor,
    device: str,
    backend: str = VLM_BACKEND,
) -> List[str]:
    """
    Used for isolated testing: produce a visual-only Task 2 ranking.
    """
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
    """
    Compatibility helper for visual-only CLIP predictions.
    """

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
    Generic multimodal prediction function for Task 2.

    The function receives the textual scores already computed for Task 1,
    adds image-title similarity scores from the VLM, and returns the final
    multimodal ranking. This avoids recalculating the textual model for Task 2.
    """
    if len(text_scores_list) != len(df_pred):
        raise ValueError(
            "text_scores_list must contain one entry per row in df_pred: "
            f"{len(text_scores_list)} != {len(df_pred)}"
        )

    preds: List[str] = []
    missing_images = 0

    for row_idx, (_, row) in enumerate(df_pred.iterrows()):
        titles = get_titles(row)
        text_scores = np.asarray(text_scores_list[row_idx], dtype=float)

        if len(text_scores) != len(titles):
            raise ValueError(
                f"Row {row_idx}: invalid number of textual scores "
                f"({len(text_scores)} scores for {len(titles)} titles)."
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
            print(f"Predicted Task 2 {row_idx + 1}/{len(df_pred)} rows...")

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only fallback.")

    return preds


# Backwards-compatibility aliases for older code. They no longer recalculate text;
# use predict_task2_vlm_plus_text_scores with cached textual scores from run.py.
def predict_task2_crossencoder_plus_vlm(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_crossencoder_plus_vlm is deprecated. Use "
        "predict_task2_vlm_plus_text_scores with cached textual scores."
    )


def predict_task2_vlm_plus_bm25(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_vlm_plus_bm25 is deprecated. Use "
        "predict_task2_vlm_plus_text_scores with cached textual scores."
    )


def predict_task2_semantic_plus_vlm(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_semantic_plus_vlm is deprecated. Use "
        "predict_task2_vlm_plus_text_scores with cached textual scores."
    )


def predict_task2_vlm_plus_tfidf(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_vlm_plus_tfidf is deprecated. Use "
        "predict_task2_vlm_plus_text_scores with cached textual scores."
    )


def predict_task2_crossencoder_plus_clip(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_crossencoder_plus_clip is deprecated. Use "
        "predict_task2_vlm_plus_text_scores with cached textual scores."
    )


def predict_task2_semantic_plus_clip(*args, **kwargs):
    raise RuntimeError(
        "predict_task2_semantic_plus_clip is deprecated. Use "
        "predict_task2_vlm_plus_text_scores with cached textual scores."
    )
