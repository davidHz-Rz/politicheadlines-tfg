from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from config import (
    CLIP_MODEL_NAME,
    IMAGE_WEIGHT,
    SIGLIP_MODEL_NAME,
    TEXT_WEIGHT,
    TITLE_COLS,
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
    logits = outputs.logits_per_image[0].detach().float().cpu().numpy()
    return logits


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
    preds = []
    missing_images = 0

    for _, row in df_pred.iterrows():
        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        titles = get_titles(row)

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


def build_text_vectorizer(
    df_train: pd.DataFrame,
    max_features: int = 50_000,
) -> TfidfVectorizer:
    corpus = []

    for _, row in df_train.iterrows():
        corpus.append(str(row.get("article_body", "") or ""))
        corpus.extend([str(row.get(col, "") or "") for col in TITLE_COLS])

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
    )
    vectorizer.fit(corpus)
    return vectorizer


def predict_task2_vlm_plus_tfidf(
    df_train: pd.DataFrame,
    df_pred: pd.DataFrame,
    images_dir: Path,
    vlm_model,
    vlm_processor,
    device: str,
    backend: str = VLM_BACKEND,
    w_text: float = TEXT_WEIGHT,
    w_img: float = IMAGE_WEIGHT,
    max_features: int = 50_000,
) -> List[str]:
    vectorizer = build_text_vectorizer(df_train, max_features=max_features)

    preds = []
    missing_images = 0

    for _, row in df_pred.iterrows():
        titles = get_titles(row)

        source_text = str(row.get("article_body", "") or "")
        src_vec = vectorizer.transform([source_text])
        title_vecs = vectorizer.transform(titles)
        sims_text = cosine_similarity(src_vec, title_vecs)[0]

        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        if img_path is None:
            missing_images += 1
            order = np.argsort(-sims_text)
            preds.append(" ".join([TOKENS_ALL[i] for i in order]))
            continue

        sims_img = vlm_logits_image_vs_titles(
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            device=device,
            image_path=img_path,
            titles=titles,
            backend=backend,
        )

        text01 = minmax_01(sims_text)
        img01 = minmax_01(sims_img)

        scores = (w_text * text01) + (w_img * img01)
        order = np.argsort(-scores)

        preds.append(" ".join([TOKENS_ALL[i] for i in order]))

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only fallback.")

    return preds


def predict_task2_clip_plus_tfidf(
    df_train: pd.DataFrame,
    df_pred: pd.DataFrame,
    images_dir: Path,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    w_text: float = TEXT_WEIGHT,
    w_img: float = IMAGE_WEIGHT,
    max_features: int = 50_000,
) -> List[str]:
    return predict_task2_vlm_plus_tfidf(
        df_train=df_train,
        df_pred=df_pred,
        images_dir=images_dir,
        vlm_model=clip_model,
        vlm_processor=clip_processor,
        device=device,
        backend="clip",
        w_text=w_text,
        w_img=w_img,
        max_features=max_features,
    )


def predict_task2_semantic_plus_vlm(
    df_pred: pd.DataFrame,
    images_dir: Path,
    semantic_ranker,
    vlm_model,
    vlm_processor,
    device: str,
    backend: str = VLM_BACKEND,
    w_text: float = 0.96,
    w_img: float = 0.04,
) -> List[str]:
    preds = []
    missing_images = 0

    for _, row in df_pred.iterrows():
        article = str(row.get("article_body", "") or "")
        titles = get_titles(row)

        text_scores = semantic_ranker.score_titles(article, titles)

        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        if img_path is None:
            missing_images += 1
            order = np.argsort(-text_scores)
            preds.append(" ".join([TOKENS_ALL[i] for i in order]))
            continue

        img_scores = vlm_logits_image_vs_titles(
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            device=device,
            image_path=img_path,
            titles=titles,
            backend=backend,
        )

        text01 = minmax_01(text_scores)
        img01 = minmax_01(img_scores)

        final_scores = (w_text * text01) + (w_img * img01)

        order = np.argsort(-final_scores)
        preds.append(" ".join([TOKENS_ALL[i] for i in order]))

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only fallback.")

    return preds


def predict_task2_semantic_plus_clip(
    df_pred: pd.DataFrame,
    images_dir: Path,
    semantic_ranker,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    w_text: float = 0.96,
    w_img: float = 0.04,
) -> List[str]:
    return predict_task2_semantic_plus_vlm(
        df_pred=df_pred,
        images_dir=images_dir,
        semantic_ranker=semantic_ranker,
        vlm_model=clip_model,
        vlm_processor=clip_processor,
        device=device,
        backend="clip",
        w_text=w_text,
        w_img=w_img,
    )


def predict_task2_crossencoder_plus_vlm(
    df_pred: pd.DataFrame,
    images_dir: Path,
    cross_encoder_ranker,
    vlm_model,
    vlm_processor,
    device: str,
    backend: str = VLM_BACKEND,
    w_text: float = 0.90,
    w_img: float = 0.10,
) -> List[str]:
    preds = []
    missing_images = 0

    for _, row in df_pred.iterrows():
        article = str(row.get("article_body", "") or "")
        titles = get_titles(row)

        text_scores = cross_encoder_ranker.score_titles(article, titles)

        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        if img_path is None:
            missing_images += 1
            order = np.argsort(-text_scores)
            preds.append(" ".join([TOKENS_ALL[i] for i in order]))
            continue

        img_scores = vlm_logits_image_vs_titles(
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            device=device,
            image_path=img_path,
            titles=titles,
            backend=backend,
        )

        text01 = minmax_01(text_scores)
        img01 = minmax_01(img_scores)

        final_scores = (w_text * text01) + (w_img * img01)

        order = np.argsort(-final_scores)
        preds.append(" ".join([TOKENS_ALL[i] for i in order]))

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only fallback.")

    return preds


def predict_task2_crossencoder_plus_clip(
    df_pred: pd.DataFrame,
    images_dir: Path,
    cross_encoder_ranker,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    w_text: float = 0.90,
    w_img: float = 0.10,
) -> List[str]:
    return predict_task2_crossencoder_plus_vlm(
        df_pred=df_pred,
        images_dir=images_dir,
        cross_encoder_ranker=cross_encoder_ranker,
        vlm_model=clip_model,
        vlm_processor=clip_processor,
        device=device,
        backend="clip",
        w_text=w_text,
        w_img=w_img,
    )
