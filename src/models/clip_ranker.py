from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor

from config import (
    CLIP_MODEL_NAME,
    IMAGE_WEIGHT,
    TEXT_WEIGHT,
    TITLE_COLS,
    TOKENS_ALL,
)
from data_utils import find_image_path, get_titles, minmax_01


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(model_name: str = CLIP_MODEL_NAME) -> Tuple[CLIPModel, CLIPProcessor, str]:
    device = get_device()
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device


@torch.inference_mode()
def clip_logits_image_vs_titles(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    image_path: Path,
    titles: List[str],
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(
        text=titles,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0].detach().float().cpu().numpy()
    return logits


@torch.inference_mode()
def clip_rank_titles_for_row(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    image_path: Path,
    titles: List[str],
) -> List[str]:
    logits = clip_logits_image_vs_titles(
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
        image_path=image_path,
        titles=titles,
    )
    order = np.argsort(-logits)
    return [TOKENS_ALL[i] for i in order]


def predict_task2_clip(
    df_pred: pd.DataFrame,
    images_dir: Path,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
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
                clip_rank_titles_for_row(
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=device,
                    image_path=img_path,
                    titles=titles,
                )
            )
        )

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used identity ranking.")

    return preds


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
    vectorizer = build_text_vectorizer(df_train, max_features=max_features)

    preds = []
    missing_images = 0

    for _, row in df_pred.iterrows():
        titles = get_titles(row)

        # Text score
        source_text = str(row.get("article_body", "") or "")
        src_vec = vectorizer.transform([source_text])
        title_vecs = vectorizer.transform(titles)
        sims_text = cosine_similarity(src_vec, title_vecs)[0]

        # Image score
        img_path = find_image_path(images_dir, row.get("image_hash", ""))
        if img_path is None:
            missing_images += 1
            order = np.argsort(-sims_text)
            preds.append(" ".join([TOKENS_ALL[i] for i in order]))
            continue

        sims_img = clip_logits_image_vs_titles(
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            image_path=img_path,
            titles=titles,
        )

        # Normalize per row
        text01 = minmax_01(sims_text)
        img01 = minmax_01(sims_img)

        # Weighted fusion
        scores = (w_text * text01) + (w_img * img01)
        order = np.argsort(-scores)

        preds.append(" ".join([TOKENS_ALL[i] for i in order]))

    if missing_images:
        print(f"[WARN] Missing images for {missing_images} rows. Used text-only fallback.")

    return preds