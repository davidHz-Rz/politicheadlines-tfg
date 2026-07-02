from __future__ import annotations

"""
Main inference and evaluation pipeline for the PoliticHeadlinES project

This script loads the dataset, builds or loads the ranking model, generates
predictions for task 1, applies multimodal fusion for task 2 if active, saves
the submission file with the rankings, and computes local metrics if golden
labels are available.

Model selection and other parameters are defined in config.py.
"""

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (  # noqa: E402
    ALPHA,
    IMAGE_WEIGHT,
    IMAGES_DIR,
    MODEL_NAME,
    NDCG_K,
    OUTPUT_METRICS,
    OUTPUT_SUBMISSION,
    TEST_CSV,
    TEXT_WEIGHT,
    TRAIN_CSV,
    USE_VLM_FOR_TASK2,
    VLM_BACKEND,
    get_vlm_model_name,
)
from models.factory import build_ranker  # noqa: E402
from models.vlm_ranker import load_vlm, predict_task2_vlm_plus_text_scores  # noqa: E402
from utils.data_utils import validate_columns  # noqa: E402
from utils.inference import score_dataframe_with_ranker  # noqa: E402
from utils.metrics import score_submission  # noqa: E402
from utils.submission import build_submission, save_submission, validate_submission  # noqa: E402


def main() -> None:
    """
    Execute the full configured pipeline.

    The function loads train/test data, validates the expected columns, builds
    the selected ranker, generates Task 1 predictions, optionally generates
    multimodal Task 2 predictions, saves the submission and evaluates it when
    ground truth labels are available.
    """
    # DATA LOAD
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    validate_columns(train_df)
    validate_columns(test_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Model: {MODEL_NAME}")

    # Textual model (Task 1)
    ranker = build_ranker(MODEL_NAME, train_df=train_df)

    print(f"Generating predictions for Task 1 ({MODEL_NAME})...")
    task_1_preds, text_scores_cache = score_dataframe_with_ranker(
        ranker,
        test_df,
        progress_every=10,
        progress_label="Predicted Task 1",
    )

    # Optional visual fusion for Task 2
    if USE_VLM_FOR_TASK2:
        vlm_name = get_vlm_model_name()
        print(f"Loading {VLM_BACKEND.upper()} model: {vlm_name}...")
        vlm_model, vlm_processor, device = load_vlm()
        print(f"Device {VLM_BACKEND.upper()}: {device}")

        print(
            f"Generating predictions for Task 2 "
            f"({MODEL_NAME} loaded from cache + {VLM_BACKEND.upper()})..."
        )
        task_2_preds = predict_task2_vlm_plus_text_scores(
            df_pred=test_df,
            images_dir=IMAGES_DIR,
            text_scores_list=text_scores_cache,
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            backend=VLM_BACKEND,
            device=device,
            w_text=TEXT_WEIGHT,
            w_img=IMAGE_WEIGHT,
        )
    else:
        print("USE_VLM_FOR_TASK2 = False. Reusing Task 1 scores for Task 2.")
        task_2_preds = task_1_preds

    # SUBMISSION
    
    print("Building submission file...")
    submission = build_submission(
        ids=test_df["id"].astype(str),
        task_1_preds=task_1_preds,
        task_2_preds=task_2_preds,
    )

    validate_submission(submission)

    OUTPUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    save_submission(submission, output_path=OUTPUT_SUBMISSION)

    print(submission.head())

    # LOCAL EVALUATION
    
    if "y_true" in test_df.columns:
        print("Evaluating submission...")
        scores = score_submission(
            validation_csv=str(TEST_CSV),
            results_csv=str(OUTPUT_SUBMISSION),
            k=NDCG_K,
            alpha=ALPHA,
        )

        print("\nResults:")
        for key, value in scores.items():
            print(f"{key}: {value}")

        with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        print(f"Saved metrics in: {OUTPUT_METRICS}")
    else:
        print("No 'y_true' column was found. Skipping local evaluation.")


if __name__ == "__main__":
    main()


