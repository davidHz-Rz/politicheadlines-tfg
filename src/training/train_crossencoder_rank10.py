from __future__ import annotations

"""
Training entry point for the rank10 cross-encoder.

This script trains a regression-based cross-encoder that uses the full ground
truth ranking instead of only the top-1 headline. Each article-title pair is
assigned a graded relevance value according to the candidate position in
``y_true``.

Usage examples
--------------
Train the default rank10 configuration:
    python src/training/train_crossencoder_rank10.py

Train a specific rank10 configuration:
    python src/training/train_crossencoder_rank10.py beto_rank10
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import SEED, VAL_CSV, TRAIN_CSV, get_cross_encoder_rank10_runtime_config  # noqa: E402
from models.cross_encoder_rank10_ranker import (  # noqa: E402
    CrossEncoderRank10Config,
    CrossEncoderRank10Ranker,
    build_rank10_examples,
)
from utils.data_utils import validate_columns  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def print_training_config(model_key: str, cfg: dict) -> None:
    """
    Print the resolved training parameters for the rank10 model.
    """
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    effective_batch_size = cfg["batch_size"] * gradient_accumulation_steps

    print("\nRank10 training configuration")
    print("=" * 70)
    print(f"Seed:                       {SEED}")
    print(f"Active model:               {model_key}")
    print(f"Base model/checkpoint:      {cfg['model_name']}")
    print(f"Output directory:           {cfg['model_dir']}")
    print(f"Max length:                 {cfg['max_length']}")
    print(f"Batch size:                 {cfg['batch_size']}")
    print(f"Gradient accumulation:      {gradient_accumulation_steps}")
    print(f"Effective batch size:       {effective_batch_size}")
    print(f"Epochs:                     {cfg['epochs']}")
    print(f"Learning rate:              {cfg['learning_rate']}")
    print(f"Weight decay:               {cfg['weight_decay']}")
    print(f"Warmup ratio:               {cfg['warmup_ratio']}")
    print(f"AMP:                        {cfg['use_amp']}")
    print(f"Early stopping patience:    {cfg.get('early_stopping_patience', 3)}")
    print(f"Early stopping min_delta:   {cfg.get('early_stopping_min_delta', 0.0005)}")
    print(f"Early stopping monitor:     {cfg.get('early_stopping_monitor', 'task_1_pa_ndcg')}")
    print("=" * 70)


def save_training_config(model_key: str, cfg: dict, output_dir: Path) -> None:
    """
    Save the rank10 training configuration next to the model checkpoint.
    The JSON file records the paths, hyperparameters, objective name and seed
    used during training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

    training_cfg = {
        "model_key": model_key,
        "model_name": str(cfg["model_name"]),
        "model_dir": str(cfg["model_dir"]),
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
        "max_length": cfg["max_length"],
        "batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": cfg["batch_size"] * gradient_accumulation_steps,
        "learning_rate": cfg["learning_rate"],
        "epochs": cfg["epochs"],
        "weight_decay": cfg["weight_decay"],
        "warmup_ratio": cfg["warmup_ratio"],
        "use_amp": cfg["use_amp"],
        "early_stopping_patience": cfg.get("early_stopping_patience", 3),
        "early_stopping_min_delta": cfg.get("early_stopping_min_delta", 0.0005),
        "early_stopping_monitor": cfg.get("early_stopping_monitor", "task_1_pa_ndcg"),
        "objective": "graded_regression_rank10",
        "seed": SEED,
    }

    config_path = output_dir / "training_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(training_cfg, f, ensure_ascii=False, indent=4)

    print(f"Training configuration saved to: {config_path}")


def main(model_key: str = "beto_rank10") -> None:
    """
    Train the selected rank10 cross-encoder configuration.

    The model key is resolved from CROSS_ENCODER_RANK10_CONFIGS in config.py.
    By default, the script trains the ``beto_rank10`` configuration.
    """
    set_seed(SEED)

    cfg = get_cross_encoder_rank10_runtime_config(model_key)

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(VAL_CSV)

    validate_columns(train_df)
    validate_columns(dev_df)

    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Dev CSV:   {VAL_CSV}")
    print(f"Train rows: {len(train_df)}")
    print(f"Dev rows: {len(dev_df)}")

    print_training_config(model_key, cfg)
    save_training_config(model_key, cfg, Path(cfg["model_dir"]))

    print("Building graded rank10 training examples...")
    train_examples = build_rank10_examples(train_df)
    dev_examples = build_rank10_examples(dev_df) if "y_true" in dev_df.columns else None

    print(f"Train pairs: {len(train_examples)}")
    print(f"Dev pairs: {len(dev_examples) if dev_examples is not None else 0}")

    ranker_config = CrossEncoderRank10Config(
        model_name=str(cfg["model_name"]),
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        use_amp=cfg["use_amp"],
        early_stopping_patience=cfg.get("early_stopping_patience", 3),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 0.0005),
        early_stopping_monitor=cfg.get("early_stopping_monitor", "task_1_pa_ndcg"),
    )

    print("Initializing rank10 model...")
    ranker = CrossEncoderRank10Ranker(ranker_config)
    print(f"Device: {ranker.device}")
    print("Training rank10 model...")

    ranker.fit(
        train_examples=train_examples,
        val_examples=dev_examples,
        val_df=dev_df if "y_true" in dev_df.columns else None,
        output_dir=cfg["model_dir"],
    )

    print(f"Training completed. Best checkpoint saved to: {cfg['model_dir']}")


if __name__ == "__main__":
    selected_model = sys.argv[1] if len(sys.argv) > 1 else "beto_rank10"
    main(selected_model)


