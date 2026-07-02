from __future__ import annotations

"""
Training entry point for pointwise cross-encoder models.

This script trains one of the binary cross-encoder configurations defined in
config.py, such as BETO, BETO head-tail, BERTIN or mDeBERTa. Training examples
are generated as article-title pairs, where the headline ranked first in
``y_true`` is treated as the positive class and the remaining candidates as
negative examples.

Usage examples
--------------
Train the active cross-encoder defined in config.py:
    python src/training/train_crossencoder.py

Train a specific configuration:
    python src/training/train_crossencoder.py beto
    python src/training/train_crossencoder.py beto_headtail
    python src/training/train_crossencoder.py bertin
    python src/training/train_crossencoder.py mdeberta
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (  # noqa: E402
    SEED,
    VAL_CSV,
    TRAIN_CSV,
    get_cross_encoder_runtime_config,
)
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker  # noqa: E402
from utils.data_pairs import build_pairs  # noqa: E402
from utils.data_utils import validate_columns  # noqa: E402
from utils.reproducibility import set_seed  # noqa: E402


def print_training_config(model_key: str, cfg: dict) -> None:
    """
    Print the most relevant training parameters for the selected model.
    """
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    effective_batch_size = cfg["batch_size"] * gradient_accumulation_steps
    use_head_tail = cfg.get("use_head_tail", False)

    print("\nTraining configuration")
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
    print(f"Head-tail:                  {use_head_tail}")

    if use_head_tail:
        print(f"Target head tokens:         {cfg.get('head_tokens', 384)}")
        print(f"Target tail tokens:         {cfg.get('tail_tokens', 125)}")

    print("=" * 70)


def save_training_config(model_key: str, cfg: dict, output_dir: Path) -> None:
    """
    Save the resolved training configuration next to the model checkpoint.
    The saved JSON file records the model key, paths, hyperparameters and seed
    used for the run.
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
        "use_head_tail": cfg.get("use_head_tail", False),
        "head_tokens": cfg.get("head_tokens", 384),
        "tail_tokens": cfg.get("tail_tokens", 125),
        "seed": SEED,
    }

    config_path = output_dir / "training_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(training_cfg, f, ensure_ascii=False, indent=4)

    print(f"Training configuration saved to: {config_path}")


def main(model_key: str | None = None) -> None:
    """
    Train the selected pointwise cross-encoder.

    If ``model_key`` is not provided, the script trains the default BETO
    configuration. Otherwise, the provided key is resolved from
    CROSS_ENCODER_CONFIGS.
    """
    set_seed(SEED)

    model_key = model_key or "beto"
    cfg = get_cross_encoder_runtime_config(model_key)

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

    print("Building binary training pairs...")
    train_examples = build_pairs(train_df)
    dev_examples = build_pairs(dev_df) if "y_true" in dev_df.columns else None

    print(f"Train pairs: {len(train_examples)}")
    print(f"Dev pairs: {len(dev_examples) if dev_examples is not None else 0}")

    ranker_config = CrossEncoderConfig(
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
        use_head_tail=cfg.get("use_head_tail", False),
        head_tokens=cfg.get("head_tokens", 384),
        tail_tokens=cfg.get("tail_tokens", 125),
    )

    print("Initializing model...")
    ranker = CrossEncoderRanker(ranker_config)
    print(f"Device: {ranker.device}")
    print("Training model...")

    ranker.fit(
        train_examples=train_examples,
        val_examples=dev_examples,
        val_df=dev_df if "y_true" in dev_df.columns else None,
        output_dir=cfg["model_dir"],
    )

    print(f"Training completed. Best checkpoint saved to: {cfg['model_dir']}")


if __name__ == "__main__":
    selected_model = sys.argv[1] if len(sys.argv) > 1 else None
    main(selected_model)



