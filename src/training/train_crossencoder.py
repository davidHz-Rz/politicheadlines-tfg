from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import ACTIVE_CROSS_ENCODER, TEST_CSV, TRAIN_CSV, get_cross_encoder_runtime_config
from utils.data_utils import validate_columns
from models.cross_encoder_ranker import CrossEncoderConfig, CrossEncoderRanker, build_pair_examples


def main(model_key: str | None = None) -> None:
    model_key = model_key or ACTIVE_CROSS_ENCODER
    cfg = get_cross_encoder_runtime_config(model_key)

    print("Cargando datos...")
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(TEST_CSV)

    validate_columns(train_df)
    validate_columns(dev_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Dev rows: {len(dev_df)}")
    print(f"Modelo activo: {model_key} -> {cfg['model_name']}")

    print("Construyendo pares de entrenamiento...")
    train_examples = build_pair_examples(train_df)
    dev_examples = build_pair_examples(dev_df) if "y_true" in dev_df.columns else None

    print(f"Número de pares train: {len(train_examples)}")
    print(f"Número de pares dev: {len(dev_examples) if dev_examples is not None else 0}")

    ranker_config = CrossEncoderConfig(
        model_name=cfg["model_name"],
        max_length=cfg["max_length"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        use_amp=cfg["use_amp"],
    )

    print("Inicializando modelo...")
    ranker = CrossEncoderRanker(ranker_config)
    print(f"Dispositivo: {ranker.device}")
    print("Entrenando modelo...")

    ranker.fit(
        train_examples=train_examples,
        val_examples=dev_examples,
        val_df=dev_df if "y_true" in dev_df.columns else None,
        output_dir=cfg["model_dir"],
    )

    print(f"Entrenamiento completado. Mejor checkpoint guardado en: {cfg['model_dir']}")


if __name__ == "__main__":
    selected_model = sys.argv[1] if len(sys.argv) > 1 else None
    main(selected_model)
