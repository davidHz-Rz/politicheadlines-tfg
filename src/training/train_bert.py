from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (
    TRAIN_CSV,
    TEST_CSV,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    BERT_MAX_LENGTH,
    BERT_BATCH_SIZE,
    BERT_LEARNING_RATE,
    BERT_EPOCHS,
    BERT_WEIGHT_DECAY,
    BERT_WARMUP_RATIO,
    BERT_USE_AMP,
)

from utils.data_utils import validate_columns
from models.bert_ranker import (
    CrossEncoderConfig,
    CrossEncoderRanker,
    build_pair_examples,
)


def main() -> None:
    print("Cargando datos...")

    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(TEST_CSV)

    validate_columns(train_df)
    validate_columns(dev_df)

    print(f"Train rows: {len(train_df)}")
    print(f"Dev rows: {len(dev_df)}")

    print("Construyendo pares de entrenamiento...")

    train_examples = build_pair_examples(train_df)
    dev_examples = build_pair_examples(dev_df)

    print(f"Número de pares train: {len(train_examples)}")
    print(f"Número de pares dev: {len(dev_examples)}")

    config = CrossEncoderConfig(
        model_name=BERT_MODEL_NAME,
        max_length=BERT_MAX_LENGTH,
        batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LEARNING_RATE,
        epochs=BERT_EPOCHS,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_ratio=BERT_WARMUP_RATIO,
        use_amp=BERT_USE_AMP,
    )

    print("Inicializando modelo...")
    ranker = CrossEncoderRanker(config)

    print(f"Dispositivo: {ranker.device}")
    print("Entrenando modelo...")

    ranker.fit(
        train_examples=train_examples,
        val_examples=dev_examples,
    )

    print("Guardando modelo...")
    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    ranker.save(str(BERT_MODEL_DIR))

    print(f"Modelo guardado en: {BERT_MODEL_DIR}")


if __name__ == "__main__":
    main()