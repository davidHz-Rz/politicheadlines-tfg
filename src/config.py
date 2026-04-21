from pathlib import Path

# ============================================================
# Paths base
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# Dataset activo
# ============================================================

# Opciones:
# - "development_phase_initial"
# - "train_corpora"
# - "test_public"
ACTIVE_DATASET = "train_corpora"

DATASET_DIR = DATA_DIR / ACTIVE_DATASET

# ============================================================
# Archivos principales
# ============================================================

TRAIN_CSV = DATASET_DIR / "train_public.csv"
TEST_CSV = DATASET_DIR / "dev_public.csv"   # dev_public / test_public.csv
IMAGES_DIR = DATASET_DIR / "images"

# ============================================================
# Selección de modelo de inferencia
# ============================================================

# Opciones:
# - "tfidf"
# - "semantic"
# - "bert"
MODEL_NAME = "semantic"

# ============================================================
# Salidas
# ============================================================

RUN_NAME = MODEL_NAME
OUTPUT_SUBMISSION = OUTPUTS_DIR / f"{RUN_NAME}_results.csv"
OUTPUT_METRICS = OUTPUTS_DIR / f"{RUN_NAME}_metrics.json"

# ============================================================
# Modelos entrenados
# ============================================================

BERT_MODEL_DIR = OUTPUTS_DIR / "bert_model"

# ============================================================
# Configuración BERT
# ============================================================

BERT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 4
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 2
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1
BERT_USE_AMP = True

# ============================================================
# Task 2: configuración multimodal
# ============================================================

USE_CLIP_FOR_TASK2 = True
TEXT_WEIGHT = 0.96
IMAGE_WEIGHT = 0.04

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ============================================================
# Columnas y tokens
# ============================================================

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENS_ALL = [f"t{i}" for i in range(1, 11)]
REQUIRED_COLUMNS = ["id", "article_body", "image_hash"] + TITLE_COLS

# ============================================================
# Evaluación
# ============================================================

NDCG_K = 10
ALPHA = 0.9
N_COLS = 10

# ============================================================
# General
# ============================================================

SEED = 42


def print_config() -> None:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("ACTIVE_DATASET:", ACTIVE_DATASET)
    print("MODEL_NAME:", MODEL_NAME)
    print("TRAIN_CSV:", TRAIN_CSV)
    print("TEST_CSV:", TEST_CSV)
    print("IMAGES_DIR:", IMAGES_DIR)
    print("OUTPUT_SUBMISSION:", OUTPUT_SUBMISSION)
    print("OUTPUT_METRICS:", OUTPUT_METRICS)