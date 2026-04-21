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
ACTIVE_DATASET = "development_phase_initial"

DATASET_DIR = DATA_DIR / ACTIVE_DATASET

# ============================================================
# Archivos principales
# ============================================================

TRAIN_CSV = DATASET_DIR / "train_public.csv"
TEST_CSV = DATASET_DIR / "dev_public.csv"   # dev_public.csv (metrics) / test_public.csv (submission)
IMAGES_DIR = DATASET_DIR / "images"

# ============================================================
# Selección de modelo de inferencia / entrenamiento
# ============================================================

# Opciones:
# - "tfidf"
# - "semantic"
# - "bert"
# - "bertin"
# - "mdeberta"
MODEL_NAME = "mdeberta"
ACTIVE_CROSS_ENCODER = MODEL_NAME

# ============================================================
# Salidas
# ============================================================

RUN_NAME = MODEL_NAME
OUTPUT_SUBMISSION = OUTPUTS_DIR / f"{RUN_NAME}_results.csv"
OUTPUT_METRICS = OUTPUTS_DIR / f"{RUN_NAME}_metrics.json"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"

# ============================================================
# Modelos entrenados
# ============================================================

BERT_MODEL_DIR = OUTPUTS_DIR / "bert_model"
BERTIN_MODEL_DIR = OUTPUTS_DIR / "bertin_model"
MDEBERTA_MODEL_DIR = OUTPUTS_DIR / "mdeberta_model"

# ============================================================
# Configuración de cross-encoders
# ============================================================

CROSS_ENCODER_CONFIGS = {
    "bert": {
        "model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "model_dir": BERT_MODEL_DIR,
        "max_length": 512,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "epochs": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
    },
    "bertin": {
        "model_name": "bertin-project/bertin-roberta-base-spanish",
        "model_dir": BERTIN_MODEL_DIR,
        "max_length": 512,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "epochs": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
    },
    "mdeberta": {
        "model_name": "microsoft/mdeberta-v3-base",
        "model_dir": MDEBERTA_MODEL_DIR,
        "max_length": 512,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "epochs": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
    },
}

# Compatibilidad con el código existente
BERT_MODEL_NAME = CROSS_ENCODER_CONFIGS["bert"]["model_name"]
BERT_MAX_LENGTH = CROSS_ENCODER_CONFIGS["bert"]["max_length"]
BERT_BATCH_SIZE = CROSS_ENCODER_CONFIGS["bert"]["batch_size"]
BERT_LEARNING_RATE = CROSS_ENCODER_CONFIGS["bert"]["learning_rate"]
BERT_EPOCHS = CROSS_ENCODER_CONFIGS["bert"]["epochs"]
BERT_WEIGHT_DECAY = CROSS_ENCODER_CONFIGS["bert"]["weight_decay"]
BERT_WARMUP_RATIO = CROSS_ENCODER_CONFIGS["bert"]["warmup_ratio"]
BERT_USE_AMP = CROSS_ENCODER_CONFIGS["bert"]["use_amp"]

BERTIN_MODEL_NAME = CROSS_ENCODER_CONFIGS["bertin"]["model_name"]
BERTIN_MAX_LENGTH = CROSS_ENCODER_CONFIGS["bertin"]["max_length"]
BERTIN_BATCH_SIZE = CROSS_ENCODER_CONFIGS["bertin"]["batch_size"]
BERTIN_LEARNING_RATE = CROSS_ENCODER_CONFIGS["bertin"]["learning_rate"]
BERTIN_EPOCHS = CROSS_ENCODER_CONFIGS["bertin"]["epochs"]
BERTIN_WEIGHT_DECAY = CROSS_ENCODER_CONFIGS["bertin"]["weight_decay"]
BERTIN_WARMUP_RATIO = CROSS_ENCODER_CONFIGS["bertin"]["warmup_ratio"]
BERTIN_USE_AMP = CROSS_ENCODER_CONFIGS["bertin"]["use_amp"]

MDEBERTA_MODEL_NAME = CROSS_ENCODER_CONFIGS["mdeberta"]["model_name"]
MDEBERTA_MAX_LENGTH = CROSS_ENCODER_CONFIGS["mdeberta"]["max_length"]
MDEBERTA_BATCH_SIZE = CROSS_ENCODER_CONFIGS["mdeberta"]["batch_size"]
MDEBERTA_LEARNING_RATE = CROSS_ENCODER_CONFIGS["mdeberta"]["learning_rate"]
MDEBERTA_EPOCHS = CROSS_ENCODER_CONFIGS["mdeberta"]["epochs"]
MDEBERTA_WEIGHT_DECAY = CROSS_ENCODER_CONFIGS["mdeberta"]["weight_decay"]
MDEBERTA_WARMUP_RATIO = CROSS_ENCODER_CONFIGS["mdeberta"]["warmup_ratio"]
MDEBERTA_USE_AMP = CROSS_ENCODER_CONFIGS["mdeberta"]["use_amp"]

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
FORCE_CPU = False


def get_cross_encoder_runtime_config(model_key: str) -> dict:
    if model_key not in CROSS_ENCODER_CONFIGS:
        raise ValueError(f"Modelo cross-encoder no soportado: {model_key}")
    return dict(CROSS_ENCODER_CONFIGS[model_key])


def get_cross_encoder_model_dir(model_key: str) -> Path:
    return Path(get_cross_encoder_runtime_config(model_key)["model_dir"])


def print_config() -> None:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("ACTIVE_DATASET:", ACTIVE_DATASET)
    print("MODEL_NAME:", MODEL_NAME)
    print("ACTIVE_CROSS_ENCODER:", ACTIVE_CROSS_ENCODER)
    print("TRAIN_CSV:", TRAIN_CSV)
    print("TEST_CSV:", TEST_CSV)
    print("IMAGES_DIR:", IMAGES_DIR)
    print("OUTPUT_SUBMISSION:", OUTPUT_SUBMISSION)
    print("OUTPUT_METRICS:", OUTPUT_METRICS)
