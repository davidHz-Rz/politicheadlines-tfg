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
ACTIVE_DATASET = "development_phase_initial"

DATASET_DIR = DATA_DIR / ACTIVE_DATASET
INPUTS_DIR = DATA_DIR / "input_data"

# ============================================================
# Archivos principales
# ============================================================

TRAIN_CSV = DATASET_DIR / "train_public.csv"
DEV_CSV = DATASET_DIR / "dev_public.csv"
IMAGES_DIR = DATASET_DIR / "images"

OUTPUT_SUBMISSION = OUTPUTS_DIR / "results.csv"
OUTPUT_METRICS = OUTPUTS_DIR / "results.metrics.json"

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
# Configuración general
# ============================================================

SEED = 42
METHOD = "tfidf"

# ============================================================
# Task 2: pesos de fusión texto + imagen
# ============================================================

TEXT_WEIGHT = 0.85
IMAGE_WEIGHT = 0.15

# ============================================================
# Modelos
# ============================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ============================================================
# Debug
# ============================================================

def print_config() -> None:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("ACTIVE_DATASET:", ACTIVE_DATASET)
    print("DATASET_DIR:", DATASET_DIR)
    print("TRAIN_CSV:", TRAIN_CSV)
    print("DEV_CSV:", DEV_CSV)
    print("IMAGES_DIR:", IMAGES_DIR)
    print("OUTPUT_SUBMISSION:", OUTPUT_SUBMISSION)