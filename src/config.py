from pathlib import Path

# ============================================================
# 1. Rutas base
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"


# ============================================================
# 2. Dataset activo
# ============================================================

# Opciones antiguas:
# - "development_phase_initial"
# - "train_corpora"
# - "test_public"
#
# Opcion actual (dataset particionado):
# - "tfg_split"
# Nota: la organización de estos CSV se revisará en la fase de particionado.
ACTIVE_DATASET = "tfg_split"
DATASET_DIR = DATA_DIR / ACTIVE_DATASET

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
TEST_CSV = DATASET_DIR / "test.csv"
IMAGES_DIR = DATASET_DIR / "images"


# ============================================================
# 3. Selección de ejecución
# ============================================================

# Opciones:
# - "tfidf"
# - "semantic"
# - "bm25"
# - "bert"
# - "bert_headtail"
# - "bertin"
# - "mdeberta"
# - "crossencoder_ensemble"
# - "llm_ranker"
# - "modern_reranker"
# - "tail_reranker"
# - "bert_rank10"
MODEL_NAME = "bert_headtail"
ACTIVE_CROSS_ENCODER = MODEL_NAME

RUN_NAME = MODEL_NAME
OUTPUT_SUBMISSION = OUTPUTS_DIR / f"{RUN_NAME}_results.csv"
OUTPUT_METRICS = OUTPUTS_DIR / f"{RUN_NAME}_metrics.json"


# ============================================================
# 4. Columnas, evaluación y parámetros generales
# ============================================================

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENS_ALL = [f"t{i}" for i in range(1, 11)]
REQUIRED_COLUMNS = ["id", "article_body", "image_hash"] + TITLE_COLS

NDCG_K = 10
ALPHA = 0.9
N_COLS = 10

SEED = 42
FORCE_CPU = False


# ============================================================
# 5. Directorios de modelos entrenados
# ============================================================

BERT_MODEL_DIR = OUTPUTS_DIR / "bert_model"
BERT_HEADTAIL_MODEL_DIR = OUTPUTS_DIR / "bert_headtail_model"
BERTIN_MODEL_DIR = OUTPUTS_DIR / "bertin_model"
MDEBERTA_MODEL_DIR = OUTPUTS_DIR / "mdeberta_model"
BERT_RANK10_MODEL_DIR = OUTPUTS_DIR / "bert_rank10_model"


# ============================================================
# 6. Cross-encoders pointwise
# ============================================================

CROSS_ENCODER_CONFIGS = {
    "bert": {
        "model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "model_dir": BERT_MODEL_DIR,
        "max_length": 512,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "epochs": 10,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_monitor": "task_1_pa_ndcg",
    },
    "bert_headtail": {
        "model_name": BERT_MODEL_DIR,
        "model_dir": BERT_HEADTAIL_MODEL_DIR,
        "max_length": 512,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "epochs": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_monitor": "task_1_pa_ndcg",
        "use_head_tail": True,
        "head_tokens": 384,
        "tail_tokens": 125,
    },
    "bertin": {
        "model_name": "bertin-project/bertin-roberta-base-spanish",
        "model_dir": BERTIN_MODEL_DIR,
        "max_length": 512,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "epochs": 10,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_monitor": "task_1_pa_ndcg",
    },
    "mdeberta": {
        "model_name": "microsoft/mdeberta-v3-base",
        "model_dir": MDEBERTA_MODEL_DIR,
        "max_length": 512,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "epochs": 10,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_monitor": "task_1_pa_ndcg",
    },
}


# ============================================================
# 7. Cross-encoder para ranking completo (rank10)
# ============================================================

# Inicializa desde el checkpoint BETO ya entrenado si existe.
# Si no existe, cambia model_name a "dccuchile/bert-base-spanish-wwm-cased".
CROSS_ENCODER_RANK10_CONFIGS = {
    "bert_rank10": {
        "model_name": BERT_MODEL_DIR,
        "model_dir": BERT_RANK10_MODEL_DIR,
        "max_length": 512,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
        "epochs": 10,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "use_amp": True,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_monitor": "task_1_pa_ndcg",
    },
}


# ============================================================
# 8. Ensemble textual
# ============================================================

CROSS_ENCODER_ENSEMBLE_MEMBERS = [
    ("bert", 0.70),
    ("bertin", 0.30),
]


# ============================================================
# 9. Baselines léxicos
# ============================================================

BM25_K1 = 1.75
BM25_B = 0.5
# Limita los términos de query del artículo a los más informativos.
# Usa None para emplear todos los términos del artículo.
BM25_QUERY_TERM_LIMIT = 512


# ============================================================
# 10. Task 2: configuración multimodal
# ============================================================

USE_VLM_FOR_TASK2 = True
TEXT_WEIGHT = 0.90
IMAGE_WEIGHT = 0.10

# Backends soportados:
# - "clip"
# - "siglip"
VLM_BACKEND = "siglip"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"  # siglip-base-patch16-224, siglip2-base-patch16-224, siglip-base-patch16-384

# Compatibilidad con el código anterior
USE_CLIP_FOR_TASK2 = USE_VLM_FOR_TASK2


# ============================================================
# 11. Configuración LLM ranker
# ============================================================

# Para usarlo, poner MODEL_NAME = "llm_ranker".
# Modelos recomendados:
# - "Qwen/Qwen2.5-7B-Instruct"
# - "meta-llama/Llama-3.1-8B-Instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Modos:
# - "solo": usa únicamente el LLM como ranker textual.
# - "ensemble": combina score base + score LLM.
# - "rerank": genera ranking base y reordena el top-k con LLM.
LLM_RANKER_MODE = "solo"

# Ranker base para "ensemble" o "rerank".
# Soportado: "crossencoder_ensemble", "bert", "bertin", "mdeberta", "bm25", "semantic".
LLM_BASE_RANKER = "crossencoder_ensemble"
LLM_RERANK_TOP_K = 10
LLM_BASE_WEIGHT = 0.85
LLM_WEIGHT = 0.15

# Inferencia LLM.
LLM_MAX_INPUT_CHARS = 3500
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.0
LLM_DO_SAMPLE = False
LLM_LOAD_IN_4BIT = True
LLM_TORCH_DTYPE = "auto"
LLM_TRUST_REMOTE_CODE = False


# ============================================================
# 12. Configuración Modern Reranker (BGE)
# ============================================================

MODERN_RERANKER_MODEL_KEY = "bge_reranker_v2_m3"

MODERN_RERANKER_CONFIGS = {
    "bge_reranker_v2_m3": {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "max_length": 256,
        "batch_size": 8,
        "use_fp16": True,
    },
}

# Modos:
# "solo"        -> BGE puntúa los 10 titulares
# "ensemble"    -> combina ranker base + BGE
# "rerank"      -> reordena top-k del ranker base
# "rerank_tail" -> mantiene top1 y reordena 2..k
MODERN_RERANKER_MODE = "solo"

# Ranker base para ensemble / rerank
# Opciones: crossencoder_ensemble, bert, bertin, mdeberta, bm25, tfidf, semantic
MODERN_RERANKER_BASE_RANKER = "crossencoder_ensemble"

# Para rerank / rerank_tail
MODERN_RERANKER_TOP_K = 10

# Para ensemble
MODERN_RERANKER_BASE_WEIGHT = 0.90
MODERN_RERANKER_WEIGHT = 0.10


# ============================================================
# 13. Tail reranker
# ============================================================

TAIL_RERANKER_BASE_RANKER = "crossencoder_ensemble"
TAIL_RERANKER_AUX_RANKER = "bert_rank10"  # bm25, tfidf, semantic, bge, bert_rank10
TAIL_RERANKER_TOP_K = 10


# ============================================================
# 14. Compatibilidad con código existente
# ============================================================

BERT_MODEL_NAME = CROSS_ENCODER_CONFIGS["bert"]["model_name"]
BERT_MAX_LENGTH = CROSS_ENCODER_CONFIGS["bert"]["max_length"]
BERT_BATCH_SIZE = CROSS_ENCODER_CONFIGS["bert"]["batch_size"]
BERT_GRADIENT_ACCUMULATION_STEPS = CROSS_ENCODER_CONFIGS["bert"]["gradient_accumulation_steps"]
BERT_LEARNING_RATE = CROSS_ENCODER_CONFIGS["bert"]["learning_rate"]
BERT_EPOCHS = CROSS_ENCODER_CONFIGS["bert"]["epochs"]
BERT_WEIGHT_DECAY = CROSS_ENCODER_CONFIGS["bert"]["weight_decay"]
BERT_WARMUP_RATIO = CROSS_ENCODER_CONFIGS["bert"]["warmup_ratio"]
BERT_USE_AMP = CROSS_ENCODER_CONFIGS["bert"]["use_amp"]

BERTIN_MODEL_NAME = CROSS_ENCODER_CONFIGS["bertin"]["model_name"]
BERTIN_MAX_LENGTH = CROSS_ENCODER_CONFIGS["bertin"]["max_length"]
BERTIN_BATCH_SIZE = CROSS_ENCODER_CONFIGS["bertin"]["batch_size"]
BERTIN_GRADIENT_ACCUMULATION_STEPS = CROSS_ENCODER_CONFIGS["bertin"]["gradient_accumulation_steps"]
BERTIN_LEARNING_RATE = CROSS_ENCODER_CONFIGS["bertin"]["learning_rate"]
BERTIN_EPOCHS = CROSS_ENCODER_CONFIGS["bertin"]["epochs"]
BERTIN_WEIGHT_DECAY = CROSS_ENCODER_CONFIGS["bertin"]["weight_decay"]
BERTIN_WARMUP_RATIO = CROSS_ENCODER_CONFIGS["bertin"]["warmup_ratio"]
BERTIN_USE_AMP = CROSS_ENCODER_CONFIGS["bertin"]["use_amp"]

MDEBERTA_MODEL_NAME = CROSS_ENCODER_CONFIGS["mdeberta"]["model_name"]
MDEBERTA_MAX_LENGTH = CROSS_ENCODER_CONFIGS["mdeberta"]["max_length"]
MDEBERTA_BATCH_SIZE = CROSS_ENCODER_CONFIGS["mdeberta"]["batch_size"]
MDEBERTA_GRADIENT_ACCUMULATION_STEPS = CROSS_ENCODER_CONFIGS["mdeberta"]["gradient_accumulation_steps"]
MDEBERTA_LEARNING_RATE = CROSS_ENCODER_CONFIGS["mdeberta"]["learning_rate"]
MDEBERTA_EPOCHS = CROSS_ENCODER_CONFIGS["mdeberta"]["epochs"]
MDEBERTA_WEIGHT_DECAY = CROSS_ENCODER_CONFIGS["mdeberta"]["weight_decay"]
MDEBERTA_WARMUP_RATIO = CROSS_ENCODER_CONFIGS["mdeberta"]["warmup_ratio"]
MDEBERTA_USE_AMP = CROSS_ENCODER_CONFIGS["mdeberta"]["use_amp"]


# ============================================================
# 15. Helpers de configuración
# ============================================================

def get_cross_encoder_runtime_config(model_key: str) -> dict:
    if model_key not in CROSS_ENCODER_CONFIGS:
        raise ValueError(f"Modelo cross-encoder no soportado: {model_key}")
    return dict(CROSS_ENCODER_CONFIGS[model_key])


def get_cross_encoder_rank10_runtime_config(model_key: str) -> dict:
    if model_key not in CROSS_ENCODER_RANK10_CONFIGS:
        raise ValueError(f"Modelo cross-encoder rank10 no soportado: {model_key}")
    return dict(CROSS_ENCODER_RANK10_CONFIGS[model_key])


def get_cross_encoder_model_dir(model_key: str) -> Path:
    return Path(get_cross_encoder_runtime_config(model_key)["model_dir"])


def get_vlm_model_name() -> str:
    backend = VLM_BACKEND.lower().strip()
    if backend == "clip":
        return CLIP_MODEL_NAME
    if backend == "siglip":
        return SIGLIP_MODEL_NAME
    raise ValueError(f"VLM_BACKEND no soportado: {VLM_BACKEND}")


def get_modern_reranker_runtime_config(model_key: str) -> dict:
    if model_key not in MODERN_RERANKER_CONFIGS:
        raise ValueError(f"Reranker moderno no soportado: {model_key}")
    return dict(MODERN_RERANKER_CONFIGS[model_key])


def print_config() -> None:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("ACTIVE_DATASET:", ACTIVE_DATASET)
    print("MODEL_NAME:", MODEL_NAME)
    print("ACTIVE_CROSS_ENCODER:", ACTIVE_CROSS_ENCODER)
    print("TRAIN_CSV:", TRAIN_CSV)
    print("VAL_CSV:", VAL_CSV)
    print("TEST_CSV:", TEST_CSV)
    print("IMAGES_DIR:", IMAGES_DIR)
    print("OUTPUT_SUBMISSION:", OUTPUT_SUBMISSION)
    print("OUTPUT_METRICS:", OUTPUT_METRICS)
