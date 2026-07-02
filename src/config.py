"""
Global configuration for the PoliticHeadlinES project.

It covers paths, dataset selection, model selection, training and inference parameters,
ensemble weights, reranking settings, multimodal and auxiliary functions for the
executions as well. 
"""

from pathlib import Path

# ROOTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"


# ACTIVE DATASET

# The current version uses a data split made for further experimentation for
# the TFG. Original competition files followed similar structure and are 
# compatible.

ACTIVE_DATASET = "tfg_split" # Name of the folder
DATASET_DIR = DATA_DIR / ACTIVE_DATASET

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
TEST_CSV = DATASET_DIR / "test.csv"
IMAGES_DIR = DATASET_DIR / "images"


# EXECUTION SELECTION

# Main model used by src/run.py. Options:
# - "tfidf"
# - "semantic"
# - "bm25"
# - "bert"                               CHANGE
# - "bert_headtail"                      CHANGE
# - "bertin"
# - "mdeberta"
# - "crossencoder_ensemble"
# - "llm_ranker"
# - "modern_reranker"                    CHANGE
# - "tail_reranker"
# - "bert_rank10"                        CHANGE
MODEL_NAME = "bm25"

# Kept for compatibility with older scripts. CHANGE
ACTIVE_CROSS_ENCODER = MODEL_NAME

# Prefix used to name output files
RUN_NAME = MODEL_NAME # CAN BE REMOVED
OUTPUT_SUBMISSION = OUTPUTS_DIR / f"{RUN_NAME}_results.csv"
OUTPUT_METRICS = OUTPUTS_DIR / f"{RUN_NAME}_metrics.json"


# MAIN PARAMETERS FOR THE DATASET AND EVALUATION

TITLE_COLS = [f"title_{i}" for i in range(1, 11)]
TOKENS_ALL = [f"t{i}" for i in range(1, 11)]
REQUIRED_COLUMNS = ["id", "article_body", "image_hash"] + TITLE_COLS

NDCG_K = 10
ALPHA = 0.9
N_COLS = 10

SEED = 42
FORCE_CPU = False # Forces the execution to be done by the CPU instead of GPU


# FOLDERS FOR THE TRAINED MODELS

BERT_MODEL_DIR = OUTPUTS_DIR / "bert_model"                         # CHANGE
BERT_HEADTAIL_MODEL_DIR = OUTPUTS_DIR / "bert_headtail_model"       # CHANGE
BERTIN_MODEL_DIR = OUTPUTS_DIR / "bertin_model"
MDEBERTA_MODEL_DIR = OUTPUTS_DIR / "mdeberta_model"
BERT_RANK10_MODEL_DIR = OUTPUTS_DIR / "bert_rank10_model"           # CHANGE


# POINTWISE CROSSENCODER CONFIGURATIONS

# BINARY TRAINED MODELS

# beto_headtail takes part of the end of the article instead of truncating only
# from the beginning

CROSS_ENCODER_CONFIGS = {
    "bert": {                                                       # CHANGE
        "model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "model_dir": BERT_MODEL_DIR,
        "max_length": 512,
        "batch_size": 64,
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
    "bert_headtail": {                                                       # CHANGE
        "model_name": BERT_MODEL_DIR, # Initializaed from the previously fine-tuned BETO checkpoint.
        "model_dir": BERT_HEADTAIL_MODEL_DIR,
        "max_length": 512,
        "batch_size": 64,
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
        "head_tokens": 384, # Approximate token budget for the beginning of the article.
        "tail_tokens": 125,
    },
    "bertin": {
        "model_name": "bertin-project/bertin-roberta-base-spanish",
        "model_dir": BERTIN_MODEL_DIR,
        "max_length": 512,
        "batch_size": 64,
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


# BETO RANK10 CONFIGURATION

# It is trained to rank the whole list of candidates, not only the first one
# like the binary trained models

CROSS_ENCODER_RANK10_CONFIGS = {
    "bert_rank10": {
        "model_name": BERT_MODEL_DIR,
        "model_dir": BERT_RANK10_MODEL_DIR,
        "max_length": 512,
        "batch_size": 32,
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


# TEXTUAL ENSEMBLE WEIGHTS

# Each member produces one relevance score per candidate article. Then, they
# are combined linearly using the weights below. After sorting the list with
# those new values the new ranking is obtained

CROSS_ENCODER_ENSEMBLE_MEMBERS = [
    ("bert_headtail", 0.40),
    ("bert", 0.45),
    ("mdeberta", 0.15)
]


# LEXICAL BASELINES CONFIGURATION

BM25_K1 = 1.75
BM25_B = 0.5
# Limits the amount of terms used. Use None to use no limit
BM25_QUERY_TERM_LIMIT = 512


# MULTIMODAL SETTINGS

USE_VLM_FOR_TASK2 = False 
TEXT_WEIGHT = 0.90
IMAGE_WEIGHT = 0.10

# Supported models are:
# - clip
# - siglip
# Change SIGLIP_MODEL_NAME to use siglip2
VLM_BACKEND = "siglip"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"  # siglip-base-patch16-224, siglip2-base-patch16-224

# Compatibility with older scripts. REMOVE WHEN POSSIBLE
USE_CLIP_FOR_TASK2 = USE_VLM_FOR_TASK2


# LLMs CONFIGURATION

# Recommended models:
# - "Qwen/Qwen2.5-7B-Instruct"
# - "meta-llama/Llama-3.1-8B-Instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Modes:
# - "solo": evaluates the LLM as an individual text ranker.
# - "ensemble": combines the LLM with other textual model. Possible with previous ensembles
# - "rerank": uses the llm as a reranker over another model or ensemble
LLM_RANKER_MODE = "solo"

# Model selection for ensemble or rerank mode
# Supported: 
# - "crossencoder_ensemble"
# - "bert"
# - "bertin"
# - "mdeberta"
# - "bm25"
# - "semantic"
LLM_BASE_RANKER = "crossencoder_ensemble"
LLM_RERANK_TOP_K = 10
LLM_BASE_WEIGHT = 0.85
LLM_WEIGHT = 0.15

# Main LLM parameters
LLM_MAX_INPUT_CHARS = 3500
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.0
LLM_DO_SAMPLE = False
LLM_LOAD_IN_4BIT = True
LLM_TORCH_DTYPE = "auto"
LLM_TRUST_REMOTE_CODE = False


# MODERN RERANKER (BGE) SETTINGS

MODERN_RERANKER_MODEL_KEY = "bge_reranker_v2_m3"

MODERN_RERANKER_CONFIGS = {
    "bge_reranker_v2_m3": {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "max_length": 256,
        "batch_size": 32,
        "use_fp16": True,
    },
}

# Modess (same as LLMs, but can be used as a tail reranker too):             REVIEW and delete unnecessary modes (at least tail reranking)
# "solo", "ensemble", "rerank", "rerank_tail"
MODERN_RERANKER_MODE = "solo"

# Base ranker selection for ensemble/reranking mode
# Supported models: crossencoder_ensemble, bert, bertin, mdeberta, bm25, tfidf, semantic
MODERN_RERANKER_BASE_RANKER = "crossencoder_ensemble"


MODERN_RERANKER_TOP_K = 10

MODERN_RERANKER_BASE_WEIGHT = 0.90
MODERN_RERANKER_WEIGHT = 0.10


# TAIL RERANKING CONFIGUTARION

TAIL_RERANKER_BASE_RANKER = "crossencoder_ensemble"
TAIL_RERANKER_AUX_RANKER = "bert_rank10"  # bm25, tfidf, semantic, bge, bert_rank10
TAIL_RERANKER_TOP_K = 10


# BACKWARDS COMPATIBILITY (REMOVE WHEN POSSIBLE)

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


# HELPER FUNCTIONS

def get_cross_encoder_runtime_config(model_key: str) -> dict:
    """
    Return a copy of the runtime configuration for a pointwise cross-encoder.

    Parameters
    ----------
    model_key : str
        Identifier of the cross.encoder configuration.

    Raises
    ------
    ValueError
        If key is not registered.

    Returns
    -------
    dict
        Copy of the selected configuration dictionary.

    """
    if model_key not in CROSS_ENCODER_CONFIGS:
        raise ValueError(f"Cross-encoder model not supported: {model_key}")
    return dict(CROSS_ENCODER_CONFIGS[model_key])


def get_cross_encoder_rank10_runtime_config(model_key: str) -> dict:
    if model_key not in CROSS_ENCODER_RANK10_CONFIGS:
        raise ValueError(f"Cross-encoder rank10 model not supported: {model_key}")
    return dict(CROSS_ENCODER_RANK10_CONFIGS[model_key])


def get_cross_encoder_model_dir(model_key: str) -> Path:
    return Path(get_cross_encoder_runtime_config(model_key)["model_dir"])


def get_modern_reranker_runtime_config(model_key: str) -> dict:
    if model_key not in MODERN_RERANKER_CONFIGS:
        raise ValueError(f"Reranker moderno no soportado: {model_key}")
    return dict(MODERN_RERANKER_CONFIGS[model_key])


def get_vlm_model_name() -> str:
    """
    Resolve the Huggin Face model name associated with the selected VLM backend.

    Raises
    ------
    ValueError
        If VLM_BACKEND is not one of the supported backends.

    Returns
    -------
    str
        Model identifier for the selected backend

    """
    backend = VLM_BACKEND.lower().strip()
    if backend == "clip":
        return CLIP_MODEL_NAME
    if backend == "siglip":
        return SIGLIP_MODEL_NAME
    raise ValueError(f"VLM_BACKEND not supported: {VLM_BACKEND}")



def print_config() -> None:
    """
    Print the most relevant active configuration values.
    
    Useful for debugging and verifying the used dataset, model and output paths
    before running any experiment.
    """
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
