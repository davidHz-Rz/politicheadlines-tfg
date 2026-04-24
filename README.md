# PoliticHeadlinES TFG

Sistema experimental para la tarea de **selección y ranking automático de titulares en noticias políticas en español**, desarrollado como **Trabajo Fin de Grado (TFG)** y orientado a la competición **PoliticHeadlinES**.

---

## Objetivo

Construir y evaluar modelos capaces de **ordenar automáticamente diez titulares candidatos** según su adecuación a una noticia política en español.

Se trabajan dos subtareas:

- **Task 1 (textual):** ranking usando únicamente el contenido textual del artículo.
- **Task 2 (multimodal):** ranking combinando texto e imagen asociada.

---

## Enfoques implementados

A lo largo del proyecto se implementaron y compararon varios sistemas:

### 1. Métodos clásicos de recuperación

- TF-IDF
- BM25

Útiles como baseline y para análisis comparativo frente a modelos neuronales.

### 2. Rankers semánticos densos

- Semantic Ranker basado en embeddings.

### 3. Cross-Encoders Transformer

Modelos fine-tuned para puntuar pares:

```text
(article_body, title)
```

Modelos principales utilizados:

- `dccuchile/bert-base-spanish-wwm-cased` (**BETO**)
- **BERTIN**
- mDeBERTa (experimental)

### 4. Sistemas multimodales

Fusión tardía de scores textuales con modelos visión-lenguaje:

- CLIP
- SigLIP

### 5. Ensemble de Cross-Encoders

Combinación ponderada de modelos textuales para mejorar robustez y generalización.

---

## Mejor resultado en leaderboard

El mejor sistema enviado hasta la fecha fue un **ensemble textual BETO + BERTIN** con señal visual ligera para Task 2.

### Resultado oficial

```text
metric_1_3 = 0.8249201498036288
metric_2_3 = 0.8244442729634858
mean_ndcg = 0.8246822113835572
```

### Configuración ganadora

```text
BETO   = 0.70
BERTIN = 0.30
```

Para Task 2:

```text
texto = 0.96
imagen = 0.04
```

---

## Hallazgos relevantes

- Los **cross-encoders** superaron claramente a métodos clásicos como BM25 y TF-IDF en leaderboard real.
- Los métodos clásicos fueron útiles como referencia y análisis, pero no como sistema final.
- El **ensemble de modelos fuertes** generalizó mejor que los modelos individuales.
- Una pequeña señal visual ayudó en Task 2 sin dominar la decisión.

---

## Métrica de evaluación

Se emplea la métrica oficial de la competición:

- **PA-nDCG@K / mean_ndcg**

---

## Estructura del proyecto

```text
politicheadlines/
├── src/
│   ├── config.py
│   ├── run.py
│   ├── models/
│   │   ├── bm25_ranker.py
│   │   ├── cross_encoder_ranker.py
│   │   ├── cross_encoder_ensemble_ranker.py
│   │   ├── semantic_ranker.py
│   │   ├── tfidf_ranker.py
│   │   └── vlm_ranker.py
│   ├── training/
│   └── utils/
├── requirements.txt
└── README.md
```

---

## Uso

### Entrenamiento

```bash
python src/training/train_bert.py
python src/training/train_bertin.py
```

### Inferencia / evaluación

```bash
python src/run.py
```

---

## Tecnologías utilizadas

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy

---

## Trabajo futuro inmediato

- Integración de **LLMs como rankers o rerankers**
- Comparación frente al ensemble actual
- Estudio de explicabilidad de rankings
