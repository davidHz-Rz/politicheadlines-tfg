# PoliticHeadlinES TFG

Sistema experimental para la tarea de **selección y ranking automático de titulares en noticias políticas en español**, desarrollado en el contexto del **Trabajo Fin de Grado (TFG)** y orientado a la competición **PoliticHeadlinES**.

---

# Objetivo

El objetivo del proyecto es construir y evaluar modelos capaces de **ordenar automáticamente un conjunto de diez titulares candidatos** según su adecuación a una noticia política en español.

Se abordan dos subtareas:

* **Task 1 (textual):** ranking de titulares utilizando únicamente el contenido textual del artículo.
* **Task 2 (multimodal):** ranking de titulares combinando texto e imagen asociada a la noticia.

---

# Enfoque desarrollado

A lo largo del proyecto se implementaron varios enfoques progresivamente más potentes:

## 1. Baseline TF-IDF

* Sistema de referencia basado en similitud léxica.
* Integración multimodal inicial mediante CLIP.

### Resultados aproximados

* **Mean PA-nDCG@K:** `0.8159`

---

## 2. Modelo semántico

* Ranking textual basado en representaciones semánticas profundas.
* Mejora significativa sobre el baseline léxico.

### Resultados obtenidos

* **Task 1:** `0.8758677822040812`
* **Task 2:** `0.8158745760583925`
* **Mean:** `0.8458711791312369`

---

## 3. Cross-Encoder textual (modelo principal)

* Fine-tuning de un modelo BERT para español sobre pares:

```text
(article_body, title)
```

### Modelo utilizado

```text
dccuchile/bert-base-spanish-wwm-cased
```

### Entrenamiento inicial (dataset reducido)

* **Mean PA-nDCG@K:** `0.9349178978633903`

### Entrenamiento ampliado (train_corpora)

* **Task 1:** `0.9526597750789877`

---

## 4. Fusión multimodal con CLIP (modelo final para Task 2)

Combinación del score textual del cross-encoder con una señal visual obtenida mediante:

```text
openai/clip-vit-base-patch32
```

### Estrategia

* Fusión tardía de scores normalizados.
* Ajuste manual de pesos.

### Mejor configuración encontrada

```text
texto = 0.96
imagen = 0.04
```

---

# Resultados finales

## Mejor sistema global obtenido

* **Task 1 (textual):** `0.9526597750789877`
* **Task 2 (multimodal):** `0.95315642541915`
* **Mean PA-nDCG@K:** `0.9529081002490689`

### Resumen técnico

* Modelo textual: `dccuchile/bert-base-spanish-wwm-cased`
* Enfoque textual: Cross-Encoder entrenado
* Enfoque multimodal: Fusión tardía con CLIP
* Pesos finales:

  * texto = `0.96`
  * imagen = `0.04`

---

# Métrica de evaluación

Se emplea la métrica oficial de la competición:

* **PA-nDCG@K**

---

# Estructura actual del proyecto

```text
politicheadlines/
│
├── data/                 # datasets (ignorado por git)
├── outputs/              # modelos, métricas y resultados
│   └── bert_model/
│
├── src/
│   ├── config.py         # configuración principal
│   ├── run.py            # inferencia unificada
│
│   ├── models/
│   │   ├── bert_ranker.py
│   │   ├── clip_ranker.py
│   │   ├── semantic_ranker.py
│   │   └── tfidf_ranker.py
│
│   ├── training/
│   │   └── train_bert.py
│
│   └── utils/
│       ├── data_utils.py
│       ├── metrics.py
│       └── submission.py
│
├── requirements.txt
└── README.md
```

---

# Configuración principal

Todo se controla desde:

```text
src/config.py
```

## Selección de modelo

```python
MODEL_NAME = "bert"
```

Opciones disponibles:

```text
tfidf
semantic
bert
```

## Selección de dataset

```python
ACTIVE_DATASET = "train_corpora"
```

---

# Uso

## Entrenamiento del modelo BERT

```bash
python src/training/train_bert.py
```

## Inferencia / generación de resultados

```bash
python src/run.py
```

---

# Tecnologías utilizadas

* Python
* PyTorch
* Transformers (Hugging Face)
* Scikit-learn
* CLIP
* Pandas
* NumPy

---

# Hardware principal utilizado

```text
NVIDIA GeForce RTX 4050 Laptop GPU
CUDA 12.6
PyTorch 2.x
```

---

# Trabajo futuro (segunda fase)

Durante la segunda fase del proyecto se entrenarán y compararán nuevos modelos:

* BETO optimizado
* XLM-R
* Nuevos rerankers
* Modelos LLM
* Ensambles multimodales

---

# Autor

**David Hernández Ruiz**
Trabajo Fin de Grado
