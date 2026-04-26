\# PoliticHeadlinES TFG



Sistema experimental para la tarea de \*\*selección y ranking automático de titulares en noticias políticas en español\*\*, desarrollado como \*\*Trabajo Fin de Grado (TFG)\*\* y orientado a la competición \*\*PoliticHeadlinES\*\*. :contentReference\[oaicite:0]{index=0}



\---



\## Objetivo



Construir y evaluar modelos capaces de \*\*ordenar automáticamente diez titulares candidatos\*\* según su adecuación a una noticia política en español.



Se trabajan dos subtareas:



\- \*\*Task 1 (textual):\*\* ranking usando únicamente el contenido textual del artículo.

\- \*\*Task 2 (multimodal):\*\* ranking combinando texto e imagen asociada.



\---



\## Enfoques implementados



A lo largo del proyecto se desarrollaron y compararon distintos enfoques de ranking.



\### 1. Métodos clásicos de recuperación



\- TF-IDF

\- BM25



Útiles como baseline rápido y referencia frente a modelos neuronales.



\### 2. Rankers semánticos densos



\- Semantic Ranker basado en embeddings.



\### 3. Cross-Encoders Transformer



Modelos fine-tuned para puntuar pares:



```text

(article\_body, title)

````



Modelos principales utilizados:



\* `dccuchile/bert-base-spanish-wwm-cased` (\*\*BETO\*\*)

\* `bertin-project/bertin-roberta-base-spanish` (\*\*BERTIN\*\*)

\* `microsoft/mdeberta-v3-base` (\*\*mDeBERTa\*\*, experimental)



\### 4. Sistemas multimodales



Fusión tardía entre scores textuales y modelos visión-lenguaje:



\* CLIP

\* SigLIP / SigLIP2



\### 5. Ensemble de Cross-Encoders



Combinación ponderada de modelos fuertes para mejorar robustez y generalización.



\### 6. Large Language Models (LLMs)



Se integraron modelos instruction-tuned como rankers zero-shot y como señal auxiliar:



\* `Qwen/Qwen2.5-7B-Instruct`

\* `meta-llama/Llama-3.1-8B-Instruct`



Modos evaluados:



\* \*\*solo\*\*: LLM como ranker textual principal

\* \*\*rerank\*\*: reordenación del top-k de un baseline fuerte

\* \*\*ensemble\*\*: combinación ponderada con modelos supervisados



\---



\## Mejor resultado en leaderboard



El mejor sistema enviado hasta la fecha fue un \*\*ensemble textual BETO + BERTIN\*\* con señal visual ligera para Task 2.



\### Resultado oficial



```text

metric\_1\_3 = 0.8249201498036288

metric\_2\_3 = 0.8244442729634858

mean\_ndcg  = 0.8246822113835572

```



\### Configuración ganadora



```text

BETO   = 0.70

BERTIN = 0.30

```



Para Task 2:



```text

texto  = 0.96

imagen = 0.04

```



\---



\## Resultados destacados en validación interna



\### Ensemble textual



```text

BETO + BERTIN ≈ 0.974

```



\### LLMs (zero-shot)



```text

Qwen 7B solo   ≈ 0.696

Llama 8B solo  ≈ 0.555

```



\### LLMs como ensemble auxiliar



```text

Qwen 7B + ensemble base (0.97 / 0.03)  ≈ 0.97464

Llama 8B + ensemble base (0.97 / 0.03) ≈ 0.97413

```



\---



\## Hallazgos relevantes



\* Los \*\*cross-encoders supervisados\*\* superaron claramente a métodos clásicos como TF-IDF y BM25.

\* El \*\*ensemble BETO + BERTIN\*\* fue el sistema más sólido y competitivo.

\* Una pequeña señal visual mejoró ligeramente Task 2.

\* Los \*\*LLMs zero-shot\*\* no superaron a los modelos entrenados específicos.

\* Sin embargo, los LLMs sí aportaron valor como \*\*señal complementaria de bajo peso\*\* dentro de ensembles.

\* `Qwen2.5-7B-Instruct` rindió mejor que `Llama 3.1 8B Instruct` en esta tarea.



\---



\## Métrica de evaluación



Se emplea la métrica oficial de la competición:



\* \*\*PA-nDCG@10\*\*

\* \*\*mean\_ndcg\*\*



\---



\## Estructura del proyecto



```text

politicheadlines/

├── src/

│   ├── config.py

│   ├── run.py

│   ├── models/

│   │   ├── bm25\_ranker.py

│   │   ├── cross\_encoder\_ranker.py

│   │   ├── cross\_encoder\_ensemble\_ranker.py

│   │   ├── llm\_ranker.py

│   │   ├── semantic\_ranker.py

│   │   ├── tfidf\_ranker.py

│   │   └── vlm\_ranker.py

│   ├── training/

│   └── utils/

├── requirements.txt

└── README.md

```



\---



\## Uso



\### Entrenamiento



```bash

python src/training/train\_bert.py

python src/training/train\_bertin.py

python src/training/train\_mdeberta.py

```



\### Inferencia / evaluación



```bash

python src/run.py

```



\### Selección de modelo



Desde `src/config.py`:



```python

MODEL\_NAME = "crossencoder\_ensemble"

MODEL\_NAME = "llm\_ranker"

MODEL\_NAME = "bm25"

MODEL\_NAME = "semantic"

```



\---



\## Tecnologías utilizadas



\* Python

\* PyTorch

\* Hugging Face Transformers

\* Scikit-learn

\* Pandas

\* NumPy



\---



\## Trabajo futuro



\* Integración de rerankers modernos especializados.

\* Fine-tuning específico de LLMs sobre titulares políticos.

\* Gating dinámico entre ensemble y LLM según incertidumbre.

\* Técnicas de explicabilidad de rankings.

\* Optimización de inferencia multimodal a gran escala.



```

```



