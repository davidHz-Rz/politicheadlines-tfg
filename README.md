# PoliticHeadlinES TFG

Sistema experimental para la tarea de **selección y ranking automático de titulares en noticias políticas en español**, desarrollado en el contexto del Trabajo Fin de Grado y orientado a la competición **PoliticHeadlinES**.

## Objetivo

El objetivo del proyecto es construir y evaluar modelos capaces de ordenar automáticamente un conjunto de diez titulares candidatos según su adecuación a una noticia política en español.

Se abordan dos subtareas:

- **Task 1 (textual):** ranking de titulares utilizando únicamente el contenido textual del artículo.
- **Task 2 (multimodal):** ranking de titulares combinando texto e imagen asociada a la noticia.

---

## Enfoque desarrollado

A lo largo del proyecto se implementaron varios enfoques progresivamente más potentes:

1. **Baseline TF-IDF**
   - Sistema de referencia basado en similitud léxica.

2. **Modelo semántico**
   - Ranking textual basado en representaciones semánticas.

3. **Cross-Encoder textual (modelo final para Task 1)**
   - Fine-tuning de un modelo BERT para español sobre pares *(artículo, titular)*.
   - Modelo utilizado:
     - `dccuchile/bert-base-spanish-wwm-cased`

4. **Fusión multimodal con CLIP (modelo final para Task 2)**
   - Combinación del score textual del cross-encoder con una señal visual obtenida mediante:
     - `openai/clip-vit-base-patch32`
   - Fusión tardía con pesos:
     - **texto = 0.96**
     - **imagen = 0.04**

---

## Resultados finales

### Mejor configuración obtenida

- **Task 1 (textual):** `0.9526597750789877`
- **Task 2 (multimodal):** `0.95315642541915`
- **Mean PA-nDCG@K:** `0.9529081002490689`

Modelo textual: dccuchile/bert-base-spanish-wwm-cased
Enfoque textual: cross-encoder entrenado
Enfoque multimodal: fusión tardía con CLIP
Pesos Task 2:
- texto = 0.96
- imagen = 0.04

### Métrica de evaluación
Se emplea la métrica oficial de la competición:

- **PA-nDCG@K**