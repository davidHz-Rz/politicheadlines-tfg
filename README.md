# PoliticHeadlinES - TFG

Sistema automático para el ranking de titulares en noticias políticas.

## Estado actual
- Baseline TF-IDF implementado
- Integración multimodal con CLIP
- Evaluación con métrica PA-nDCG@K
- GPU (CUDA) habilitada

## Ejecución

```bash
python src/run_baseline.py


## Mejor sistema final actual

Task 1 (textual): 0.9526597750789877
Task 2 (multimodal): 0.95315642541915
Mean PA-nDCG@K: 0.9529081002490689

Modelo textual: dccuchile/bert-base-spanish-wwm-cased
Enfoque textual: cross-encoder entrenado
Enfoque multimodal: fusión tardía con CLIP
Pesos Task 2:
- texto = 0.96
- imagen = 0.04