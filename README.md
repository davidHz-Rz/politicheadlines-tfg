\# PoliticHeadlinES-IberLEF 2026 – I2C-UHU-Aries



This repository contains the code developed for my Bachelor's Degree Final Project and for the participation of the I2C-UHU-Aries team in the PoliticHeadlinES-IberLEF 2026 shared task.



The task focuses on ranking candidate headlines for Spanish political news articles. For each article, the system must order ten candidate headlines according to their relevance to the article content. The task includes both a text-based and a multimodal setting.



\## Project overview



The project explores different ranking strategies, including:



\- Lexical and semantic baselines.

\- Transformer-based cross-encoders.

\- Weighted ensembles of Spanish language models.

\- Tail reranking strategies.

\- Vision-language models for lightweight multimodal fusion.

\- Exploratory zero-shot ranking with large language models.



The main experimental system is based on Spanish cross-encoder models, weighted score fusion, and a conservative tail reranking stage designed to refine the lower part of the ranking while preserving the top prediction.



\## Repository status



This repository is currently being cleaned and reorganized. Some scripts, paths, and configuration files may still be updated in future commits.



At this stage, the repository is mainly intended to document the experimental work carried out during the project and to provide access to the main training, evaluation, and analysis scripts.



\## Structure



```text

src/                 Main source code

data/                Dataset-related files or expected data structure

outputs/             Model outputs and predictions

results/             Evaluation results and metrics

notebooks/           Optional exploratory notebooks

README.md            Project description

````



The exact structure may change as the repository is further cleaned.



\## Technologies



The project was developed mainly with:



\* Python

\* PyTorch

\* Hugging Face Transformers

\* sentence-transformers

\* pandas, NumPy and scikit-learn

\* Google Colab

\* Vision-language models such as CLIP and SigLIP variants



\## Notes



The original dataset belongs to the PoliticHeadlinES-IberLEF 2026 shared task. Depending on the task rules and data availability, the dataset itself may not be included in this repository.



This repository is part of an academic project and is not intended as a production-ready system.



\## Authors



David Hernández Ruiz

University of Huelva

I2C Research Group

