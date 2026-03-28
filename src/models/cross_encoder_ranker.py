from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup

from config import TOKENS_ALL
from data_utils import get_source_text_task1, get_titles
from metrics import parse_rank_list


DEFAULT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
DEFAULT_MAX_LENGTH = 512



def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_positive_index(row: pd.Series, y_true_col: str = "y_true") -> int:
    y_true_tokens = parse_rank_list(row[y_true_col])
    if not y_true_tokens:
        raise ValueError("Fila sin y_true válido.")
    top_token = y_true_tokens[0]  # p.ej. t9
    return int(top_token[1:]) - 1


def build_pair_examples(df: pd.DataFrame, y_true_col: str = "y_true") -> List[Tuple[str, str, int]]:
    examples = []

    for _, row in df.iterrows():
        article = get_source_text_task1(row)
        titles = get_titles(row)
        positive_idx = extract_positive_index(row, y_true_col=y_true_col)

        for i, title in enumerate(titles):
            label = 1 if i == positive_idx else 0
            examples.append((article, title, label))

    return examples


class PairDataset(Dataset):
    def __init__(
        self,
        examples: List[Tuple[str, str, int]],
        tokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        article, title, label = self.examples[idx]

        enc = self.tokenizer(
            article,
            title,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


@dataclass
class CrossEncoderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


class CrossEncoderRanker:
    def __init__(self, config: CrossEncoderConfig):
        self.config = config
        self.device = get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
        ).to(self.device)

    def _make_loader(
        self,
        examples: List[Tuple[str, str, int]],
        shuffle: bool = False,
    ) -> DataLoader:
        dataset = PairDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def fit(
        self,
        train_examples: List[Tuple[str, str, int]],
        val_examples: List[Tuple[str, str, int]] | None = None,
    ) -> None:
        train_loader = self._make_loader(train_examples, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.train()

        for epoch in range(self.config.epochs):
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.config.epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch + 1} finished | avg train loss = {avg_loss:.4f}")

            if val_examples is not None:
                val_loss = self.evaluate_loss(val_examples)
                print(f"Epoch {epoch + 1} | val loss = {val_loss:.4f}")

    def evaluate_loss(self, examples: List[Tuple[str, str, int]]) -> float:
        loader = self._make_loader(examples, shuffle=False)
        self.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / max(len(loader), 1)

    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        self.model.eval()

        scores = []
        with torch.no_grad():
            for title in titles:
                enc = self.tokenizer(
                    article,
                    title,
                    truncation=True,
                    max_length=self.config.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                positive_score = logits[0, 1].item()
                scores.append(positive_score)

        return np.array(scores, dtype=float)

    def rank_titles(self, article: str, titles: List[str]) -> List[str]:
        scores = self.score_titles(article, titles)
        order = np.argsort(-scores)
        return [TOKENS_ALL[i] for i in order]

    def predict_dataframe(self, df_pred: pd.DataFrame) -> List[str]:
        preds = []

        for idx, (_, row) in enumerate(df_pred.iterrows(), start=1):
            article = get_source_text_task1(row)
            titles = get_titles(row)
            preds.append(" ".join(self.rank_titles(article, titles)))

            if idx % 10 == 0:
                print(f"Predichas {idx}/{len(df_pred)} filas...")

        return preds
    
    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def load(cls, model_dir: str, config: CrossEncoderConfig):
        instance = cls.__new__(cls)
        instance.config = config
        instance.device = get_device()
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(instance.device)
        return instance