from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup

from config import FORCE_CPU, TOKENS_ALL
from utils.data_utils import get_source_text_task1, get_titles
from utils.metrics import parse_rank_list, score_task1_predictions_df


DEFAULT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
DEFAULT_MAX_LENGTH = 512


def get_device() -> str:
    if FORCE_CPU:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _should_use_slow_tokenizer(model_name_or_path: str) -> bool:
    name = model_name_or_path.lower()
    return "deberta" in name


def extract_positive_index(row: pd.Series, y_true_col: str = "y_true") -> int:
    y_true_tokens = parse_rank_list(row[y_true_col])
    if not y_true_tokens:
        raise ValueError("Fila sin y_true válido.")
    top_token = y_true_tokens[0]
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
    use_amp: bool = True


class CrossEncoderRanker:
    def __init__(self, config: CrossEncoderConfig):
        self.config = config
        self.device = get_device()
        self.training_history: List[dict] = []

        tokenizer_kwargs = {}
        if _should_use_slow_tokenizer(config.model_name):
            tokenizer_kwargs["use_fast"] = False

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, **tokenizer_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
        )
        self.model = self.model.float()
        self.model = self.model.to(self.device)

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
        val_df: pd.DataFrame | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        train_loader = self._make_loader(train_examples, shuffle=True)

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-6,
            betas=(0.9, 0.999),
        )

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        use_amp = self.config.use_amp and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        save_root = Path(output_dir) if output_dir is not None else None
        if save_root is not None:
            save_root.mkdir(parents=True, exist_ok=True)

        best_score = float("-inf")
        best_epoch = None

        self.model.train()
        print(f"Dispositivo: {self.device}")
        print(f"AMP activado: {use_amp}")
        print("Model dtype:", next(self.model.parameters()).dtype)

        for epoch in range(self.config.epochs):
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast("cuda", enabled=True):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Loss no finita detectada en epoch {epoch + 1}, batch {batch_idx}: {loss.item()}"
                    )

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                total_loss += loss.item()

                if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                    print(
                        f"Epoch {epoch + 1}/{self.config.epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / max(len(train_loader), 1)
            history_entry = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
            }
            print(f"Epoch {epoch + 1} finished | avg train loss = {avg_loss:.4f}")

            if val_examples is not None:
                val_loss = self.evaluate_loss(val_examples)
                history_entry["val_loss"] = val_loss
                print(f"Epoch {epoch + 1} | val loss = {val_loss:.4f}")

            if val_df is not None and "y_true" in val_df.columns:
                metrics = self.evaluate_ranking_dataframe(val_df)
                history_entry.update(metrics)
                print(
                    f"Epoch {epoch + 1} | "
                    f"task_1_pa_ndcg = {metrics['task_1_pa_ndcg']:.6f} | "
                    f"top1_acc = {metrics['top1_acc']:.6f}"
                )

                if metrics["task_1_pa_ndcg"] > best_score:
                    best_score = metrics["task_1_pa_ndcg"]
                    best_epoch = epoch + 1
                    if save_root is not None:
                        self.save(str(save_root))
                        print(f"Nuevo mejor checkpoint guardado en: {save_root}")

            self.training_history.append(history_entry)
            if save_root is not None:
                self._save_training_history(save_root, best_epoch=best_epoch, best_score=best_score)

        if save_root is not None:
            last_dir = save_root / "last_checkpoint"
            last_dir.mkdir(parents=True, exist_ok=True)
            self.save(str(last_dir))
            self._save_training_history(save_root, best_epoch=best_epoch, best_score=best_score)

    def _save_training_history(self, output_dir: Path, best_epoch: int | None, best_score: float) -> None:
        history_payload = {
            "best_epoch": best_epoch,
            "best_task_1_pa_ndcg": None if best_score == float("-inf") else best_score,
            "history": self.training_history,
        }
        history_path = output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2, ensure_ascii=False)

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
        with torch.no_grad():
            articles = [article] * len(titles)
            enc = self.tokenizer(
                articles,
                titles,
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            positive_scores = logits[:, 1].detach().float().cpu().numpy()
        return positive_scores.astype(float)

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

    def evaluate_ranking_dataframe(self, df_val: pd.DataFrame) -> dict:
        preds = self.predict_dataframe(df_val)
        return score_task1_predictions_df(df_val, preds)

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(cls, model_dir: str, config: CrossEncoderConfig):
        instance = cls.__new__(cls)
        instance.config = config
        instance.device = get_device()
        instance.training_history = []

        tokenizer_kwargs = {}
        if _should_use_slow_tokenizer(model_dir) or _should_use_slow_tokenizer(config.model_name):
            tokenizer_kwargs["use_fast"] = False

        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.model = instance.model.float()
        instance.model = instance.model.to(instance.device)
        return instance
