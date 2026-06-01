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
from utils.data_pairs import build_pairs, extract_positive_index
from utils.metrics import score_task1_predictions_df


DEFAULT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
DEFAULT_MAX_LENGTH = 512


def get_device() -> str:
    if FORCE_CPU:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _should_use_slow_tokenizer(model_name_or_path: str | Path) -> bool:
    name = str(model_name_or_path).lower()
    return "deberta" in name


def build_head_tail_text(
    text: str,
    tokenizer,
    head_tokens: int = 384,
    tail_tokens: int = 125,
    max_article_tokens: int | None = None,
) -> str:
    """
    Construye una versión head+tail del artículo respetando el presupuesto real
    de tokens disponible para el artículo.

    El presupuesto debe calcularse fuera teniendo en cuenta:
    - max_length del modelo,
    - tokens del titular,
    - tokens especiales del par ([CLS], [SEP], [SEP] o equivalentes).

    head_tokens y tail_tokens se usan como proporción deseada. Si su suma no cabe
    en max_article_tokens, se reescalan manteniendo aproximadamente la proporción.
    """
    text = "" if text is None else str(text)
    if not text:
        return text

    tokens = tokenizer.tokenize(text)

    if max_article_tokens is None:
        max_article_tokens = head_tokens + tail_tokens

    max_article_tokens = max(1, int(max_article_tokens))

    if len(tokens) <= max_article_tokens:
        return text

    requested_tokens = max(1, int(head_tokens) + int(tail_tokens))

    if requested_tokens <= max_article_tokens:
        head_budget = min(int(head_tokens), max_article_tokens)
        tail_budget = max_article_tokens - head_budget
    else:
        head_ratio = int(head_tokens) / requested_tokens
        head_budget = int(round(max_article_tokens * head_ratio))
        head_budget = min(max(head_budget, 1), max_article_tokens)
        tail_budget = max_article_tokens - head_budget

    head = tokens[:head_budget]
    tail = tokens[-tail_budget:] if tail_budget > 0 else []

    return tokenizer.convert_tokens_to_string(head + tail)


# Alias conservado por compatibilidad con scripts anteriores.
build_pair_examples = build_pairs


class PairDataset(Dataset):
    def __init__(self, examples: List[Tuple[str, str, int]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        article, title, label = self.examples[idx]
        return {
            "article": article,
            "title": title,
            "labels": label,
        }


@dataclass
class CrossEncoderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_amp: bool = True
    use_head_tail: bool = False
    head_tokens: int = 384
    tail_tokens: int = 125
    early_stopping_patience: int | None = 3
    early_stopping_min_delta: float = 0.0005
    early_stopping_monitor: str = "task_1_pa_ndcg"


class CrossEncoderRanker:
    def __init__(self, config: CrossEncoderConfig):
        self.config = config
        self.device = get_device()
        self.training_history: List[dict] = []

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

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

    def _article_token_budget(self, title: str) -> int:
        """
        Calcula cuántos tokens pueden dedicarse al artículo dentro del par
        artículo-titular.

        El tokenizer añadirá tokens especiales automáticamente, por lo que aquí
        se reserva espacio para ellos usando la API del tokenizer en vez de fijar
        manualmente el valor a 3.
        """
        title_tokens = self.tokenizer.tokenize("" if title is None else str(title))
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        return max(1, self.config.max_length - len(title_tokens) - special_tokens)

    def _prepare_article_for_title(self, article: str, title: str) -> str:
        if not self.config.use_head_tail:
            return article

        return build_head_tail_text(
            article,
            tokenizer=self.tokenizer,
            head_tokens=self.config.head_tokens,
            tail_tokens=self.config.tail_tokens,
            max_article_tokens=self._article_token_budget(title),
        )

    def _collate_fn(self, batch):
        titles = [item["title"] for item in batch]
        articles = [
            self._prepare_article_for_title(item["article"], item["title"])
            for item in batch
        ]
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

        enc = self.tokenizer(
            articles,
            titles,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )

        enc["labels"] = labels
        return enc

    def _make_loader(
        self,
        examples: List[Tuple[str, str, int]],
        shuffle: bool = False,
    ) -> DataLoader:
        dataset = PairDataset(examples=examples)

        use_cuda_loader = self.device == "cuda"
        num_workers = 2 if use_cuda_loader else 0

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=use_cuda_loader,
            persistent_workers=(use_cuda_loader and num_workers > 0),
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

        updates_per_epoch = max(1, (len(train_loader) + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps)
        total_steps = updates_per_epoch * self.config.epochs
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
        epochs_without_improvement = 0

        self.model.train()
        print(f"Dispositivo: {self.device}")
        print(f"AMP activado: {use_amp}")
        print("Model dtype:", next(self.model.parameters()).dtype)
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Early stopping patience: {self.config.early_stopping_patience}")
        print(f"Early stopping min_delta: {self.config.early_stopping_min_delta}")
        print(f"Early stopping monitor: {self.config.early_stopping_monitor}")

        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.config.epochs):
            total_loss = 0.0
            stop_training = False

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                if use_amp:
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=True,
                    ):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Loss no finita detectada en epoch {epoch + 1}, batch {batch_idx}: {loss.item()}"
                    )

                total_loss += loss.item()
                loss_for_backward = loss / self.config.gradient_accumulation_steps

                should_step = (batch_idx % self.config.gradient_accumulation_steps == 0) or (batch_idx == len(train_loader))

                if use_amp:
                    scaler.scale(loss_for_backward).backward()
                    if should_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_for_backward.backward()
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                if should_step:
                    scheduler.step()

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
                    f"top1_accuracy = {metrics['top1_accuracy']:.6f}"
                )

                monitor_name = self.config.early_stopping_monitor
                current_score = metrics.get(monitor_name)
                if current_score is None:
                    raise KeyError(
                        f"Métrica de early stopping no encontrada: {monitor_name!r}. "
                        f"Métricas disponibles: {sorted(metrics.keys())}"
                    )

                improvement = current_score - best_score
                if improvement > self.config.early_stopping_min_delta:
                    best_score = current_score
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    if save_root is not None:
                        self.save(str(save_root))
                        print(f"Nuevo mejor checkpoint guardado en: {save_root}")
                else:
                    epochs_without_improvement += 1
                    print(
                        f"Sin mejora suficiente en {monitor_name}: "
                        f"actual={current_score:.6f}, mejor={best_score:.6f}, "
                        f"delta={improvement:.6f}, "
                        f"paciencia={epochs_without_improvement}/"
                        f"{self.config.early_stopping_patience}"
                    )

                    if (
                        self.config.early_stopping_patience is not None
                        and epochs_without_improvement >= self.config.early_stopping_patience
                    ):
                        stop_training = True

            self.training_history.append(history_entry)
            if save_root is not None:
                self._save_training_history(save_root, best_epoch=best_epoch, best_score=best_score)

            if stop_training:
                print(
                    f"Early stopping activado en epoch {epoch + 1}. "
                    f"Mejor epoch: {best_epoch}, mejor {self.config.early_stopping_monitor}: {best_score:.6f}"
                )
                break

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

        if self.training_history:
            pd.DataFrame(self.training_history).to_csv(
                output_dir / "training_history.csv",
                index=False,
            )

    def evaluate_loss(self, examples: List[Tuple[str, str, int]]) -> float:
        loader = self._make_loader(examples, shuffle=False)
        self.model.eval()

        use_amp = self.config.use_amp and self.device == "cuda"

        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                if use_amp:
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=True,
                    ):
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / max(len(loader), 1)

    def score_titles(self, article: str, titles: List[str]) -> np.ndarray:
        self.model.eval()
        use_amp = self.config.use_amp and self.device == "cuda"

        with torch.no_grad():
            articles = [
                self._prepare_article_for_title(article, title)
                for title in titles
            ]
            enc = self.tokenizer(
                articles,
                titles,
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

            if use_amp:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    logits = self.model(**enc).logits
            else:
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

        if instance.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        tokenizer_kwargs = {}
        if _should_use_slow_tokenizer(model_dir) or _should_use_slow_tokenizer(config.model_name):
            tokenizer_kwargs["use_fast"] = False

        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.model = instance.model.float()
        instance.model = instance.model.to(instance.device)
        return instance