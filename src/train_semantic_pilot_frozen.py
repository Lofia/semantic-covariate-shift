"""Train an independent frozen-encoder AG News pilot classifier.

This pilot is used only to define a fixed text-only semantic score:
    S(T) = P_pilot(Business | T)

The pilot train/validation sets are drawn from the official AG News TRAIN split
after excluding every text used in the benchmark train/validation splits.
The benchmark test split is never used.

Memory-saving design:
- freeze the entire DistilBERT encoder
- train only pre_classifier + classifier
- keep the frozen encoder in eval mode during training
- save no intermediate checkpoints
- save exactly one final pilot model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark_dir",
        default="data/processed/ag_news_small",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/semantic_pilot_frozen",
    )
    parser.add_argument(
        "--pilot_train_per_class",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--pilot_val_per_class",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
    )

    return parser.parse_args()


def make_balanced_indices(
    labels: np.ndarray,
    eligible: np.ndarray,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in range(4):
        candidates = eligible[labels[eligible] == label].copy()
        rng.shuffle(candidates)

        required = train_per_class + val_per_class

        if len(candidates) < required:
            raise ValueError(
                f"Class {label} has only {len(candidates)} eligible examples; "
                f"{required} are required."
            )

        train_indices.extend(
            candidates[:train_per_class].tolist()
        )

        val_indices.extend(
            candidates[
                train_per_class:
                train_per_class + val_per_class
            ].tolist()
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def freeze_encoder(model: torch.nn.Module) -> None:
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False


def print_parameter_summary(model: torch.nn.Module) -> None:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nParameter summary")
    print("-" * 55)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable fraction:   {100 * trainable / total:.4f}%")

    print("\nTrainable parameter names:")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print("  ", name)


class FrozenEncoderTrainer(Trainer):
    """Keep frozen DistilBERT in eval mode while the head remains trainable."""

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        model.distilbert.eval()

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("1. Load benchmark and official AG News")
    print("=" * 72)

    benchmark: DatasetDict = load_from_disk(
        args.benchmark_dir
    )

    raw = load_dataset(
        "fancyzhx/ag_news"
    )

    benchmark_texts = set(
        benchmark["train"]["text"]
    )
    benchmark_texts.update(
        benchmark["validation"]["text"]
    )

    raw_train_texts = raw["train"]["text"]
    raw_train_labels = np.asarray(
        raw["train"]["label"],
        dtype=int,
    )

    eligible_mask = np.fromiter(
        (
            text not in benchmark_texts
            for text in raw_train_texts
        ),
        dtype=bool,
        count=len(raw_train_texts),
    )

    eligible_indices = np.flatnonzero(
        eligible_mask
    )

    print(f"Official AG News train size:      {len(raw['train'])}")
    print(
        f"Excluded benchmark texts:        "
        f"{len(raw['train']) - len(eligible_indices)}"
    )
    print(
        f"Eligible independent pilot pool: "
        f"{len(eligible_indices)}"
    )

    print("\n" + "=" * 72)
    print("2. Create balanced independent pilot data")
    print("=" * 72)

    train_indices, val_indices = make_balanced_indices(
        labels=raw_train_labels,
        eligible=eligible_indices,
        train_per_class=args.pilot_train_per_class,
        val_per_class=args.pilot_val_per_class,
        seed=args.seed,
    )

    pilot = DatasetDict(
        {
            "train": raw["train"].select(
                train_indices
            ),
            "validation": raw["train"].select(
                val_indices
            ),
        }
    )

    print(pilot)

    for split_name in ["train", "validation"]:
        labels = np.asarray(
            pilot[split_name]["label"],
            dtype=int,
        )
        counts = np.bincount(
            labels,
            minlength=4,
        )
        print(
            f"{split_name} class counts: "
            f"{counts.tolist()}"
        )

    split_metadata = {
        "seed": args.seed,
        "pilot_train_per_class": args.pilot_train_per_class,
        "pilot_val_per_class": args.pilot_val_per_class,
        "train_raw_indices": train_indices,
        "validation_raw_indices": val_indices,
    }

    with (
        output_dir / "pilot_split_indices.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_metadata,
            file,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("3. Tokenize pilot data")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize_batch(
        examples: dict[str, Any],
    ) -> dict[str, Any]:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = pilot.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing pilot data",
    )

    for split_name in tokenized.keys():
        keep_columns = {
            "input_ids",
            "attention_mask",
            "label",
        }

        removable_columns = [
            column
            for column
            in tokenized[split_name].column_names
            if column not in keep_columns
        ]

        tokenized[split_name] = (
            tokenized[split_name]
            .remove_columns(
                removable_columns
            )
        )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    print("\n" + "=" * 72)
    print("4. Load DistilBERT and freeze encoder")
    print("=" * 72)

    id2label = {
        index: label
        for index, label
        in enumerate(LABEL_NAMES)
    }
    label2id = {
        label: index
        for index, label
        in enumerate(LABEL_NAMES)
    }

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=4,
            id2label=id2label,
            label2id=label2id,
        )
    )

    freeze_encoder(model)
    print_parameter_summary(model)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(
        eval_pred: Any,
    ) -> dict[str, float]:
        logits, labels = eval_pred

        if isinstance(logits, tuple):
            logits = logits[0]

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=labels,
        )["accuracy"]

        macro_f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )["f1"]

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        }

    print("\n" + "=" * 72)
    print("5. Train classification head only")
    print("=" * 72)

    print("Train batch size:", args.train_batch_size)
    print("Eval batch size: ", args.eval_batch_size)
    print("Learning rate:   ", args.learning_rate)
    print("Epochs:          ", args.num_train_epochs)

    training_args = TrainingArguments(
        output_dir=str(
            output_dir / "temporary"
        ),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=
            args.train_batch_size,
        per_device_eval_batch_size=
            args.eval_batch_size,
        num_train_epochs=
            args.num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=args.use_cpu,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = FrozenEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    print("\n" + "=" * 72)
    print("6. Evaluate independent pilot validation set")
    print("=" * 72)

    validation_metrics = trainer.evaluate(
        tokenized["validation"],
        metric_key_prefix="validation",
    )

    print("\nPilot validation metrics:")
    print(validation_metrics)

    print("\n" + "=" * 72)
    print("7. Save one final pilot model")
    print("=" * 72)

    model_dir = output_dir / "final_model"

    trainer.save_model(
        str(model_dir)
    )

    tokenizer.save_pretrained(
        str(model_dir)
    )

    metrics = {
        "method": "independent_frozen_encoder_pilot",
        "base_model": MODEL_NAME,
        "seed": args.seed,
        "pilot_train_per_class": args.pilot_train_per_class,
        "pilot_val_per_class": args.pilot_val_per_class,
        "train_n": len(pilot["train"]),
        "validation_n": len(
            pilot["validation"]
        ),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "trainable_parameters": sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
        "training": train_result.metrics,
        "validation": validation_metrics,
    }

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        "Pilot model:",
        model_dir.resolve(),
    )
    print(
        "Pilot metrics:",
        metrics_path.resolve(),
    )


if __name__ == "__main__":
    main()
