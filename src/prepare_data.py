\
"""Download AG News, create train/validation/test subsets, and save them."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def balanced_subset(
    dataset: Dataset,
    per_class: int,
    seed: int,
) -> Dataset:
    """Select the same number of examples from every class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset["label"])
    chosen_indices: list[int] = []

    for label_id in range(len(LABEL_NAMES)):
        class_indices = np.flatnonzero(labels == label_id)

        if len(class_indices) < per_class:
            raise ValueError(
                f"Class {label_id} only has {len(class_indices)} rows, "
                f"but per_class={per_class} was requested."
            )

        selected = rng.choice(
            class_indices,
            size=per_class,
            replace=False,
        )
        chosen_indices.extend(selected.tolist())

    rng.shuffle(chosen_indices)
    return dataset.select(chosen_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="data/processed/ag_news_small",
        help="Folder used by DatasetDict.save_to_disk().",
    )
    parser.add_argument(
        "--train_per_class",
        type=int,
        default=500,
        help="Training examples retained for each of the four classes.",
    )
    parser.add_argument(
        "--validation_per_class",
        type=int,
        default=100,
        help="Validation examples retained for each class.",
    )
    parser.add_argument(
        "--test_per_class",
        type=int,
        default=200,
        help="Test examples retained for each class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("1. Downloading AG News from the Hugging Face Hub...")
    raw = load_dataset("fancyzhx/ag_news")

    print("2. Splitting the original training data into train/validation...")
    split = raw["train"].train_test_split(
        test_size=0.10,
        seed=args.seed,
        stratify_by_column="label",
    )

    print("3. Building balanced small subsets...")
    prepared = DatasetDict(
        {
            "train": balanced_subset(
                split["train"],
                per_class=args.train_per_class,
                seed=args.seed,
            ),
            "validation": balanced_subset(
                split["test"],
                per_class=args.validation_per_class,
                seed=args.seed + 1,
            ),
            "test": balanced_subset(
                raw["test"],
                per_class=args.test_per_class,
                seed=args.seed + 2,
            ),
        }
    )

    print("4. Saving the prepared dataset...")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(str(output_dir))

    print("\nPrepared dataset:")
    print(prepared)

    for split_name, dataset in prepared.items():
        counts = np.bincount(dataset["label"], minlength=4)
        readable_counts = {
            LABEL_NAMES[i]: int(counts[i])
            for i in range(len(LABEL_NAMES))
        }
        print(f"{split_name}: {readable_counts}")

    print(f"\nSaved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
