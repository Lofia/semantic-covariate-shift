"""Compute a label-free Business-semantic score for each benchmark training text.

Score
-----
    S(T) = P_pilot(Business | T)

The pilot model is the independently trained frozen-encoder pilot located at

    outputs/semantic_pilot_frozen/final_model

The true benchmark label is saved only for diagnostics. It is NOT used to
construct the score or the later selection probability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

LABEL_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]

BUSINESS_LABEL_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark_dir",
        default="data/processed/ag_news_small",
    )

    parser.add_argument(
        "--pilot_model_dir",
        default="outputs/semantic_pilot_frozen/final_model",
    )

    parser.add_argument(
        "--output_file",
        default="data/semantic_business_scores.csv",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size. Use 16 or 8 if MPS memory is tight.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--use_cpu",
        action="store_true",
    )

    return parser.parse_args()


def choose_device(
    use_cpu: bool,
) -> torch.device:

    if use_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("1. Load benchmark training pool")
    print("=" * 72)

    benchmark = load_from_disk(
        args.benchmark_dir
    )

    pool = benchmark["train"]

    print(
        "Benchmark training-pool size:",
        len(pool),
    )

    print("\n" + "=" * 72)
    print("2. Load independent frozen-encoder pilot")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pilot_model_dir
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.pilot_model_dir
        )
    )

    device = choose_device(
        args.use_cpu
    )

    model.to(device)
    model.eval()

    print(
        "Pilot model directory:",
        args.pilot_model_dir,
    )

    print(
        "Device:",
        device,
    )

    print(
        "Inference batch size:",
        args.batch_size,
    )

    print("\n" + "=" * 72)
    print("3. Compute S(T) = P_pilot(Business | T)")
    print("=" * 72)

    texts = pool["text"]

    labels = np.asarray(
        pool["label"],
        dtype=int,
    )

    business_probabilities: list[float] = []
    predicted_labels: list[int] = []

    for start in range(
        0,
        len(texts),
        args.batch_size,
    ):

        stop = min(
            start + args.batch_size,
            len(texts),
        )

        batch_texts = texts[
            start:stop
        ]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            logits = model(
                **inputs
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

        business_probabilities.extend(
            probabilities[
                :,
                BUSINESS_LABEL_ID,
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        predicted_labels.extend(
            probabilities
            .argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        if (
            stop == len(texts)
            or stop % 500 == 0
        ):
            print(
                f"Processed {stop}/{len(texts)} texts"
            )

    business_probabilities_array = np.asarray(
        business_probabilities,
        dtype=float,
    )

    predicted_labels_array = np.asarray(
        predicted_labels,
        dtype=int,
    )

    if len(
        business_probabilities_array
    ) != len(pool):

        raise RuntimeError(
            "Number of semantic scores does not "
            "match benchmark-pool size."
        )

    if not np.all(
        np.isfinite(
            business_probabilities_array
        )
    ):
        raise ValueError(
            "Semantic scores contain "
            "non-finite values."
        )

    if np.any(
        business_probabilities_array < 0
    ) or np.any(
        business_probabilities_array > 1
    ):
        raise ValueError(
            "Business probabilities are "
            "outside [0, 1]."
        )

    print("\n" + "=" * 72)
    print("4. Save semantic scores")
    print("=" * 72)

    score_df = pd.DataFrame(
        {
            "pool_index":
                np.arange(
                    len(pool),
                    dtype=int,
                ),

            "business_prob":
                business_probabilities_array,

            "pilot_prediction":
                predicted_labels_array,

            # Diagnostic only.
            # This is NOT used to define
            # the score or later selection.
            "true_label":
                labels,
        }
    )

    output_path = Path(
        args.output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "Saved semantic scores to:",
        output_path.resolve(),
    )

    print("\n" + "=" * 72)
    print("5. Semantic-score diagnostics")
    print("=" * 72)

    print(
        "\nOverall Business-probability summary:"
    )

    print(
        score_df[
            "business_prob"
        ].describe()
    )

    diagnostic_df = (
        score_df.copy()
    )

    diagnostic_df[
        "true_class"
    ] = [
        LABEL_NAMES[label]
        for label
        in diagnostic_df[
            "true_label"
        ]
    ]

    class_summary = (
        diagnostic_df
        .groupby(
            "true_class"
        )[
            "business_prob"
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
                "count",
            ]
        )
        .reindex(
            LABEL_NAMES
        )
    )

    print(
        "\nBusiness probability by TRUE class "
        "(diagnostic only):"
    )

    print(
        class_summary
    )

    pilot_accuracy = float(
        np.mean(
            predicted_labels_array
            == labels
        )
    )

    print(
        "\nIndependent pilot accuracy "
        "on benchmark training pool:",
        f"{pilot_accuracy:.4f}",
    )

    print("\nMean Business probability by true class:")

    for label_id, class_name in enumerate(
        LABEL_NAMES
    ):

        class_mask = (
            labels == label_id
        )

        class_mean = float(
            business_probabilities_array[
                class_mask
            ].mean()
        )

        print(
            f"  {class_name:<10s}: "
            f"{class_mean:.6f}"
        )

    print("\n" + "=" * 72)
    print("6. Sanity checks")
    print("=" * 72)

    business_mask = (
        labels
        == BUSINESS_LABEL_ID
    )

    non_business_mask = (
        labels
        != BUSINESS_LABEL_ID
    )

    mean_business = float(
        business_probabilities_array[
            business_mask
        ].mean()
    )

    mean_non_business = float(
        business_probabilities_array[
            non_business_mask
        ].mean()
    )

    print(
        "Mean score for true Business:",
        mean_business,
    )

    print(
        "Mean score for non-Business:",
        mean_non_business,
    )

    print(
        "Business semantic separation:",
        mean_business
        - mean_non_business,
    )

    if (
        mean_business
        <= mean_non_business
    ):

        print(
            "\nWARNING: Business probability does "
            "not separate Business from non-Business "
            "on average. Inspect the pilot model "
            "before constructing the shift."
        )

    else:

        print(
            "\nSemantic-score sanity check passed: "
            "true Business texts have higher "
            "Business scores on average."
        )


if __name__ == "__main__":
    main()
