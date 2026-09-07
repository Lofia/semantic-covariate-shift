"""Create a one-time per-example loss lookup for the fixed target population.

The fixed classifier is:
    outputs/semantic_pilot_frozen/final_model

The finite empirical target population is:
    data/processed/ag_news_small["train"]  (N = 2000)

For each target-population row j, save:
    pool_index
    label
    semantic_score
    cross_entropy
    correct

This lets repeated covariate-shift simulations reuse the fixed Transformer's
loss without running Transformer inference again.

No training is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--target_data_dir",
        default="data/processed/ag_news_small",
    )

    p.add_argument(
        "--semantic_score_file",
        default="data/semantic_business_scores.csv",
    )

    p.add_argument(
        "--pilot_model_dir",
        default="outputs/semantic_pilot_frozen/final_model",
    )

    p.add_argument(
        "--output_file",
        default=(
            "data/cov_shift_semantic_business_exp_b12/"
            "target_loss_lookup.csv"
        ),
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--max_length",
        type=int,
        default=128,
    )

    p.add_argument(
        "--use_cpu",
        action="store_true",
    )

    return p.parse_args()


def choose_device(use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    args = parse_args()

    target = load_from_disk(
        args.target_data_dir
    )["train"]

    n = len(target)

    score_df = (
        pd.read_csv(
            args.semantic_score_file
        )
        .sort_values("pool_index")
        .reset_index(drop=True)
    )

    if len(score_df) != n:
        raise ValueError(
            "Semantic-score file length does not match target population."
        )

    expected_index = np.arange(
        n,
        dtype=int,
    )

    observed_index = score_df[
        "pool_index"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        expected_index,
        observed_index,
    ):
        raise ValueError(
            "semantic score pool_index is not exactly 0,...,N-1."
        )

    device = choose_device(
        args.use_cpu
    )

    print("=" * 72)
    print("Create target loss lookup")
    print("=" * 72)
    print("Target N:", n)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pilot_model_dir
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.pilot_model_dir
        )
    )

    model.to(device)
    model.eval()

    texts = list(
        target["text"]
    )

    labels = np.asarray(
        target["label"],
        dtype=int,
    )

    losses = []
    predictions = []
    correct = []

    for start in range(
        0,
        n,
        args.batch_size,
    ):
        stop = min(
            start + args.batch_size,
            n,
        )

        encoded = tokenizer(
            texts[start:stop],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        y = torch.tensor(
            labels[start:stop],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = model(
                **encoded
            ).logits

            loss_i = F.cross_entropy(
                logits,
                y,
                reduction="none",
            )

            pred = logits.argmax(
                dim=-1
            )

        loss_np = (
            loss_i
            .detach()
            .cpu()
            .numpy()
        )

        pred_np = (
            pred
            .detach()
            .cpu()
            .numpy()
        )

        losses.extend(
            loss_np.tolist()
        )

        predictions.extend(
            pred_np.tolist()
        )

        correct.extend(
            (pred_np == labels[start:stop])
            .astype(int)
            .tolist()
        )

        if (
            stop == n
            or stop % 500 == 0
        ):
            print(
                f"Processed {stop}/{n}"
            )

    out = pd.DataFrame(
        {
            "pool_index":
                expected_index,

            "label":
                labels,

            "semantic_score":
                score_df[
                    "business_prob"
                ].to_numpy(
                    dtype=float
                ),

            "prediction":
                np.asarray(
                    predictions,
                    dtype=int,
                ),

            "cross_entropy":
                np.asarray(
                    losses,
                    dtype=float,
                ),

            "correct":
                np.asarray(
                    correct,
                    dtype=int,
                ),
        }
    )

    output_path = Path(
        args.output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_path,
        index=False,
    )

    exact_ce = float(
        out[
            "cross_entropy"
        ].mean()
    )

    exact_accuracy = float(
        out[
            "correct"
        ].mean()
    )

    print("\nExact finite-population target CE:")
    print(f"{exact_ce:.6f}")

    print(
        "Exact finite-population target accuracy:"
    )
    print(f"{exact_accuracy:.6f}")

    print(
        "\nSaved:",
        output_path.resolve(),
    )


if __name__ == "__main__":
    main()
