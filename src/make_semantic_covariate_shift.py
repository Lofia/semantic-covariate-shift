"""Construct a strict monotone semantic-coverage covariate-shift benchmark.

Semantic score
--------------
For each benchmark training text T, use the independently trained frozen
pilot model to define

    S(T) = P_pilot(Business | T).

The current observation's true label is NOT used to construct S(T).

Randomized empirical PIT
------------------------
Treat the 2000-example benchmark training pool as a finite empirical target
population. For an IID draw from that population with score S=s,

    U = F_hat(s-) + R * P_hat(S=s),    R ~ Uniform(0,1),

so U is Uniform(0,1) under the empirical-population DGP. Then

    X = -log(1-U) ~ Exp(1).

Monotone selection
------------------
Use

    v_b(x) = 0.2 + 0.8 * (10x + 1) / (10x + 1 + b),

which is strictly increasing for b > 0.

The selected source density is

    g(x) = v_b(x) f(x) / Z,

where f(x)=exp(-x) is the Exp(1) target density and

    Z = E_f[v_b(X)].

The oracle target/source density ratio is therefore

    w(x) = f(x) / g(x) = Z / v_b(x),

which is strictly decreasing.

Default experiment
------------------
b = 12 is fixed before downstream model evaluation.
n_candidates = 3800 gives an expected selected source size close to 2000.

Outputs
-------
data/cov_shift_semantic_business_exp_b12/
    shifted_dataset/
    control_same_n_dataset/
    source_x.csv
    target_x.csv
    metadata.json
    figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from scipy.integrate import quad
from scipy.stats import kstest


LABEL_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="data/processed/ag_news_small",
    )

    parser.add_argument(
        "--score_file",
        default="data/semantic_business_scores.csv",
    )

    parser.add_argument(
        "--output_dir",
        default="data/cov_shift_semantic_business_exp_b12",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--n_candidates",
        type=int,
        default=3800,
        help=(
            "Number of IID target-population draws before "
            "Bernoulli selection."
        ),
    )

    parser.add_argument(
        "--n_target_x",
        type=int,
        default=2000,
        help=(
            "Independent unbiased target-X sample size "
            "for density-ratio estimation/diagnostics."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def selection_probability(
    x: np.ndarray | float,
    b: float,
) -> np.ndarray | float:
    """Strictly increasing selection probability v_b(x)."""

    return (
        0.2
        + 0.8
        * (10.0 * x + 1.0)
        / (10.0 * x + 1.0 + b)
    )


def build_empirical_pit_arrays(
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute F_hat(S-) and P_hat(S=s), aligned with pool rows.

    Using np.unique avoids fragile float-valued dictionary lookup.
    """

    _, inverse, counts = np.unique(
        scores,
        return_inverse=True,
        return_counts=True,
    )

    mass_unique = (
        counts.astype(float)
        / len(scores)
    )

    lower_unique = (
        np.cumsum(mass_unique)
        - mass_unique
    )

    lower_by_row = (
        lower_unique[inverse]
    )

    mass_by_row = (
        mass_unique[inverse]
    )

    return (
        lower_by_row,
        mass_by_row,
    )


def sample_from_empirical_target(
    pool,
    pool_scores: np.ndarray,
    lower_by_row: np.ndarray,
    mass_by_row: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> dict:
    """IID sample with replacement from the finite empirical target population."""

    indices = rng.integers(
        low=0,
        high=len(pool),
        size=n,
    )

    jitter = rng.random(n)

    u = (
        lower_by_row[indices]
        + jitter
        * mass_by_row[indices]
    )

    eps = np.finfo(float).eps

    u = np.clip(
        u,
        eps,
        1.0 - eps,
    )

    x = -np.log1p(-u)

    texts = [
        pool[int(index)]["text"]
        for index in indices
    ]

    labels = np.asarray(
        [
            int(
                pool[int(index)]["label"]
            )
            for index in indices
        ],
        dtype=int,
    )

    return {
        "pool_index":
            indices.astype(int),

        "text":
            texts,

        "label":
            labels,

        "semantic_score":
            pool_scores[indices],

        "u":
            u,

        "x":
            x,
    }


def calculate_true_z(
    b: float,
) -> float:
    """Calculate Z = E_{Exp(1)}[v_b(X)] numerically."""

    def integrand(
        x: float,
    ) -> float:

        return (
            float(
                selection_probability(
                    x,
                    b,
                )
            )
            * np.exp(-x)
        )

    result, _ = quad(
        integrand,
        0.0,
        np.inf,
        limit=200,
    )

    return float(result)


def class_proportions(
    labels: np.ndarray,
) -> pd.Series:
    """Return proportions in the fixed AG News class order."""

    counts = (
        pd.Series(labels)
        .value_counts(
            normalize=True
        )
    )

    return pd.Series(
        {
            LABEL_NAMES[label]:
                float(
                    counts.get(
                        label,
                        0.0,
                    )
                )
            for label
            in range(4)
        }
    )


def main() -> None:
    args = parse_args()

    if args.b <= 0:
        raise ValueError(
            "b must be > 0 so that the selection "
            "function is strictly increasing."
        )

    rng = np.random.default_rng(
        args.seed
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # 1. Load target population and semantic scores
    # ==============================================================

    print("=" * 72)
    print("1. Load benchmark target population and semantic scores")
    print("=" * 72)

    data: DatasetDict = (
        load_from_disk(
            args.data_dir
        )
    )

    pool = data["train"]

    score_df = pd.read_csv(
        args.score_file
    )

    score_df = (
        score_df
        .sort_values(
            "pool_index"
        )
        .reset_index(
            drop=True
        )
    )

    if len(score_df) != len(pool):
        raise ValueError(
            "Score-file length does not match "
            "benchmark training-pool length."
        )

    expected_indices = np.arange(
        len(pool),
        dtype=int,
    )

    observed_indices = score_df[
        "pool_index"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        observed_indices,
        expected_indices,
    ):
        raise ValueError(
            "pool_index is not exactly 0,...,N-1 "
            "after sorting."
        )

    pool_scores = score_df[
        "business_prob"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(
            pool_scores
        )
    ):
        raise ValueError(
            "Semantic scores contain "
            "non-finite values."
        )

    if (
        np.any(pool_scores < 0.0)
        or np.any(pool_scores > 1.0)
    ):
        raise ValueError(
            "Business probabilities must lie "
            "in [0,1]."
        )

    print(
        "Target-population size:",
        len(pool),
    )

    print(
        "\nSemantic score summary:"
    )

    print(
        pd.Series(
            pool_scores
        ).describe()
    )

    print(
        "\nNumber of distinct semantic scores:",
        len(
            np.unique(
                pool_scores
            )
        ),
    )

    # ==============================================================
    # 2. Randomized empirical PIT
    # ==============================================================

    print("\n" + "=" * 72)
    print("2. Build randomized empirical PIT")
    print("=" * 72)

    (
        lower_by_row,
        mass_by_row,
    ) = build_empirical_pit_arrays(
        pool_scores
    )

    # ==============================================================
    # 3. Draw source candidates and select them
    # ==============================================================

    print("\n" + "=" * 72)
    print("3. Draw source candidates and apply monotone selection")
    print("=" * 72)

    candidates = sample_from_empirical_target(
        pool=pool,
        pool_scores=pool_scores,
        lower_by_row=lower_by_row,
        mass_by_row=mass_by_row,
        n=args.n_candidates,
        rng=rng,
    )

    x_candidates = candidates[
        "x"
    ]

    p_select = np.asarray(
        selection_probability(
            x_candidates,
            args.b,
        ),
        dtype=float,
    )

    selected = (
        rng.random(
            args.n_candidates
        )
        < p_select
    )

    n_source = int(
        selected.sum()
    )

    z_true = calculate_true_z(
        args.b
    )

    expected_source_n = (
        args.n_candidates
        * z_true
    )

    print(
        "b:",
        args.b,
    )

    print(
        "Candidates:",
        args.n_candidates,
    )

    print(
        "Theoretical selection rate Z:",
        z_true,
    )

    print(
        "Expected selected source n:",
        expected_source_n,
    )

    print(
        "Selected source:",
        n_source,
    )

    print(
        "Observed selection rate:",
        n_source
        / args.n_candidates,
    )

    # ==============================================================
    # 4. Oracle density-ratio weights
    # ==============================================================

    print("\n" + "=" * 72)
    print("4. Compute oracle importance weights")
    print("=" * 72)

    true_weight_all = (
        z_true
        / p_select
    )

    true_weight = (
        true_weight_all[
            selected
        ]
    )

    true_weight_normalized = (
        true_weight
        / true_weight.mean()
    )

    ess = (
        true_weight.sum() ** 2
        / np.square(
            true_weight
        ).sum()
    )

    cv2 = (
        np.var(
            true_weight,
            ddof=1,
        )
        / np.mean(
            true_weight
        ) ** 2
    )

    print(
        "True Z:",
        z_true,
    )

    print(
        "Mean raw oracle w:",
        true_weight.mean(),
    )

    print(
        "Mean normalized oracle w:",
        true_weight_normalized.mean(),
    )

    print(
        "Min raw oracle w:",
        true_weight.min(),
    )

    print(
        "Max raw oracle w:",
        true_weight.max(),
    )

    print(
        "CV^2:",
        cv2,
    )

    print(
        "ESS:",
        ess,
        "/",
        n_source,
    )

    # ==============================================================
    # 5. Create shifted training dataset
    # ==============================================================

    print("\n" + "=" * 72)
    print("5. Save shifted source and same-size unshifted control")
    print("=" * 72)

    source_labels = candidates[
        "label"
    ][
        selected
    ]

    source_scores = candidates[
        "semantic_score"
    ][
        selected
    ]

    source_x = x_candidates[
        selected
    ]

    selected_texts = (
        np.asarray(
            candidates["text"],
            dtype=object,
        )[
            selected
        ]
        .tolist()
    )

    source_train = (
        Dataset.from_dict(
            {
                "row_id":
                    np.arange(
                        n_source,
                        dtype=int,
                    ).tolist(),

                "text":
                    selected_texts,

                "label":
                    source_labels.tolist(),

                "semantic_score":
                    source_scores.tolist(),

                "x":
                    source_x.tolist(),

                "selection_prob":
                    p_select[
                        selected
                    ].tolist(),

                "true_weight":
                    true_weight.tolist(),

                "true_weight_normalized":
                    true_weight_normalized.tolist(),
            }
        )
    )

    shifted_dataset = DatasetDict(
        {
            "train":
                source_train,

            "validation":
                data["validation"],

            "test":
                data["test"],
        }
    )

    shifted_path = (
        output_dir
        / "shifted_dataset"
    )

    shifted_dataset.save_to_disk(
        str(
            shifted_path
        )
    )

    # Same-size unshifted control:
    # IID sampling with replacement from the target empirical population.
    control_indices = rng.integers(
        low=0,
        high=len(pool),
        size=n_source,
    )

    control_train = (
        Dataset.from_dict(
            {
                "text":
                    [
                        pool[
                            int(index)
                        ]["text"]
                        for index
                        in control_indices
                    ],

                "label":
                    [
                        int(
                            pool[
                                int(index)
                            ]["label"]
                        )
                        for index
                        in control_indices
                    ],
            }
        )
    )

    control_dataset = DatasetDict(
        {
            "train":
                control_train,

            "validation":
                data["validation"],

            "test":
                data["test"],
        }
    )

    control_path = (
        output_dir
        / "control_same_n_dataset"
    )

    control_dataset.save_to_disk(
        str(
            control_path
        )
    )

    print(
        "Shifted dataset:",
        shifted_path,
    )

    print(
        "Same-n control:",
        control_path,
    )

    # ==============================================================
    # 6. Independent unbiased target-X sample
    # ==============================================================

    print("\n" + "=" * 72)
    print("6. Draw independent unbiased target-X sample")
    print("=" * 72)

    target_sample = sample_from_empirical_target(
        pool=pool,
        pool_scores=pool_scores,
        lower_by_row=lower_by_row,
        mass_by_row=mass_by_row,
        n=args.n_target_x,
        rng=rng,
    )

    target_x = target_sample[
        "x"
    ]

    source_df = pd.DataFrame(
        {
            "row_id":
                np.arange(
                    n_source,
                    dtype=int,
                ),

            "x":
                source_x,

            "semantic_score":
                source_scores,

            "selection_prob":
                p_select[
                    selected
                ],

            "true_weight":
                true_weight,

            "true_weight_normalized":
                true_weight_normalized,

            # Kept only for later diagnostics.
            "label":
                source_labels,
        }
    )

    # Important:
    # target_x.csv intentionally contains no target labels.
    target_df = pd.DataFrame(
        {
            "x":
                target_x,

            "semantic_score":
                target_sample[
                    "semantic_score"
                ],
        }
    )

    source_df.to_csv(
        output_dir
        / "source_x.csv",
        index=False,
    )

    target_df.to_csv(
        output_dir
        / "target_x.csv",
        index=False,
    )

    # ==============================================================
    # 7. Distribution diagnostics
    # ==============================================================

    print("\n" + "=" * 72)
    print("7. Distribution diagnostics")
    print("=" * 72)

    ks = kstest(
        target_x,
        "expon",
    )

    print(
        "KS statistic vs Exp(1):",
        ks.statistic,
    )

    print(
        "KS p-value:",
        ks.pvalue,
    )

    print(
        "\nTarget X mean:",
        target_x.mean(),
    )

    print(
        "Source X mean:",
        source_x.mean(),
    )

    print(
        "\nTarget semantic-score mean:",
        np.mean(
            target_sample[
                "semantic_score"
            ]
        ),
    )

    print(
        "Source semantic-score mean:",
        np.mean(
            source_scores
        ),
    )

    target_class_prop = (
        class_proportions(
            target_sample[
                "label"
            ]
        )
    )

    source_class_prop = (
        class_proportions(
            source_labels
        )
    )

    print(
        "\nTarget class proportions:"
    )

    print(
        target_class_prop
    )

    print(
        "\nSource class proportions:"
    )

    print(
        source_class_prop
    )

    # ==============================================================
    # 8. Common target-defined semantic-score quintiles
    # ==============================================================

    print("\n" + "=" * 72)
    print("8. Semantic-score quintile composition")
    print("=" * 72)

    target_scores = np.asarray(
        target_sample[
            "semantic_score"
        ],
        dtype=float,
    )

    cut_points = np.quantile(
        target_scores,
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
    )

    if np.any(
        np.diff(
            cut_points
        )
        <= 0
    ):
        raise ValueError(
            "Target semantic-score quintile boundaries "
            "are not unique. Inspect score ties."
        )

    cut_points[
        0
    ] = -np.inf

    cut_points[
        -1
    ] = np.inf

    quintile_labels = [
        "Q1-low-Business",
        "Q2",
        "Q3",
        "Q4",
        "Q5-high-Business",
    ]

    target_groups = pd.cut(
        target_scores,
        bins=cut_points,
        labels=quintile_labels,
        include_lowest=True,
    )

    source_groups = pd.cut(
        source_scores,
        bins=cut_points,
        labels=quintile_labels,
        include_lowest=True,
    )

    target_quintile_prop = (
        pd.Series(
            target_groups
        )
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    source_quintile_prop = (
        pd.Series(
            source_groups
        )
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print(
        "\nTarget semantic-score composition:"
    )

    print(
        target_quintile_prop
    )

    print(
        "\nSource semantic-score composition:"
    )

    print(
        source_quintile_prop
    )

    # ==============================================================
    # 9. Monotonicity diagnostics
    # ==============================================================

    print("\n" + "=" * 72)
    print("9. Monotonicity diagnostics")
    print("=" * 72)

    grid_max = max(
        6.0,
        float(
            np.quantile(
                target_x,
                0.999,
            )
        ),
    )

    grid = np.linspace(
        0.0,
        grid_max,
        500,
    )

    v_grid = np.asarray(
        selection_probability(
            grid,
            args.b,
        ),
        dtype=float,
    )

    w_grid = (
        z_true
        / v_grid
    )

    selection_increasing = bool(
        np.all(
            np.diff(
                v_grid
            )
            > 0
        )
    )

    weight_decreasing = bool(
        np.all(
            np.diff(
                w_grid
            )
            < 0
        )
    )

    print(
        "Selection function increasing:",
        selection_increasing,
    )

    print(
        "Density ratio decreasing:",
        weight_decreasing,
    )

    # ==============================================================
    # 10. Save diagnostic figures
    # ==============================================================

    print("\n" + "=" * 72)
    print("10. Save diagnostic figures")
    print("=" * 72)

    figure_dir = (
        output_dir
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_grid_max = max(
        float(
            np.quantile(
                target_x,
                0.995,
            )
        ),
        float(
            np.quantile(
                source_x,
                0.995,
            )
        ),
    )

    plot_grid = np.linspace(
        0.0,
        plot_grid_max,
        350,
    )

    # X-distribution plot.
    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        target_x,
        bins=40,
        density=True,
        alpha=0.45,
        label="Target X",
    )

    plt.hist(
        source_x,
        bins=40,
        density=True,
        alpha=0.45,
        label="Shifted source X",
    )

    plt.plot(
        plot_grid,
        np.exp(
            -plot_grid
        ),
        linewidth=2,
        label="Exp(1)",
    )

    plt.xlabel(
        "Constructed covariate X"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Semantic covariate shift: target vs source X"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "x_target_vs_source.png",
        dpi=160,
    )

    plt.close()

    # Class-composition plot.
    target_prop = (
        target_class_prop
        .reindex(
            LABEL_NAMES
        )
    )

    source_prop = (
        source_class_prop
        .reindex(
            LABEL_NAMES
        )
    )

    positions = np.arange(
        len(
            LABEL_NAMES
        )
    )

    width = 0.36

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        positions
        - width / 2,
        target_prop.to_numpy(),
        width=width,
        label="Target",
    )

    plt.bar(
        positions
        + width / 2,
        source_prop.to_numpy(),
        width=width,
        label="Shifted source",
    )

    plt.xticks(
        positions,
        LABEL_NAMES,
    )

    plt.ylabel(
        "Proportion"
    )

    plt.title(
        "Class composition induced by text-only semantic selection"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "class_composition.png",
        dpi=160,
    )

    plt.close()

    # Selection function.
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        grid,
        v_grid,
        linewidth=2,
    )

    plt.xlabel("X")
    plt.ylabel("v(X)")
    plt.title(
        "Monotone increasing selection function"
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "selection_function.png",
        dpi=160,
    )

    plt.close()

    # Oracle weight function.
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        grid,
        w_grid,
        linewidth=2,
    )

    plt.xlabel("X")
    plt.ylabel("w(X)")
    plt.title(
        "Oracle target/source density ratio"
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / "oracle_weight_function.png",
        dpi=160,
    )

    plt.close()

    # ==============================================================
    # 11. Save metadata
    # ==============================================================

    metadata = {
        "score_definition":
            "P_independent_frozen_pilot(Business | text)",

        "score_file":
            args.score_file,

        "b":
            args.b,

        "z_true":
            z_true,

        "seed":
            args.seed,

        "n_candidates":
            args.n_candidates,

        "expected_source_n":
            expected_source_n,

        "n_source":
            n_source,

        "n_target_x":
            args.n_target_x,

        "observed_selection_rate":
            n_source
            / args.n_candidates,

        "weight_mean_raw":
            float(
                true_weight.mean()
            ),

        "weight_mean_normalized":
            float(
                true_weight_normalized.mean()
            ),

        "weight_min_raw":
            float(
                true_weight.min()
            ),

        "weight_max_raw":
            float(
                true_weight.max()
            ),

        "weight_cv2":
            float(
                cv2
            ),

        "effective_sample_size":
            float(
                ess
            ),

        "target_x_mean":
            float(
                target_x.mean()
            ),

        "source_x_mean":
            float(
                source_x.mean()
            ),

        "target_semantic_score_mean":
            float(
                target_scores.mean()
            ),

        "source_semantic_score_mean":
            float(
                source_scores.mean()
            ),

        "selection_function_increasing":
            selection_increasing,

        "density_ratio_decreasing":
            weight_decreasing,

        "shifted_dataset":
            str(
                shifted_path
            ),

        "control_same_n_dataset":
            str(
                control_path
            ),
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("Done")
    print("=" * 72)

    print(
        "Output folder:",
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
