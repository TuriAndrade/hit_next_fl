from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = EXPERIMENTS_DIR / "results_analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"

DATASETS = ("code15", "ptbxl", "chapman")
DATASET_LABELS = {
    "code15": "CODE-15",
    "ptbxl": "PTB-XL",
    "chapman": "Chapman",
}
REGIMES = ("1k", "5k", "10k")
METRICS = ("accuracy", "precision", "recall", "f1")
JSON_METRICS = {
    "accuracy": "acc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

INDIVIDUAL_RUNS = {
    "code15": {"1k": "29501", "5k": "29502", "10k": "29503"},
    "ptbxl": {"1k": "29504", "5k": "29505", "10k": "29506"},
    "chapman": {"1k": "29507", "5k": "29508", "10k": "29509"},
}
BALANCED_RUNS = {"1k": "29510", "5k": "29511", "10k": "29512"}
UNBALANCED_RUNS = {
    "code15": "code_unb",
    "ptbxl": "ptbxl_unb",
    "chapman": "chapman_unb",
}


def _macro_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")

    summary = json.loads(path.read_text())
    test_metrics = summary.get("test_metrics")
    if not test_metrics:
        raise ValueError(f"No test_metrics found in {path}")

    return {
        metric: float(
            np.mean(
                [
                    float(class_metrics[json_name])
                    for class_metrics in test_metrics.values()
                ]
            )
        )
        for metric, json_name in JSON_METRICS.items()
    }


def _individual_path(source: str, regime: str, target: str) -> Path:
    if source == target:
        run = INDIVIDUAL_RUNS[source][regime]
        return (
            EXPERIMENTS_DIR
            / f"ecg_hit_next-{source}-clf-seed_0"
            / run
            / "eval_summary.json"
        )

    return (
        EXPERIMENTS_DIR
        / "cross_domain_eval"
        / f"{source}_{regime}_to_{target}"
        / "eval_summary.json"
    )


def _collect_evaluations() -> pd.DataFrame:
    rows: list[dict] = []

    for source in DATASETS:
        for regime in REGIMES:
            for target in DATASETS:
                rows.append(
                    {
                        "model_type": "individual",
                        "setup": f"{source}_{regime}",
                        "regime": regime,
                        "source_dataset": source,
                        "target_dataset": target,
                        "evaluation_type": (
                            "in_domain" if source == target else "cross_domain"
                        ),
                        **_macro_metrics(_individual_path(source, regime, target)),
                    }
                )

    for regime, run in BALANCED_RUNS.items():
        for target in DATASETS:
            path = (
                EXPERIMENTS_DIR
                / "federated"
                / run
                / "final_eval"
                / target
                / "eval_summary.json"
            )
            rows.append(
                {
                    "model_type": "balanced_federated",
                    "setup": f"balanced_{regime}",
                    "regime": regime,
                    "source_dataset": "federated",
                    "target_dataset": target,
                    "evaluation_type": "federated",
                    **_macro_metrics(path),
                }
            )

    for low_resource, run in UNBALANCED_RUNS.items():
        for target in DATASETS:
            path = (
                EXPERIMENTS_DIR
                / "federated"
                / run
                / "final_eval"
                / target
                / "eval_summary.json"
            )
            rows.append(
                {
                    "model_type": "unbalanced_federated",
                    "setup": run,
                    "regime": "unbalanced",
                    "source_dataset": "federated",
                    "target_dataset": target,
                    "evaluation_type": (
                        "low_resource" if target == low_resource else "high_resource"
                    ),
                    "low_resource_dataset": low_resource,
                    **_macro_metrics(path),
                }
            )

    return pd.DataFrame(rows)


def _individual_generalization_summary(individual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in DATASETS:
        for regime in REGIMES:
            subset = individual[
                (individual["source_dataset"] == source)
                & (individual["regime"] == regime)
            ]
            in_domain = subset[subset["target_dataset"] == source].iloc[0]
            cross_domain = subset[subset["target_dataset"] != source]

            for metric in METRICS:
                cross_mean = float(cross_domain[metric].mean())
                rows.append(
                    {
                        "source_dataset": source,
                        "regime": regime,
                        "metric": metric,
                        "in_domain": float(in_domain[metric]),
                        "cross_domain_mean": cross_mean,
                        "generalization_gap": cross_mean - float(in_domain[metric]),
                    }
                )
    return pd.DataFrame(rows)


def _balanced_comparison(
    individual: pd.DataFrame,
    balanced: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        for target in DATASETS:
            in_domain = individual[
                (individual["regime"] == regime)
                & (individual["source_dataset"] == target)
                & (individual["target_dataset"] == target)
            ].iloc[0]
            foreign = individual[
                (individual["regime"] == regime)
                & (individual["source_dataset"] != target)
                & (individual["target_dataset"] == target)
            ]
            federated = balanced[
                (balanced["regime"] == regime)
                & (balanced["target_dataset"] == target)
            ].iloc[0]

            for metric in METRICS:
                individual_value = float(in_domain[metric])
                foreign_mean = float(foreign[metric].mean())
                federated_value = float(federated[metric])
                rows.append(
                    {
                        "regime": regime,
                        "target_dataset": target,
                        "metric": metric,
                        "individual_in_domain": individual_value,
                        "foreign_individual_mean": foreign_mean,
                        "balanced_federated": federated_value,
                        "fl_target_delta": federated_value - individual_value,
                        "fl_generalization_delta": federated_value - foreign_mean,
                    }
                )
    return pd.DataFrame(rows)


def _unbalanced_low_resource_comparison(
    individual: pd.DataFrame,
    balanced: pd.DataFrame,
    unbalanced: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for low_resource, setup in UNBALANCED_RUNS.items():
        individual_row = individual[
            (individual["source_dataset"] == low_resource)
            & (individual["target_dataset"] == low_resource)
            & (individual["regime"] == "1k")
        ].iloc[0]
        balanced_row = balanced[
            (balanced["target_dataset"] == low_resource)
            & (balanced["regime"] == "1k")
        ].iloc[0]
        unbalanced_row = unbalanced[
            (unbalanced["setup"] == setup)
            & (unbalanced["target_dataset"] == low_resource)
        ].iloc[0]

        for metric in METRICS:
            individual_value = float(individual_row[metric])
            balanced_value = float(balanced_row[metric])
            unbalanced_value = float(unbalanced_row[metric])
            rows.append(
                {
                    "low_resource_dataset": low_resource,
                    "metric": metric,
                    "individual_1k": individual_value,
                    "balanced_fl_1k": balanced_value,
                    "unbalanced_fl": unbalanced_value,
                    "unbalanced_vs_individual_1k": (
                        unbalanced_value - individual_value
                    ),
                    "unbalanced_vs_balanced_1k": unbalanced_value - balanced_value,
                }
            )
    return pd.DataFrame(rows)


def _unbalanced_high_resource_impact(
    individual: pd.DataFrame,
    unbalanced: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for low_resource, setup in UNBALANCED_RUNS.items():
        for target in DATASETS:
            if target == low_resource:
                continue

            individual_row = individual[
                (individual["source_dataset"] == target)
                & (individual["target_dataset"] == target)
                & (individual["regime"] == "10k")
            ].iloc[0]
            unbalanced_row = unbalanced[
                (unbalanced["setup"] == setup)
                & (unbalanced["target_dataset"] == target)
            ].iloc[0]

            for metric in METRICS:
                individual_value = float(individual_row[metric])
                unbalanced_value = float(unbalanced_row[metric])
                rows.append(
                    {
                        "low_resource_dataset": low_resource,
                        "high_resource_dataset": target,
                        "metric": metric,
                        "individual_10k": individual_value,
                        "unbalanced_fl": unbalanced_value,
                        "fl_delta": unbalanced_value - individual_value,
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_statistics(
    generalization: pd.DataFrame,
    balanced: pd.DataFrame,
    low_resource: pd.DataFrame,
    high_resource: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    individual_by_dataset = (
        generalization.groupby(["source_dataset", "metric"], as_index=False)[
            ["in_domain", "cross_domain_mean", "generalization_gap"]
        ]
        .mean()
        .rename(
            columns={
                "in_domain": "mean_in_domain",
                "cross_domain_mean": "mean_cross_domain",
                "generalization_gap": "mean_generalization_gap",
            }
        )
    )

    balanced_by_dataset = (
        balanced.groupby(["target_dataset", "metric"], as_index=False)[
            [
                "individual_in_domain",
                "foreign_individual_mean",
                "balanced_federated",
                "fl_target_delta",
                "fl_generalization_delta",
            ]
        ]
        .mean()
        .rename(
            columns={
                "individual_in_domain": "mean_individual_in_domain",
                "foreign_individual_mean": "mean_foreign_individual",
                "balanced_federated": "mean_balanced_federated",
                "fl_target_delta": "mean_fl_target_delta",
                "fl_generalization_delta": "mean_fl_generalization_delta",
            }
        )
    )

    high_resource_by_dataset = (
        high_resource.groupby(["high_resource_dataset", "metric"], as_index=False)[
            ["individual_10k", "unbalanced_fl", "fl_delta"]
        ]
        .mean()
        .rename(
            columns={
                "individual_10k": "mean_individual_10k",
                "unbalanced_fl": "mean_unbalanced_fl",
                "fl_delta": "mean_fl_delta",
            }
        )
    )

    high_resource_by_scenario = (
        high_resource.groupby(["low_resource_dataset", "metric"], as_index=False)[
            ["individual_10k", "unbalanced_fl", "fl_delta"]
        ]
        .mean()
        .rename(
            columns={
                "individual_10k": "mean_individual_10k",
                "unbalanced_fl": "mean_unbalanced_fl",
                "fl_delta": "mean_fl_delta",
            }
        )
    )

    overall_rows = []
    for metric in METRICS:
        individual_metric = generalization[generalization["metric"] == metric]
        balanced_metric = balanced[balanced["metric"] == metric]
        low_metric = low_resource[low_resource["metric"] == metric]
        high_metric = high_resource[high_resource["metric"] == metric]
        overall_rows.append(
            {
                "metric": metric,
                "individual_generalization_gap": float(
                    individual_metric["generalization_gap"].mean()
                ),
                "balanced_fl_target_delta": float(
                    balanced_metric["fl_target_delta"].mean()
                ),
                "balanced_fl_generalization_delta": float(
                    balanced_metric["fl_generalization_delta"].mean()
                ),
                "unbalanced_low_vs_individual_1k": float(
                    low_metric["unbalanced_vs_individual_1k"].mean()
                ),
                "unbalanced_low_vs_balanced_1k": float(
                    low_metric["unbalanced_vs_balanced_1k"].mean()
                ),
                "unbalanced_high_resource_delta": float(
                    high_metric["fl_delta"].mean()
                ),
            }
        )
    overall_differences = pd.DataFrame(overall_rows)

    by_dataset_rows = []
    for _, row in individual_by_dataset.iterrows():
        by_dataset_rows.append(
            {
                "comparison": "individual_generalization_gap",
                "dataset_role": "training_dataset",
                "dataset": row["source_dataset"],
                "metric": row["metric"],
                "mean_difference": row["mean_generalization_gap"],
            }
        )
    for _, row in balanced_by_dataset.iterrows():
        for comparison, column in (
            ("balanced_fl_target_delta", "mean_fl_target_delta"),
            (
                "balanced_fl_generalization_delta",
                "mean_fl_generalization_delta",
            ),
        ):
            by_dataset_rows.append(
                {
                    "comparison": comparison,
                    "dataset_role": "target_dataset",
                    "dataset": row["target_dataset"],
                    "metric": row["metric"],
                    "mean_difference": row[column],
                }
            )
    for _, row in low_resource.iterrows():
        for comparison, column in (
            (
                "unbalanced_low_vs_individual_1k",
                "unbalanced_vs_individual_1k",
            ),
            (
                "unbalanced_low_vs_balanced_1k",
                "unbalanced_vs_balanced_1k",
            ),
        ):
            by_dataset_rows.append(
                {
                    "comparison": comparison,
                    "dataset_role": "low_resource_dataset",
                    "dataset": row["low_resource_dataset"],
                    "metric": row["metric"],
                    "mean_difference": row[column],
                }
            )
    for _, row in high_resource_by_dataset.iterrows():
        by_dataset_rows.append(
            {
                "comparison": "unbalanced_high_resource_delta",
                "dataset_role": "high_resource_dataset",
                "dataset": row["high_resource_dataset"],
                "metric": row["metric"],
                "mean_difference": row["mean_fl_delta"],
            }
        )

    return {
        "individual_by_dataset": individual_by_dataset,
        "balanced_by_dataset": balanced_by_dataset,
        "low_resource_by_dataset": low_resource.copy(),
        "high_resource_by_dataset": high_resource_by_dataset,
        "high_resource_by_scenario": high_resource_by_scenario,
        "overall_differences": overall_differences,
        "differences_by_dataset": pd.DataFrame(by_dataset_rows),
    }


def _annotated_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    masked = np.ma.masked_invalid(matrix)
    color_map = plt.get_cmap(cmap)
    normalizer = plt.Normalize(vmin=vmin, vmax=vmax)
    image = ax.imshow(
        masked,
        cmap=color_map,
        norm=normalizer,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                text = "-"
                color = "black"
            else:
                text = f"{value:.3f}"
                red, green, blue, _ = color_map(normalizer(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                color = "black" if luminance > 0.55 else "white"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_individual_heatmaps(individual: pd.DataFrame) -> None:
    labels = [DATASET_LABELS[name] for name in DATASETS]
    for regime in REGIMES:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        subset = individual[individual["regime"] == regime]
        for ax, metric in zip(axes.flat, METRICS):
            matrix = (
                subset.pivot(
                    index="source_dataset",
                    columns="target_dataset",
                    values=metric,
                )
                .reindex(index=DATASETS, columns=DATASETS)
                .to_numpy()
            )
            _annotated_heatmap(
                ax,
                matrix,
                labels,
                labels,
                metric.title(),
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
            )
            ax.set_xlabel("Evaluation dataset")
            ax.set_ylabel("Training dataset")
        fig.suptitle(f"Individual models: {regime} training examples", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"individual_cross_domain_{regime}.png", dpi=200)
        plt.close(fig)


def _plot_individual_generalization(summary: pd.DataFrame) -> None:
    subset = summary[summary["metric"] == "f1"].copy()
    subset["label"] = subset.apply(
        lambda row: f"{DATASET_LABELS[row['source_dataset']]}\n{row['regime']}",
        axis=1,
    )
    x = np.arange(len(subset))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, subset["in_domain"], width, label="In-domain")
    ax.bar(
        x + width / 2,
        subset["cross_domain_mean"],
        width,
        label="Mean cross-domain",
    )
    ax.set_xticks(x, labels=subset["label"])
    ax.set_ylabel("Macro F1")
    ax.set_title("Individual-model in-domain performance and generalization")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "individual_generalization_f1.png", dpi=200)
    plt.close(fig)


def _plot_balanced_comparisons(comparison: pd.DataFrame) -> None:
    labels = [DATASET_LABELS[name] for name in DATASETS]
    colors = ("#35618f", "#db8b3c", "#3b9278")

    for regime in REGIMES:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        subset = comparison[comparison["regime"] == regime]
        x = np.arange(len(DATASETS))
        width = 0.25

        for ax, metric in zip(axes.flat, METRICS):
            metric_rows = subset[subset["metric"] == metric].set_index(
                "target_dataset"
            ).reindex(DATASETS)
            ax.bar(
                x - width,
                metric_rows["individual_in_domain"],
                width,
                label="Individual in-domain",
                color=colors[0],
            )
            ax.bar(
                x,
                metric_rows["foreign_individual_mean"],
                width,
                label="Mean foreign models",
                color=colors[1],
            )
            ax.bar(
                x + width,
                metric_rows["balanced_federated"],
                width,
                label="Balanced FL",
                color=colors[2],
            )
            ax.set_xticks(x, labels=labels)
            ax.set_ylim(0, 1)
            ax.set_ylabel(metric.title())
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(metric.title())

        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=3)
        fig.suptitle(f"Balanced federated comparison: {regime} per client", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(PLOTS_DIR / f"balanced_fl_comparison_{regime}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for column, metric in enumerate(METRICS):
        subset = comparison[comparison["metric"] == metric]
        for row, delta_col in enumerate(
            ("fl_target_delta", "fl_generalization_delta")
        ):
            matrix = (
                subset.pivot(
                    index="regime",
                    columns="target_dataset",
                    values=delta_col,
                )
                .reindex(index=REGIMES, columns=DATASETS)
                .to_numpy()
            )
            limit = max(0.05, float(np.nanmax(np.abs(matrix))))
            title_prefix = "FL - in-domain" if row == 0 else "FL - foreign mean"
            _annotated_heatmap(
                axes[row, column],
                matrix,
                list(REGIMES),
                labels,
                f"{title_prefix}\n{metric.title()}",
                cmap="RdBu",
                vmin=-limit,
                vmax=limit,
            )
            axes[row, column].set_xlabel("Target dataset")
            axes[row, column].set_ylabel("Data regime")
    fig.suptitle("Balanced federated learning deltas", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "balanced_fl_delta_heatmaps.png", dpi=200)
    plt.close(fig)


def _plot_balanced_f1_by_regime(
    individual: pd.DataFrame,
    balanced: pd.DataFrame,
) -> None:
    x = np.arange(len(REGIMES))
    regime_labels = ["1,000", "5,000", "10,000"]
    individual_values = []
    federated_values = []
    for regime in REGIMES:
        individual_regime = individual[individual["regime"] == regime]
        individual_values.append(
            float(
                individual_regime[
                    individual_regime["source_dataset"]
                    == individual_regime["target_dataset"]
                ]["f1"].mean()
            )
        )
        federated_values.append(
            float(balanced[balanced["regime"] == regime]["f1"].mean())
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        individual_values,
        marker="o",
        linewidth=2,
        label="Individual training",
        color="#35618f",
    )
    ax.plot(
        x,
        federated_values,
        marker="o",
        linewidth=2,
        label="Balanced FL",
        color="#3b9278",
    )
    ax.set_xticks(x, labels=regime_labels)
    ax.set_xlabel("Training examples per dataset")
    ax.set_ylabel("Average macro F1 on training dataset")
    ax.set_title("Average in-domain F1 by data regime")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "balanced_fl_target_f1_by_regime.png", dpi=200)
    plt.close(fig)


def _plot_balanced_generalization_f1_by_regime(
    individual: pd.DataFrame,
    balanced: pd.DataFrame,
) -> None:
    x = np.arange(len(REGIMES))
    regime_labels = ["1,000", "5,000", "10,000"]
    individual_values = [
        float(individual[individual["regime"] == regime]["f1"].mean())
        for regime in REGIMES
    ]
    federated_values = [
        float(balanced[balanced["regime"] == regime]["f1"].mean())
        for regime in REGIMES
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        individual_values,
        marker="o",
        linewidth=2,
        label="Individual training",
        color="#35618f",
    )
    ax.plot(
        x,
        federated_values,
        marker="o",
        linewidth=2,
        label="Balanced FL",
        color="#3b9278",
    )
    ax.set_xticks(x, labels=regime_labels)
    ax.set_xlabel("Training examples per dataset")
    ax.set_ylabel("Average macro F1 across all three datasets")
    ax.set_title("Average cross-dataset F1 by data regime")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "balanced_fl_generalization_f1_by_regime.png",
        dpi=200,
    )
    plt.close(fig)


def _plot_unbalanced_low_resource(comparison: pd.DataFrame) -> None:
    labels = [DATASET_LABELS[name] for name in DATASETS]
    x = np.arange(len(DATASETS))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, METRICS):
        rows = comparison[comparison["metric"] == metric].set_index(
            "low_resource_dataset"
        ).reindex(DATASETS)
        ax.bar(x - width, rows["individual_1k"], width, label="Individual 1k")
        ax.bar(x, rows["balanced_fl_1k"], width, label="Balanced FL 1k")
        ax.bar(x + width, rows["unbalanced_fl"], width, label="Unbalanced FL")
        ax.set_xticks(x, labels=labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.title())
        ax.set_title(metric.title())
        ax.grid(axis="y", alpha=0.25)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3)
    fig.suptitle("Benefit to the 1k low-resource client", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PLOTS_DIR / "unbalanced_low_resource_comparison.png", dpi=200)
    plt.close(fig)


def _plot_unbalanced_high_resource(impact: pd.DataFrame) -> None:
    labels = [DATASET_LABELS[name] for name in DATASETS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, metric in zip(axes.flat, METRICS):
        matrix = np.full((len(DATASETS), len(DATASETS)), np.nan)
        metric_rows = impact[impact["metric"] == metric]
        for row_idx, low_resource in enumerate(DATASETS):
            for col_idx, high_resource in enumerate(DATASETS):
                match = metric_rows[
                    (metric_rows["low_resource_dataset"] == low_resource)
                    & (metric_rows["high_resource_dataset"] == high_resource)
                ]
                if not match.empty:
                    matrix[row_idx, col_idx] = float(match.iloc[0]["fl_delta"])

        limit = max(0.05, float(np.nanmax(np.abs(matrix))))
        _annotated_heatmap(
            ax,
            matrix,
            labels,
            labels,
            metric.title(),
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_xlabel("10k high-resource dataset")
        ax.set_ylabel("1k low-resource dataset")

    fig.suptitle("Unbalanced FL impact on high-resource clients (FL - individual 10k)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "unbalanced_high_resource_delta.png", dpi=200)
    plt.close(fig)


def _plot_unbalanced_tradeoff(
    low_resource: pd.DataFrame,
    high_resource: pd.DataFrame,
) -> None:
    low_f1 = low_resource[low_resource["metric"] == "f1"].set_index(
        "low_resource_dataset"
    ).reindex(DATASETS)
    high_f1 = (
        high_resource[high_resource["metric"] == "f1"]
        .groupby("low_resource_dataset")["fl_delta"]
        .mean()
        .reindex(DATASETS)
    )

    x = np.arange(len(DATASETS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        low_f1["unbalanced_vs_individual_1k"],
        width,
        label="Gain for 1k client",
        color="#3b9278",
    )
    ax.bar(
        x + width / 2,
        high_f1,
        width,
        label="Mean change for 10k clients",
        color="#b95b58",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels=[DATASET_LABELS[name] for name in DATASETS])
    ax.set_ylabel("Macro F1 change")
    ax.set_xlabel("Low-resource client")
    ax.set_title("Unbalanced FL benefit-cost tradeoff")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "unbalanced_f1_tradeoff.png", dpi=200)
    plt.close(fig)


def _plot_overall_metric_differences(overall: pd.DataFrame) -> None:
    comparison_columns = [
        "individual_generalization_gap",
        "balanced_fl_target_delta",
        "balanced_fl_generalization_delta",
        "unbalanced_low_vs_individual_1k",
        "unbalanced_low_vs_balanced_1k",
        "unbalanced_high_resource_delta",
    ]
    labels = [
        "Individual\ncross - in",
        "Balanced FL\n- in-domain",
        "Balanced FL\n- foreign",
        "Unbalanced low\n- individual 1k",
        "Unbalanced low\n- balanced 1k",
        "Unbalanced high\n- individual 10k",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, metric in zip(axes.flat, METRICS):
        row = overall[overall["metric"] == metric].iloc[0]
        values = np.array([float(row[column]) for column in comparison_columns])
        colors = ["#3b9278" if value >= 0 else "#b95b58" for value in values]
        bars = ax.bar(np.arange(len(values)), values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(values)), labels=labels, fontsize=8)
        ax.set_ylabel("Mean metric difference")
        ax.set_title(metric.title())
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            offset = 3 if value >= 0 else -12
            ax.annotate(
                f"{value:+.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle("Average differences across comparison scenarios", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "overall_metric_differences.png", dpi=200)
    plt.close(fig)


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str] | None = None,
) -> str:
    headers = headers or columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    individual: pd.DataFrame,
    generalization: pd.DataFrame,
    balanced: pd.DataFrame,
    low_resource: pd.DataFrame,
    high_resource: pd.DataFrame,
    aggregates: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# ECG Results Analysis",
        "",
        "All reported values are macro averages across the six classification targets.",
        "Cross-domain evaluations use thresholds selected on the target validation set, "
        "so they represent target-calibrated transfer rather than calibration-free transfer.",
        "",
        "## Individual and Cross-Domain Models",
        "",
    ]

    display_columns = [
        "source_dataset",
        "target_dataset",
        "evaluation_type",
        *METRICS,
    ]
    display_headers = [
        "Training dataset",
        "Evaluation dataset",
        "Evaluation",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]
    for regime in REGIMES:
        table = individual[individual["regime"] == regime].copy()
        table["source_dataset"] = table["source_dataset"].map(DATASET_LABELS)
        table["target_dataset"] = table["target_dataset"].map(DATASET_LABELS)
        lines.extend(
            [
                f"### {regime} training examples",
                "",
                _markdown_table(table, display_columns, display_headers),
                "",
            ]
        )

    f1_generalization = generalization[generalization["metric"] == "f1"].copy()
    f1_generalization["source_dataset"] = f1_generalization["source_dataset"].map(
        DATASET_LABELS
    )
    individual_average = aggregates["individual_by_dataset"].copy()
    individual_average["source_dataset"] = individual_average["source_dataset"].map(
        DATASET_LABELS
    )
    lines.extend(
        [
            "### Individual-model F1 generalization",
            "",
            _markdown_table(
                f1_generalization,
                [
                    "source_dataset",
                    "regime",
                    "in_domain",
                    "cross_domain_mean",
                    "generalization_gap",
                ],
                [
                    "Training dataset",
                    "Regime",
                    "In-domain F1",
                    "Mean cross-domain F1",
                    "Generalization gap",
                ],
            ),
            "",
            "### Average across data regimes by training dataset",
            "",
            _markdown_table(
                individual_average,
                [
                    "source_dataset",
                    "metric",
                    "mean_in_domain",
                    "mean_cross_domain",
                    "mean_generalization_gap",
                ],
                [
                    "Training dataset",
                    "Metric",
                    "Mean in-domain",
                    "Mean cross-domain",
                    "Mean generalization gap",
                ],
            ),
            "",
            "## Balanced Federated Learning",
            "",
        ]
    )

    for metric in METRICS:
        table = balanced[balanced["metric"] == metric].copy()
        table["target_dataset"] = table["target_dataset"].map(DATASET_LABELS)
        lines.extend(
            [
                f"### {metric.title()}",
                "",
                _markdown_table(
                    table,
                    [
                        "regime",
                        "target_dataset",
                        "individual_in_domain",
                        "foreign_individual_mean",
                        "balanced_federated",
                        "fl_target_delta",
                        "fl_generalization_delta",
                    ],
                    [
                        "Regime",
                        "Target",
                        "Individual in-domain",
                        "Foreign-model mean",
                        "Balanced FL",
                        "FL target delta",
                        "FL generalization delta",
                    ],
                ),
                "",
            ]
        )

    balanced_average = aggregates["balanced_by_dataset"].copy()
    balanced_average["target_dataset"] = balanced_average["target_dataset"].map(
        DATASET_LABELS
    )
    lines.extend(
        [
            "### Average across data regimes by target dataset",
            "",
            _markdown_table(
                balanced_average,
                [
                    "target_dataset",
                    "metric",
                    "mean_individual_in_domain",
                    "mean_foreign_individual",
                    "mean_balanced_federated",
                    "mean_fl_target_delta",
                    "mean_fl_generalization_delta",
                ],
                [
                    "Target",
                    "Metric",
                    "Mean individual",
                    "Mean foreign",
                    "Mean balanced FL",
                    "Mean target delta",
                    "Mean generalization delta",
                ],
            ),
            "",
        ]
    )

    lines.extend(["## Unbalanced Federated Learning", "", "### Low-resource client", ""])
    low_table = low_resource.copy()
    low_table["low_resource_dataset"] = low_table["low_resource_dataset"].map(
        DATASET_LABELS
    )
    lines.extend(
        [
            _markdown_table(
                low_table,
                [
                    "low_resource_dataset",
                    "metric",
                    "individual_1k",
                    "balanced_fl_1k",
                    "unbalanced_fl",
                    "unbalanced_vs_individual_1k",
                    "unbalanced_vs_balanced_1k",
                ],
                [
                    "1k client",
                    "Metric",
                    "Individual 1k",
                    "Balanced FL 1k",
                    "Unbalanced FL",
                    "vs individual",
                    "vs balanced FL",
                ],
            ),
            "",
            "### High-resource clients",
            "",
        ]
    )
    high_table = high_resource.copy()
    high_table["low_resource_dataset"] = high_table["low_resource_dataset"].map(
        DATASET_LABELS
    )
    high_table["high_resource_dataset"] = high_table["high_resource_dataset"].map(
        DATASET_LABELS
    )
    lines.append(
        _markdown_table(
            high_table,
            [
                "low_resource_dataset",
                "high_resource_dataset",
                "metric",
                "individual_10k",
                "unbalanced_fl",
                "fl_delta",
            ],
            [
                "1k client",
                "10k client",
                "Metric",
                "Individual 10k",
                "Unbalanced FL",
                "FL delta",
            ],
        )
    )
    lines.append("")

    high_average = aggregates["high_resource_by_dataset"].copy()
    high_average["high_resource_dataset"] = high_average[
        "high_resource_dataset"
    ].map(DATASET_LABELS)
    high_scenario_average = aggregates["high_resource_by_scenario"].copy()
    high_scenario_average["low_resource_dataset"] = high_scenario_average[
        "low_resource_dataset"
    ].map(DATASET_LABELS)
    lines.extend(
        [
            "### Average impact by high-resource dataset",
            "",
            _markdown_table(
                high_average,
                [
                    "high_resource_dataset",
                    "metric",
                    "mean_individual_10k",
                    "mean_unbalanced_fl",
                    "mean_fl_delta",
                ],
                [
                    "10k dataset",
                    "Metric",
                    "Mean individual 10k",
                    "Mean unbalanced FL",
                    "Mean FL delta",
                ],
            ),
            "",
            "### Average impact on the two high-resource clients per scenario",
            "",
            _markdown_table(
                high_scenario_average,
                [
                    "low_resource_dataset",
                    "metric",
                    "mean_individual_10k",
                    "mean_unbalanced_fl",
                    "mean_fl_delta",
                ],
                [
                    "1k dataset",
                    "Metric",
                    "Mean individual 10k",
                    "Mean unbalanced FL",
                    "Mean FL delta",
                ],
            ),
            "",
            "## Final Aggregated Metric Differences",
            "",
            "Positive values favor the first method named in each comparison; "
            "negative values indicate a performance reduction.",
            "",
            _markdown_table(
                aggregates["overall_differences"],
                [
                    "metric",
                    "individual_generalization_gap",
                    "balanced_fl_target_delta",
                    "balanced_fl_generalization_delta",
                    "unbalanced_low_vs_individual_1k",
                    "unbalanced_low_vs_balanced_1k",
                    "unbalanced_high_resource_delta",
                ],
                [
                    "Metric",
                    "Individual cross - in",
                    "Balanced FL - in-domain",
                    "Balanced FL - foreign",
                    "Unbalanced low - individual",
                    "Unbalanced low - balanced",
                    "Unbalanced high - individual",
                ],
            ),
            "",
        ]
    )

    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))


def _write_summary(
    aggregates: dict[str, pd.DataFrame],
) -> None:
    summary = {
        "metric_definition": "macro average across six ECG abnormalities",
        "cross_domain_calibration": (
            "Thresholds are selected on each target dataset's validation set."
        ),
        "overall_metric_differences": json.loads(
            aggregates["overall_differences"].to_json(orient="records")
        ),
        "individual_by_dataset": json.loads(
            aggregates["individual_by_dataset"].to_json(orient="records")
        ),
        "balanced_federated_by_dataset": json.loads(
            aggregates["balanced_by_dataset"].to_json(orient="records")
        ),
        "unbalanced_low_resource_by_dataset": json.loads(
            aggregates["low_resource_by_dataset"].to_json(orient="records")
        ),
        "unbalanced_high_resource_by_dataset": json.loads(
            aggregates["high_resource_by_dataset"].to_json(orient="records")
        ),
        "unbalanced_high_resource_by_scenario": json.loads(
            aggregates["high_resource_by_scenario"].to_json(orient="records")
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    evaluations = _collect_evaluations()
    individual = evaluations[evaluations["model_type"] == "individual"].copy()
    balanced_models = evaluations[
        evaluations["model_type"] == "balanced_federated"
    ].copy()
    unbalanced_models = evaluations[
        evaluations["model_type"] == "unbalanced_federated"
    ].copy()

    generalization = _individual_generalization_summary(individual)
    balanced_comparison = _balanced_comparison(individual, balanced_models)
    low_resource_comparison = _unbalanced_low_resource_comparison(
        individual,
        balanced_models,
        unbalanced_models,
    )
    high_resource_impact = _unbalanced_high_resource_impact(
        individual,
        unbalanced_models,
    )
    aggregates = _aggregate_statistics(
        generalization,
        balanced_comparison,
        low_resource_comparison,
        high_resource_impact,
    )

    all_results = evaluations.melt(
        id_vars=[
            "model_type",
            "setup",
            "regime",
            "source_dataset",
            "target_dataset",
            "evaluation_type",
            "low_resource_dataset",
        ],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )

    all_results.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    individual.to_csv(OUTPUT_DIR / "individual_cross_domain.csv", index=False)
    generalization.to_csv(
        OUTPUT_DIR / "individual_generalization_summary.csv",
        index=False,
    )
    balanced_comparison.to_csv(
        OUTPUT_DIR / "balanced_fl_comparison.csv",
        index=False,
    )
    low_resource_comparison.to_csv(
        OUTPUT_DIR / "unbalanced_low_resource_comparison.csv",
        index=False,
    )
    high_resource_impact.to_csv(
        OUTPUT_DIR / "unbalanced_high_resource_impact.csv",
        index=False,
    )
    aggregates["individual_by_dataset"].to_csv(
        OUTPUT_DIR / "individual_generalization_by_dataset.csv",
        index=False,
    )
    aggregates["balanced_by_dataset"].to_csv(
        OUTPUT_DIR / "balanced_fl_by_dataset.csv",
        index=False,
    )
    aggregates["high_resource_by_dataset"].to_csv(
        OUTPUT_DIR / "unbalanced_high_resource_by_dataset.csv",
        index=False,
    )
    aggregates["high_resource_by_scenario"].to_csv(
        OUTPUT_DIR / "unbalanced_high_resource_by_scenario.csv",
        index=False,
    )
    aggregates["overall_differences"].to_csv(
        OUTPUT_DIR / "aggregate_metric_differences.csv",
        index=False,
    )
    aggregates["differences_by_dataset"].to_csv(
        OUTPUT_DIR / "aggregate_metric_differences_by_dataset.csv",
        index=False,
    )

    _plot_individual_heatmaps(individual)
    _plot_individual_generalization(generalization)
    _plot_balanced_comparisons(balanced_comparison)
    _plot_balanced_f1_by_regime(individual, balanced_models)
    _plot_balanced_generalization_f1_by_regime(individual, balanced_models)
    _plot_unbalanced_low_resource(low_resource_comparison)
    _plot_unbalanced_high_resource(high_resource_impact)
    _plot_unbalanced_tradeoff(low_resource_comparison, high_resource_impact)
    _plot_overall_metric_differences(aggregates["overall_differences"])

    _write_report(
        individual,
        generalization,
        balanced_comparison,
        low_resource_comparison,
        high_resource_impact,
        aggregates,
    )
    _write_summary(aggregates)

    print(f"Wrote results analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
