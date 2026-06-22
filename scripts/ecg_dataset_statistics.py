from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import chi2_contingency, ks_2samp

from tasks.ecg_supervised_task import ECGMultilabelClassification
from utils import get_dataset_paths


DATASETS = ("code15", "ptbxl", "chapman")
DATASET_LABELS = {
    "code15": "CODE-15",
    "ptbxl": "PTB-XL",
    "chapman": "Chapman-Shaoxing",
}
TARGET_COLUMNS = tuple(ECGMultilabelClassification.TARGET_CLF_COLUMNS)
OUTPUT_DIR = Path("experiments/dataset_statistics")

AGE_COLUMNS = ("age", "Age", "patient_age", "PatientAge")
SEX_COLUMNS = ("sex", "Sex", "gender", "Gender", "patient_sex", "PatientSex")
SPLIT_COLUMNS = ("group", "split", "Split", "strat_fold", "fold")


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_sex(value) -> str:
    if pd.isna(value):
        return "missing"

    text = str(value).strip().lower()
    if text in {"m", "male", "man", "1"}:
        return "male"
    if text in {"f", "female", "woman", "0"}:
        return "female"
    if text in {"", "nan", "none", "unknown"}:
        return "missing"
    return text


def _read_dataset(name: str) -> pd.DataFrame:
    _, csv_path = get_dataset_paths(name)
    df = pd.read_csv(csv_path)
    missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"{name} is missing target columns: {missing_targets}")
    return df


def _target_frame(df: pd.DataFrame) -> pd.DataFrame:
    targets = df.loc[:, TARGET_COLUMNS].copy()
    return targets.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)


def _dataset_summary(name: str, df: pd.DataFrame, targets: pd.DataFrame) -> dict:
    age_col = _first_existing_column(df, AGE_COLUMNS)
    sex_col = _first_existing_column(df, SEX_COLUMNS)
    split_col = _first_existing_column(df, SPLIT_COLUMNS)

    return {
        "dataset": name,
        "n_records": int(len(df)),
        "n_positive_labels": int(targets.sum(axis=1).sum()),
        "mean_positive_labels_per_record": float(targets.sum(axis=1).mean()),
        "records_with_any_target": int((targets.sum(axis=1) > 0).sum()),
        "records_with_any_target_pct": float((targets.sum(axis=1) > 0).mean()),
        "age_column": age_col,
        "sex_column": sex_col,
        "split_column": split_col,
    }


def _class_prevalence(name: str, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(targets)
    for col in TARGET_COLUMNS:
        positives = int(targets[col].sum())
        rows.append(
            {
                "dataset": name,
                "class": col,
                "n": n,
                "positives": positives,
                "prevalence": positives / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _label_burden(name: str, targets: pd.DataFrame) -> pd.DataFrame:
    counts = targets.sum(axis=1).value_counts().sort_index()
    total = len(targets)
    return pd.DataFrame(
        {
            "dataset": name,
            "n_positive_labels": counts.index.astype(int),
            "records": counts.values.astype(int),
            "fraction": counts.values / total if total else np.nan,
        }
    )


def _cooccurrence(name: str, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(targets)
    for class_a in TARGET_COLUMNS:
        for class_b in TARGET_COLUMNS:
            count = int(((targets[class_a] == 1) & (targets[class_b] == 1)).sum())
            rows.append(
                {
                    "dataset": name,
                    "class_a": class_a,
                    "class_b": class_b,
                    "count": count,
                    "prevalence": count / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _age_summary(name: str, df: pd.DataFrame) -> tuple[dict | None, pd.Series | None]:
    age_col = _first_existing_column(df, AGE_COLUMNS)
    if age_col is None:
        return None, None

    age = pd.to_numeric(df[age_col], errors="coerce")
    clean = age.dropna()
    if clean.empty:
        return None, age

    return (
        {
            "dataset": name,
            "column": age_col,
            "n": int(clean.shape[0]),
            "missing": int(age.isna().sum()),
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "median": float(clean.median()),
            "q1": float(clean.quantile(0.25)),
            "q3": float(clean.quantile(0.75)),
            "min": float(clean.min()),
            "max": float(clean.max()),
        },
        age,
    )


def _sex_distribution(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.Series | None]:
    sex_col = _first_existing_column(df, SEX_COLUMNS)
    if sex_col is None:
        return None, None

    sex = df[sex_col].map(_normalize_sex)
    counts = sex.value_counts(dropna=False).sort_index()
    total = counts.sum()
    table = pd.DataFrame(
        {
            "dataset": name,
            "column": sex_col,
            "sex": counts.index,
            "count": counts.values.astype(int),
            "fraction": counts.values / total if total else np.nan,
        }
    )
    return table, sex


def _split_distribution(name: str, df: pd.DataFrame) -> pd.DataFrame | None:
    split_col = _first_existing_column(df, SPLIT_COLUMNS)
    if split_col is None:
        return None

    counts = df[split_col].fillna("missing").astype(str).value_counts().sort_index()
    total = counts.sum()
    return pd.DataFrame(
        {
            "dataset": name,
            "column": split_col,
            "split": counts.index,
            "count": counts.values.astype(int),
            "fraction": counts.values / total if total else np.nan,
        }
    )


def _pairwise_prevalence_tests(prevalence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_a, dataset_b in combinations(DATASETS, 2):
        for cls in TARGET_COLUMNS:
            row_a = prevalence[
                (prevalence["dataset"] == dataset_a) & (prevalence["class"] == cls)
            ].iloc[0]
            row_b = prevalence[
                (prevalence["dataset"] == dataset_b) & (prevalence["class"] == cls)
            ].iloc[0]

            table = np.array(
                [
                    [row_a["positives"], row_a["n"] - row_a["positives"]],
                    [row_b["positives"], row_b["n"] - row_b["positives"]],
                ]
            )
            _, p_value, _, _ = chi2_contingency(table)

            rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "class": cls,
                    "prevalence_a": row_a["prevalence"],
                    "prevalence_b": row_b["prevalence"],
                    "absolute_difference": abs(row_a["prevalence"] - row_b["prevalence"]),
                    "chi2_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def _pairwise_demographic_tests(
    ages: dict[str, pd.Series],
    sexes: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for dataset_a, dataset_b in combinations(DATASETS, 2):
        if dataset_a in ages and dataset_b in ages:
            age_a = ages[dataset_a].dropna()
            age_b = ages[dataset_b].dropna()
            if not age_a.empty and not age_b.empty:
                stat, p_value = ks_2samp(age_a, age_b)
                rows.append(
                    {
                        "dataset_a": dataset_a,
                        "dataset_b": dataset_b,
                        "variable": "age",
                        "test": "ks_2samp",
                        "statistic": stat,
                        "p_value": p_value,
                    }
                )

        if dataset_a in sexes and dataset_b in sexes:
            all_levels = sorted(set(sexes[dataset_a]) | set(sexes[dataset_b]))
            table = np.array(
                [
                    [(sexes[dataset_a] == level).sum() for level in all_levels],
                    [(sexes[dataset_b] == level).sum() for level in all_levels],
                ]
            )
            _, p_value, _, _ = chi2_contingency(table)
            rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "variable": "sex",
                    "test": "chi2",
                    "statistic": np.nan,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def _save_class_prevalence_plot(prevalence: pd.DataFrame, out_dir: Path) -> None:
    pivot = prevalence.pivot(index="class", columns="dataset", values="prevalence").loc[
        list(TARGET_COLUMNS)
    ]
    ax = pivot.plot(kind="bar", figsize=(11, 5), width=0.8)
    ax.set_ylabel("Prevalence")
    ax.set_xlabel("Abnormality")
    ax.set_title("Abnormality prevalence by dataset")
    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "class_prevalence.png", dpi=200)
    plt.close()


def _save_label_burden_plot(label_burden: pd.DataFrame, out_dir: Path) -> None:
    pivot = label_burden.pivot(
        index="n_positive_labels", columns="dataset", values="fraction"
    ).fillna(0)
    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.8)
    ax.set_ylabel("Fraction of records")
    ax.set_xlabel("Number of positive labels")
    ax.set_title("Multilabel burden by dataset")
    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "label_burden.png", dpi=200)
    plt.close()


def _save_age_plot(ages: dict[str, pd.Series], out_dir: Path) -> None:
    if not ages:
        return

    plt.figure(figsize=(10, 5))
    for dataset, age in ages.items():
        clean = age.dropna()
        if clean.empty:
            continue
        plt.hist(clean, bins=30, alpha=0.45, density=True, label=dataset)
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.title("Age distribution by dataset")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "age_distribution.png", dpi=200)
    plt.close()

    labels = []
    data = []
    for dataset, age in ages.items():
        clean = age.dropna()
        if not clean.empty:
            labels.append(dataset)
            data.append(clean)
    if data:
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Age")
        plt.title("Age distribution summary")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "age_boxplot.png", dpi=200)
        plt.close()


def _save_sex_plot(sex_distribution: pd.DataFrame, out_dir: Path) -> None:
    if sex_distribution.empty:
        return

    pivot = sex_distribution.pivot(index="dataset", columns="sex", values="fraction").fillna(0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(8, 5), width=0.75)
    ax.set_ylabel("Fraction of records")
    ax.set_xlabel("Dataset")
    ax.set_title("Sex distribution by dataset")
    ax.legend(title="Sex", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "sex_distribution.png", dpi=200)
    plt.close()


def _save_cooccurrence_plots(cooccurrence: pd.DataFrame, out_dir: Path) -> None:
    for dataset in DATASETS:
        subset = cooccurrence[cooccurrence["dataset"] == dataset]
        matrix = subset.pivot(
            index="class_a", columns="class_b", values="prevalence"
        ).loc[list(TARGET_COLUMNS), list(TARGET_COLUMNS)]

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix.values, cmap="magma")
        ax.set_xticks(np.arange(len(TARGET_COLUMNS)), labels=TARGET_COLUMNS, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(TARGET_COLUMNS)), labels=TARGET_COLUMNS)
        ax.set_title(f"{dataset}: target co-occurrence prevalence")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_dir / f"cooccurrence_{dataset}.png", dpi=200)
        plt.close()


def _format_value(value, *, percent: bool = False) -> str:
    if pd.isna(value):
        return "NA"

    if percent:
        return f"{100 * float(value):.2f}%"

    if isinstance(value, (float, np.floating)):
        value = float(value)
        if 0 < abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.3f}"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    return str(value)


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str] | None = None,
    *,
    percent_columns: set[str] | None = None,
    max_rows: int | None = None,
) -> str:
    if frame.empty:
        return "_No data available._"

    headers = headers or columns
    percent_columns = percent_columns or set()
    table = frame.loc[:, columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [
            _format_value(row[column], percent=column in percent_columns)
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _with_dataset_labels(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(DATASET_LABELS).fillna(out[column])
    return out


def _prevalence_extremes(prevalence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        subset = prevalence[prevalence["dataset"] == dataset]
        most = subset.sort_values("prevalence", ascending=False).iloc[0]
        least = subset.sort_values("prevalence", ascending=True).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "most_prevalent_class": most["class"],
                "most_prevalent_pct": most["prevalence"],
                "least_prevalent_class": least["class"],
                "least_prevalent_pct": least["prevalence"],
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    *,
    dataset_summary: pd.DataFrame,
    prevalence: pd.DataFrame,
    label_burden: pd.DataFrame,
    age_summary: pd.DataFrame,
    sex_distribution: pd.DataFrame,
    split_distribution: pd.DataFrame,
    prevalence_tests: pd.DataFrame,
    demographic_tests: pd.DataFrame,
    out_dir: Path,
) -> None:
    dataset_table = _with_dataset_labels(dataset_summary, ("dataset",))
    prevalence_table = _with_dataset_labels(prevalence, ("dataset",))
    label_burden_table = _with_dataset_labels(label_burden, ("dataset",))
    age_table = _with_dataset_labels(age_summary, ("dataset",))
    sex_table = _with_dataset_labels(sex_distribution, ("dataset",))
    split_table = _with_dataset_labels(split_distribution, ("dataset",))
    prevalence_extremes = _with_dataset_labels(
        _prevalence_extremes(prevalence),
        ("dataset",),
    )

    largest_prevalence_differences = _with_dataset_labels(
        prevalence_tests.sort_values("absolute_difference", ascending=False),
        ("dataset_a", "dataset_b"),
    )
    demographic_table = _with_dataset_labels(
        demographic_tests.sort_values("p_value") if not demographic_tests.empty else demographic_tests,
        ("dataset_a", "dataset_b"),
    )

    lines = [
        "# Dataset Statistics Report",
        "",
        "This report summarizes distributional differences across the three ECG datasets used in the project.",
        "All target statistics are computed from the harmonized multilabel classification columns:",
        "`" + "`, `".join(TARGET_COLUMNS) + "`.",
        "",
        "The goal is to quantify heterogeneity that may affect supervised training, cross-domain transfer, "
        "and federated learning behavior.",
        "",
        "## Dataset Overview",
        "",
        _markdown_table(
            dataset_table,
            [
                "dataset",
                "n_records",
                "n_positive_labels",
                "mean_positive_labels_per_record",
                "records_with_any_target",
                "records_with_any_target_pct",
                "age_column",
                "sex_column",
                "split_column",
            ],
            [
                "Dataset",
                "Records",
                "Positive labels",
                "Mean labels/record",
                "Records with any target",
                "Any target %",
                "Age column",
                "Sex column",
                "Split column",
            ],
            percent_columns={"records_with_any_target_pct"},
        ),
        "",
        "## Abnormality Prevalence",
        "",
        "The table below reports the prevalence of each target abnormality in each dataset.",
        "",
        _markdown_table(
            prevalence_table,
            ["dataset", "class", "n", "positives", "prevalence"],
            ["Dataset", "Class", "Records", "Positive records", "Prevalence"],
            percent_columns={"prevalence"},
        ),
        "",
        "![Abnormality prevalence](class_prevalence.png)",
        "",
        "### Most and Least Prevalent Targets",
        "",
        _markdown_table(
            prevalence_extremes,
            [
                "dataset",
                "most_prevalent_class",
                "most_prevalent_pct",
                "least_prevalent_class",
                "least_prevalent_pct",
            ],
            [
                "Dataset",
                "Most prevalent",
                "Most prevalent %",
                "Least prevalent",
                "Least prevalent %",
            ],
            percent_columns={"most_prevalent_pct", "least_prevalent_pct"},
        ),
        "",
        "### Largest Pairwise Prevalence Differences",
        "",
        "These rows highlight the target labels with the largest absolute prevalence differences between datasets.",
        "",
        _markdown_table(
            largest_prevalence_differences,
            [
                "dataset_a",
                "dataset_b",
                "class",
                "prevalence_a",
                "prevalence_b",
                "absolute_difference",
                "chi2_p_value",
            ],
            [
                "Dataset A",
                "Dataset B",
                "Class",
                "Prevalence A",
                "Prevalence B",
                "Absolute difference",
                "Chi-square p",
            ],
            percent_columns={
                "prevalence_a",
                "prevalence_b",
                "absolute_difference",
            },
            max_rows=12,
        ),
        "",
        "## Multilabel Burden",
        "",
        "This table shows how many target abnormalities are positive per record. "
        "Large differences here can change the difficulty of threshold selection and the balance between precision and recall.",
        "",
        _markdown_table(
            label_burden_table,
            ["dataset", "n_positive_labels", "records", "fraction"],
            ["Dataset", "Positive labels/record", "Records", "Fraction"],
            percent_columns={"fraction"},
        ),
        "",
        "![Multilabel burden](label_burden.png)",
        "",
        "## Co-occurrence Structure",
        "",
        "The co-occurrence plots show how often target pairs are positive in the same ECG. "
        "Different co-occurrence structures can make a shared model learn different label dependencies in each dataset.",
        "",
    ]

    for dataset in DATASETS:
        lines.extend(
            [
                f"![{DATASET_LABELS[dataset]} co-occurrence](cooccurrence_{dataset}.png)",
                "",
            ]
        )

    lines.extend(["## Age Distribution", ""])
    if age_table.empty:
        lines.extend(
            [
                "No usable age column was detected in the CSV metadata for these datasets.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                _markdown_table(
                    age_table,
                    [
                        "dataset",
                        "column",
                        "n",
                        "missing",
                        "mean",
                        "std",
                        "median",
                        "q1",
                        "q3",
                        "min",
                        "max",
                    ],
                    [
                        "Dataset",
                        "Column",
                        "N",
                        "Missing",
                        "Mean",
                        "Std",
                        "Median",
                        "Q1",
                        "Q3",
                        "Min",
                        "Max",
                    ],
                ),
                "",
                "![Age distribution](age_distribution.png)",
                "",
                "![Age boxplot](age_boxplot.png)",
                "",
            ]
        )

    lines.extend(["## Sex Distribution", ""])
    if sex_table.empty:
        lines.extend(
            [
                "No usable sex/gender column was detected in the CSV metadata for these datasets.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                _markdown_table(
                    sex_table,
                    ["dataset", "column", "sex", "count", "fraction"],
                    ["Dataset", "Column", "Sex", "Count", "Fraction"],
                    percent_columns={"fraction"},
                ),
                "",
                "![Sex distribution](sex_distribution.png)",
                "",
            ]
        )

    lines.extend(["## Split or Fold Distribution", ""])
    if split_table.empty:
        lines.extend(
            [
                "No split/fold column was detected in the CSV metadata.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                _markdown_table(
                    split_table,
                    ["dataset", "column", "split", "count", "fraction"],
                    ["Dataset", "Column", "Split", "Count", "Fraction"],
                    percent_columns={"fraction"},
                ),
                "",
            ]
        )

    lines.extend(["## Pairwise Demographic Tests", ""])
    if demographic_table.empty:
        lines.extend(
            [
                "No pairwise demographic tests were available, usually because age or sex columns were absent.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Age differences are tested with a two-sample Kolmogorov-Smirnov test. "
                "Sex distribution differences are tested with a chi-square test.",
                "",
                _markdown_table(
                    demographic_table,
                    [
                        "dataset_a",
                        "dataset_b",
                        "variable",
                        "test",
                        "statistic",
                        "p_value",
                    ],
                    [
                        "Dataset A",
                        "Dataset B",
                        "Variable",
                        "Test",
                        "Statistic",
                        "p-value",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Generated Files",
            "",
            "The script also writes full CSV tables for downstream analysis:",
            "",
        ]
    )
    for path in sorted(out_dir.glob("*.csv")):
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "And these plot files:",
            "",
        ]
    )
    for path in sorted(out_dir.glob("*.png")):
        lines.append(f"- `{path.name}`")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    load_dotenv()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []
    prevalence_tables = []
    label_burden_tables = []
    cooccurrence_tables = []
    age_summaries = []
    sex_tables = []
    split_tables = []
    ages = {}
    sexes = {}

    for dataset in DATASETS:
        df = _read_dataset(dataset)
        targets = _target_frame(df)

        dataset_summaries.append(_dataset_summary(dataset, df, targets))
        prevalence_tables.append(_class_prevalence(dataset, targets))
        label_burden_tables.append(_label_burden(dataset, targets))
        cooccurrence_tables.append(_cooccurrence(dataset, targets))

        age_summary, age = _age_summary(dataset, df)
        if age_summary is not None:
            age_summaries.append(age_summary)
        if age is not None:
            ages[dataset] = age

        sex_table, sex = _sex_distribution(dataset, df)
        if sex_table is not None:
            sex_tables.append(sex_table)
        if sex is not None:
            sexes[dataset] = sex

        split_table = _split_distribution(dataset, df)
        if split_table is not None:
            split_tables.append(split_table)

    dataset_summary = pd.DataFrame(dataset_summaries)
    prevalence = pd.concat(prevalence_tables, ignore_index=True)
    label_burden = pd.concat(label_burden_tables, ignore_index=True)
    cooccurrence = pd.concat(cooccurrence_tables, ignore_index=True)
    age_summary = pd.DataFrame(age_summaries)
    sex_distribution = (
        pd.concat(sex_tables, ignore_index=True) if sex_tables else pd.DataFrame()
    )
    split_distribution = (
        pd.concat(split_tables, ignore_index=True) if split_tables else pd.DataFrame()
    )
    prevalence_tests = _pairwise_prevalence_tests(prevalence)
    demographic_tests = _pairwise_demographic_tests(ages, sexes)

    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    prevalence.to_csv(OUTPUT_DIR / "class_prevalence.csv", index=False)
    label_burden.to_csv(OUTPUT_DIR / "label_burden.csv", index=False)
    cooccurrence.to_csv(OUTPUT_DIR / "class_cooccurrence.csv", index=False)
    age_summary.to_csv(OUTPUT_DIR / "age_summary.csv", index=False)
    sex_distribution.to_csv(OUTPUT_DIR / "sex_distribution.csv", index=False)
    split_distribution.to_csv(OUTPUT_DIR / "split_distribution.csv", index=False)
    prevalence_tests.to_csv(OUTPUT_DIR / "pairwise_prevalence_tests.csv", index=False)
    demographic_tests.to_csv(OUTPUT_DIR / "pairwise_demographic_tests.csv", index=False)

    _save_class_prevalence_plot(prevalence, OUTPUT_DIR)
    _save_label_burden_plot(label_burden, OUTPUT_DIR)
    _save_age_plot(ages, OUTPUT_DIR)
    _save_sex_plot(sex_distribution, OUTPUT_DIR)
    _save_cooccurrence_plots(cooccurrence, OUTPUT_DIR)

    _write_report(
        dataset_summary=dataset_summary,
        prevalence=prevalence,
        label_burden=label_burden,
        age_summary=age_summary,
        sex_distribution=sex_distribution,
        split_distribution=split_distribution,
        prevalence_tests=prevalence_tests,
        demographic_tests=demographic_tests,
        out_dir=OUTPUT_DIR,
    )

    summary = {
        "datasets": DATASETS,
        "target_columns": TARGET_COLUMNS,
        "output_dir": str(OUTPUT_DIR),
        "tables": sorted(path.name for path in OUTPUT_DIR.glob("*.csv")),
        "plots": sorted(path.name for path in OUTPUT_DIR.glob("*.png")),
        "report": "report.md",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote dataset statistics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
