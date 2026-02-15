"""EDA generation for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .preprocessing import PreprocessOutputs


@dataclass(frozen=True)
class EDAOutputs:
    class_distribution_table: Path
    length_summary_table: Path
    class_distribution_figure: Path
    length_distribution_figure: Path
    key_metrics: dict[str, float | int]


def _load_split_frames(preprocessed: PreprocessOutputs) -> pd.DataFrame:
    full_df = pd.read_csv(preprocessed.full_dataset_path)
    full_df["split"] = "full"

    train_df = pd.read_csv(preprocessed.train_path)
    train_df["split"] = "train"

    dev_df = pd.read_csv(preprocessed.dev_path)
    dev_df["split"] = "dev"

    combined = pd.concat([full_df, train_df, dev_df], ignore_index=True)
    combined["label_name"] = combined["label_binary"].map({0: "No PCL", 1: "PCL"})
    return combined


def _class_distribution(combined: pd.DataFrame, table_path: Path, figure_path: Path) -> dict[str, float | int]:
    counts = (
        combined.groupby(["split", "label_binary", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    split_totals = counts.groupby("split", as_index=False)["count"].sum().rename(
        columns={"count": "split_total"}
    )
    table = counts.merge(split_totals, on="split", how="left")
    table["percentage"] = (table["count"] / table["split_total"] * 100.0).round(3)
    table = table.sort_values(["split", "label_binary"]).reset_index(drop=True)
    table.to_csv(table_path, index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    chart = sns.barplot(data=table, x="split", y="count", hue="label_name")
    chart.set_title("Class Distribution by Split")
    chart.set_xlabel("Split")
    chart.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()

    metric_table = table[table["label_binary"] == 1].set_index("split")

    def _metric(split: str, column: str, default: float = 0.0) -> float:
        if split not in metric_table.index:
            return default
        return float(metric_table.loc[split, column])

    metrics = {
        "train_positive_count": int(_metric("train", "count")),
        "train_positive_pct": _metric("train", "percentage"),
        "dev_positive_count": int(_metric("dev", "count")),
        "dev_positive_pct": _metric("dev", "percentage"),
    }
    return metrics


def _length_analysis(
    combined: pd.DataFrame, table_path: Path, figure_path: Path
) -> dict[str, float | int]:
    stats = (
        combined.groupby(["split", "label_binary", "label_name"])["token_count"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            min="min",
            max="max",
            p90=lambda x: x.quantile(0.90),
            p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    stats = stats.round(3).sort_values(["split", "label_binary"]).reset_index(drop=True)
    stats.to_csv(table_path, index=False)

    plot_df = combined[combined["split"].isin(["train", "dev"])].copy()
    sns.set_theme(style="whitegrid")
    g = sns.displot(
        data=plot_df,
        x="token_count",
        hue="label_name",
        col="split",
        kind="hist",
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Token Count", "Density")
    g.figure.suptitle("Token-Length Distribution by Label and Split", y=1.05)
    g.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    train_stats = stats[stats["split"] == "train"]
    train_pos = train_stats[train_stats["label_binary"] == 1]
    train_neg = train_stats[train_stats["label_binary"] == 0]

    train_p95 = float(train_stats["p95"].max()) if not train_stats.empty else 0.0
    metrics = {
        "train_p95_token_count": train_p95,
        "train_pos_mean_tokens": float(train_pos["mean"].iloc[0]) if not train_pos.empty else 0.0,
        "train_neg_mean_tokens": float(train_neg["mean"].iloc[0]) if not train_neg.empty else 0.0,
    }
    return metrics


def run_eda(preprocessed: PreprocessOutputs, outputs_root: Path) -> EDAOutputs:
    """Generate Stage 2 EDA tables and figures."""
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined = _load_split_frames(preprocessed)

    class_distribution_table = tables_dir / "class_distribution.csv"
    class_distribution_figure = figures_dir / "class_distribution.png"
    length_summary_table = tables_dir / "length_summary.csv"
    length_distribution_figure = figures_dir / "length_distribution.png"

    class_metrics = _class_distribution(
        combined, class_distribution_table, class_distribution_figure
    )
    length_metrics = _length_analysis(
        combined, length_summary_table, length_distribution_figure
    )

    key_metrics = {**class_metrics, **length_metrics}

    return EDAOutputs(
        class_distribution_table=class_distribution_table,
        length_summary_table=length_summary_table,
        class_distribution_figure=class_distribution_figure,
        length_distribution_figure=length_distribution_figure,
        key_metrics=key_metrics,
    )
