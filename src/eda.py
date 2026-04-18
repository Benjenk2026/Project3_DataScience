"""Exploratory data analysis utilities for the HIGGS dataset.

This module provides reusable plotting functions and a small CLI that can be
run directly from the src directory:

    cd src
    python eda.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _get_label_column(df: pd.DataFrame) -> str:
    """Return the label column name, preferring an explicit 'label' column."""
    if "label" in df.columns:
        return "label"
    return df.columns[0]


def _get_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    """Return feature columns used for plots.

    If the canonical HIGGS feature names are present, use feature_1..feature_28
    to ensure a 4x7 visualization grid. Otherwise fall back to all numeric
    non-label columns.
    """
    canonical = [f"feature_{i}" for i in range(1, 29)]
    if all(col in df.columns for col in canonical):
        return canonical

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c != label_col]


def _ensure_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_or_show(fig: plt.Figure, save: bool, output_path: Path | None = None) -> None:
    if save:
        if output_path is None:
            raise ValueError("output_path must be provided when save=True")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_class_distribution(df: pd.DataFrame, save: bool = True, output_dir: str = "../output/eda") -> None:
    """Bar chart of label=0 vs label=1 counts."""
    label_col = _get_label_column(df)
    counts = df[label_col].value_counts(dropna=False).sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        palette=["#2b8cbe", "#e34a33"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Class Distribution (Signal vs Background)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")

    for idx, value in enumerate(counts.values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_dir = _ensure_output_dir(output_dir)
    _save_or_show(fig, save=save, output_path=out_dir / "class_distribution.png")


def plot_feature_histograms(df: pd.DataFrame, save: bool = True, output_dir: str = "../output/eda") -> None:
    """Histogram grid for all features (4x7 when 28 HIGGS features are present)."""
    label_col = _get_label_column(df)
    feature_cols = _get_feature_columns(df, label_col)

    n_features = len(feature_cols)
    n_cols = 7
    n_rows = 4 if n_features == 28 else max(1, (n_features + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12 if n_rows == 4 else 3 * n_rows))
    axes = axes.flatten() if isinstance(axes, Iterable) else [axes]

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.histplot(df[col], bins=40, kde=False, color="#1f78b4", edgecolor=None, ax=ax)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature Distributions", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir = _ensure_output_dir(output_dir)
    _save_or_show(fig, save=save, output_path=out_dir / "feature_histograms.png")


def plot_correlation_matrix(df: pd.DataFrame, save: bool = True, output_dir: str = "../output/eda") -> None:
    """Heatmap of feature-feature Pearson correlations."""
    label_col = _get_label_column(df)
    feature_cols = _get_feature_columns(df, label_col)
    corr = df[feature_cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Pearson)")

    fig.tight_layout()
    out_dir = _ensure_output_dir(output_dir)
    _save_or_show(fig, save=save, output_path=out_dir / "correlation_matrix.png")


def plot_boxplots_by_label(df: pd.DataFrame, save: bool = True, output_dir: str = "../output/eda") -> None:
    """Boxplots per feature, grouped and colored by label."""
    label_col = _get_label_column(df)
    feature_cols = _get_feature_columns(df, label_col)

    n_features = len(feature_cols)
    n_cols = 7
    n_rows = 4 if n_features == 28 else max(1, (n_features + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12 if n_rows == 4 else 3 * n_rows))
    axes = axes.flatten() if isinstance(axes, Iterable) else [axes]

    palette = {0: "#2b8cbe", 1: "#e34a33", "0": "#2b8cbe", "1": "#e34a33"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x=label_col,
            y=col,
            hue=label_col,
            palette=palette,
            dodge=False,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Label")
        ax.set_ylabel("")

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature Boxplots by Label", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir = _ensure_output_dir(output_dir)
    _save_or_show(fig, save=save, output_path=out_dir / "boxplots_by_label.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA plots for the HIGGS dataset.")
    parser.add_argument(
        "--input",
        default="../data/processed/higgs_cleaned.csv",
        help="Path to cleaned input CSV (default: ../data/processed/higgs_cleaned.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="../output/eda",
        help="Directory to save plot images (default: ../output/eda)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional row limit to speed up plotting during quick checks.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of only saving files.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, nrows=args.rows)
    save_plots = not args.show

    plot_class_distribution(df, save=save_plots, output_dir=args.output_dir)
    plot_feature_histograms(df, save=save_plots, output_dir=args.output_dir)
    plot_correlation_matrix(df, save=save_plots, output_dir=args.output_dir)
    plot_boxplots_by_label(df, save=save_plots, output_dir=args.output_dir)

    if save_plots:
        print(f"EDA plots saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
