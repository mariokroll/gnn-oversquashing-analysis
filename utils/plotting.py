import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent colour / marker / line style for every model across all charts
MODEL_ORDER = ["GCN", "GIN", "GCN_VN", "GIN_VN"]

_STYLES: dict[str, dict] = {
    "GCN": {"color": "#e74c3c", "marker": "o", "linestyle": "-", "label": "GCN"},
    "GIN": {"color": "#e67e22", "marker": "s", "linestyle": "--", "label": "GIN"},
    "GCN_VN": {
        "color": "#2980b9",
        "marker": "^",
        "linestyle": "-",
        "label": "GCN + VN",
    },
    "GIN_VN": {
        "color": "#27ae60",
        "marker": "D",
        "linestyle": "--",
        "label": "GIN + VN",
    },
}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_results(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Merge base_models and virtual_node_models CSVs into single summary and epoch DataFrames.

    Missing files are silently skipped, so the function works after Step 1,
    Step 2, or a combined run.  Duplicate (model, depth) rows are dropped so
    a model that appears in both files is not double-counted.
    """

    def _try(fname: str) -> pd.DataFrame | None:
        path = os.path.join(results_dir, fname)
        return pd.read_csv(path) if os.path.exists(path) else None

    summary_parts = [
        _try("base_models_summary.csv"),
        _try("virtual_node_models_summary.csv"),
    ]
    epoch_parts = [
        _try("base_models_epochs.csv"),
        _try("virtual_node_models_epochs.csv"),
    ]

    valid_s = [p for p in summary_parts if p is not None]
    valid_e = [p for p in epoch_parts if p is not None]

    if not valid_s:
        return pd.DataFrame(), None

    summary_df = (
        pd.concat(valid_s, ignore_index=True)
        .drop_duplicates(subset=["model", "depth"], keep="last")
        .reset_index(drop=True)
    )
    epoch_df = (
        pd.concat(valid_e, ignore_index=True)
        .drop_duplicates(subset=["model", "depth", "epoch"], keep="last")
        .reset_index(drop=True)
        if valid_e
        else None
    )
    return summary_df, epoch_df


def plot_accuracy_vs_depth(summary_df: pd.DataFrame, save_path: str) -> None:
    """Line chart comparing best train accuracy across tree depths for all models.

    A random-chance baseline (1 / num_leaves = 1 / 2^depth) is overlaid as a
    dashed grey line so the reader can immediately gauge how far each model is
    above chance.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = sorted(summary_df["depth"].unique())

    for model in MODEL_ORDER:
        subset = summary_df[summary_df["model"] == model].sort_values("depth")
        if subset.empty:
            continue
        s = _STYLES[model]
        ax.plot(
            subset["depth"],
            subset["final_train_acc"],
            label=s["label"],
            color=s["color"],
            marker=s["marker"],
            linestyle=s["linestyle"],
            linewidth=2.5,
            markersize=9,
            zorder=3,
        )

    # Random-chance baseline: 1 / 2^depth
    random_acc = [1.0 / (2**d) for d in depths]
    ax.plot(
        depths,
        random_acc,
        linestyle=":",
        color="gray",
        linewidth=1.8,
        label="Random baseline",
        zorder=2,
    )

    ax.set_xlabel("Tree Depth", fontsize=13)
    ax.set_ylabel("Best Train Accuracy", fontsize=13)
    ax.set_title(
        "GNN Performance vs. Tree Depth\n"
        "(Tree-NeighborsMatch — over-squashing benchmark)",
        fontsize=13,
    )
    ax.set_xticks(depths)
    ax.set_ylim(-0.03, 1.07)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_training_time(summary_df: pd.DataFrame, save_path: str) -> None:
    """Grouped bar chart: wall-clock training time (seconds) per model and depth.

    Each depth has one cluster of bars — one bar per model present in the data.
    This lets the reader compare both within-depth overhead and across-depth
    scaling for baselines vs. VN augmented models.
    """
    depths = sorted(summary_df["depth"].unique())
    models_present = [m for m in MODEL_ORDER if m in summary_df["model"].values]
    n_groups, n_bars = len(depths), len(models_present)

    x = np.arange(n_groups)
    width = 0.75 / n_bars

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models_present):
        subset = summary_df[summary_df["model"] == model].sort_values("depth")
        offsets = x + (i - n_bars / 2 + 0.5) * width
        s = _STYLES[model]
        bars = ax.bar(
            offsets,
            subset["training_time_s"],
            width=width * 0.88,
            label=s["label"],
            color=s["color"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )
        # Annotate each bar with the value
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(summary_df["training_time_s"]) * 0.01,
                f"{h:.0f}s",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

    ax.set_xlabel("Tree Depth", fontsize=13)
    ax.set_ylabel("Training Time (seconds)", fontsize=13)
    ax.set_title("Wall-clock Training Time per Model and Depth", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Depth {d}" for d in depths])
    ax.legend(framealpha=0.9, fontsize=11)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_ylim(0, summary_df["training_time_s"].max() * 1.20)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def plot_learning_curves(epoch_df: pd.DataFrame, save_path: str) -> None:
    """Train-accuracy vs. epoch curves, one subplot per tree depth.

    A 10-epoch rolling mean is applied to each curve so that the convergence
    trend is visible despite batch-level noise.  All subplots share the same
    y-axis so depths can be compared at a glance.
    """
    depths = sorted(epoch_df["depth"].unique())
    n_depths = len(depths)

    fig, axes = plt.subplots(
        1,
        n_depths,
        figsize=(4.8 * n_depths, 4.8),
        sharey=True,
    )
    if n_depths == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        for model in MODEL_ORDER:
            subset = epoch_df[
                (epoch_df["depth"] == depth) & (epoch_df["model"] == model)
            ].sort_values("epoch")
            if subset.empty:
                continue
            s = _STYLES[model]
            # 10-epoch rolling mean to smooth noise
            smoothed = subset["train_acc"].rolling(10, min_periods=1).mean()
            ax.plot(
                subset["epoch"],
                smoothed,
                label=s["label"],
                color=s["color"],
                linestyle=s["linestyle"],
                linewidth=1.8,
                alpha=0.9,
            )

        # Random-chance line for this depth
        random_acc = 1.0 / (2**depth)
        ax.axhline(
            random_acc,
            linestyle=":",
            color="gray",
            linewidth=1.2,
            alpha=0.7,
            label="Random" if ax is axes[0] else None,
        )

        ax.set_title(f"Depth = {depth}", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Train Accuracy (10-ep rolling mean)", fontsize=11)
        ax.set_ylim(-0.03, 1.07)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(True, alpha=0.35)

    # Shared legend, attached to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        fontsize=10,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        "Learning Curves: Training Accuracy vs. Epoch",
        fontsize=13,
        y=1.09,
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def generate_all_plots(results_dir: str) -> None:
    """Load all available CSV logs and produce every chart.

    Safe to call after any step — charts whose required data is absent are
    silently skipped rather than raising an error.
    """
    _setup_style()

    summary_df, epoch_df = load_results(results_dir)

    if summary_df.empty:
        print("  No summary CSV found — skipping all plots.")
        return

    print("\nGenerating plots …")

    plot_accuracy_vs_depth(
        summary_df,
        os.path.join(results_dir, "plot_accuracy_vs_depth.png"),
    )
    plot_training_time(
        summary_df,
        os.path.join(results_dir, "plot_training_time.png"),
    )
    if epoch_df is not None and not epoch_df.empty:
        plot_learning_curves(
            epoch_df,
            os.path.join(results_dir, "plot_learning_curves.png"),
        )

    print("All plots saved.")
