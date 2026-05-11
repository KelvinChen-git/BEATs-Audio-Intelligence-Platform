from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_LABELS = {
    "baseline": "Baseline",
    "official_beats": "Official BEATs",
    "official_beats_improved": "Improved BEATs",
}
METRIC_LABELS = {
    "val_acc": "Val Acc",
    "val_mAUC": "Val mAUC",
    "test_acc": "Test Acc",
    "test_mAUC": "Test mAUC",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric_fold_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in _read_rows(path) if row.get("fold", "").isdigit()]


def plot_model_comparison() -> Path:
    rows = _read_rows(OUTPUT_DIR / "model_comparison_5fold.csv")
    models = ["baseline", "official_beats", "official_beats_improved"]
    metrics = ["val_acc", "val_mAUC", "test_acc", "test_mAUC"]
    by_model = {row["model"]: row for row in rows}

    fig, ax = plt.subplots(figsize=(11, 6))
    x_positions = range(len(metrics))
    width = 0.24
    offsets = [-width, 0, width]
    colors = ["#7f8c8d", "#2f80ed", "#27ae60"]

    for offset, model, color in zip(offsets, models, colors):
        row = by_model[model]
        means = [float(row[f"{metric}_mean"]) for metric in metrics]
        stds = [float(row[f"{metric}_std"]) for metric in metrics]
        positions = [x + offset for x in x_positions]
        ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=MODEL_LABELS[model],
            color=color,
            alpha=0.9,
        )

    ax.set_title("ESC-50 5-Fold Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x_positions), [METRIC_LABELS[metric] for metric in metrics])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "model_comparison_5fold.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_beats_foldwise_comparison() -> Path:
    official_rows = _numeric_fold_rows(OUTPUT_DIR / "metrics_summary_official_beats_5fold.csv")
    improved_rows = _numeric_fold_rows(OUTPUT_DIR / "metrics_summary_official_beats_improved_5fold.csv")
    folds = [int(row["fold"]) for row in official_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    series = [
        ("test_acc", "Per-Fold Test Accuracy"),
        ("test_mAUC", "Per-Fold Test mAUC"),
    ]

    for ax, (metric, title) in zip(axes, series):
        official_values = [float(row[metric]) for row in official_rows]
        improved_values = [float(row[metric]) for row in improved_rows]
        ax.plot(folds, official_values, marker="o", linewidth=2.2, label=MODEL_LABELS["official_beats"])
        ax.plot(
            folds,
            improved_values,
            marker="o",
            linewidth=2.2,
            label=MODEL_LABELS["official_beats_improved"],
        )
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.set_xticks(folds)
        ax.grid(axis="both", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.6, 1.02)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Official BEATs Foldwise Test Performance", y=1.02)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "beats_foldwise_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = plot_model_comparison()
    foldwise_path = plot_beats_foldwise_comparison()
    print(f"Saved: {comparison_path}")
    print(f"Saved: {foldwise_path}")


if __name__ == "__main__":
    main()
