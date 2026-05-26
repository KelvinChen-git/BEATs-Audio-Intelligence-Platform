from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path("outputs") / "final_two_line_summary.csv"
FIGURE_PATH = Path("outputs") / "final_two_line_summary.png"
METRICS_FIGURE_PATH = Path("outputs") / "final_two_line_summary_metrics.png"


def read_rows() -> list[dict[str, str]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_PATH}")
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("test_acc") and row.get("test_mAUC")]
    wanted = {"Reproduced BEATs", "Improved BEATs + TTA-3"}
    rows = [row for row in rows if row.get("setting_name") in wanted]
    return sorted(rows, key=lambda row: 0 if row["setting_name"] == "Reproduced BEATs" else 1)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def save_placeholder(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "No final replication/TTA metrics found yet.",
        ha="center",
        va="center",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)


def main() -> int:
    rows = read_rows()
    if not rows:
        save_placeholder(FIGURE_PATH)
        save_placeholder(METRICS_FIGURE_PATH)
        print(f"No rows to plot in {SUMMARY_PATH}; saved placeholder to {FIGURE_PATH}.")
        return 0

    labels = ["Reproduced\nBEATs", "Improved\nBEATs + TTA-3"]
    test_acc = [float(row["test_acc"]) for row in rows]
    test_mauc = [float(row["test_mAUC"]) for row in rows]
    improvement = test_acc[1] - test_acc[0] if len(test_acc) == 2 else 0.0

    x_positions = np.arange(len(rows))
    colors = ["#5f6f7a", "#1f6f6d"]

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    bars = ax.bar(x_positions, test_acc, width=0.52, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0.94, 0.96)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Final ESC-50 Fold-1 Result")
    ax.set_xticks(x_positions, labels)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, test_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.00035,
            format_percent(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if len(test_acc) == 2:
        y = max(test_acc) + 0.0022
        ax.annotate(
            "+0.75 percentage points",
            xy=(1, test_acc[1]),
            xytext=(0.5, y),
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "black"},
            fontsize=10,
        )
    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300)

    metrics = ("test_acc", "test_mAUC")
    metric_values = {
        "test_acc": test_acc,
        "test_mAUC": test_mauc,
    }
    width = 0.34
    fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
    ax2.bar(x_positions - width / 2, metric_values["test_acc"], width, label="test_acc", color="#1f6f6d")
    ax2.bar(x_positions + width / 2, metric_values["test_mAUC"], width, label="test_mAUC", color="#d28c45")
    ax2.set_ylim(0.94, 1.0)
    ax2.set_ylabel("Score")
    ax2.set_title("Final Metrics")
    ax2.set_xticks(x_positions, labels)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for offset, metric in [(-width / 2, "test_acc"), (width / 2, "test_mAUC")]:
        for index, value in enumerate(metric_values[metric]):
            ax2.text(index + offset, value + 0.001, format_percent(value), ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(METRICS_FIGURE_PATH, dpi=300)

    print(f"Saved plot to {FIGURE_PATH}")
    print(f"Saved metrics plot to {METRICS_FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
