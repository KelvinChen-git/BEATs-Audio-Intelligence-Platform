from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


CSV_PATH = Path("outputs") / "tta_ablation_fold1.csv"
FIGURE_PATH = Path("outputs") / "tta_ablation_fold1.png"


def read_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("tta_num_views") and row.get("test_acc") and row.get("test_mAUC")
        ]


def save_placeholder() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, "No TTA ablation metrics found yet.", ha="center", va="center", fontsize=12)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)


def main() -> int:
    rows = sorted(read_rows(), key=lambda row: int(row["tta_num_views"]))
    if not rows:
        save_placeholder()
        print(f"No rows to plot; saved placeholder to {FIGURE_PATH}.")
        return 0

    views = [int(row["tta_num_views"]) for row in rows]
    test_acc = [float(row["test_acc"]) for row in rows]
    test_mauc = [float(row["test_mAUC"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(views, test_acc, marker="o", linewidth=2.2, color="#2f6f73")
    axes[0].set_ylabel("test_acc")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Fold 1 TTA View Count Ablation")

    axes[1].plot(views, test_mauc, marker="o", linewidth=2.2, color="#d28c45")
    axes[1].set_xlabel("TTA views")
    axes[1].set_ylabel("test_mAUC")
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(views)

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    print(f"Saved plot to {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
