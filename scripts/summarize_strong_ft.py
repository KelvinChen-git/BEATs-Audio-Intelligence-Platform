from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS = ["val_acc", "val_mAUC", "test_acc", "test_mAUC"]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(OUTPUT_DIR.glob("metrics_official_beats_strong_ft_fold_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {
            "model": payload.get("model", "official_beats_strong_ft"),
            "fold": int(payload["fold"]),
            "best_epoch": payload.get("best_epoch", ""),
            "checkpoint": payload.get("checkpoint", ""),
            "protocol": payload.get("protocol", ""),
            "metrics_path": str(path),
        }
        for metric in METRICS:
            row[metric] = float(payload[metric])
        rows.append(row)
    return rows


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    rows = load_rows()
    if not rows:
        raise FileNotFoundError(f"No strong fine-tuning metrics found under {OUTPUT_DIR}")

    summary_path = OUTPUT_DIR / "metrics_summary_official_beats_strong_ft_5fold.csv"
    fieldnames = [
        "model",
        "fold",
        "best_epoch",
        "val_acc",
        "val_mAUC",
        "test_acc",
        "test_mAUC",
        "checkpoint",
        "protocol",
        "metrics_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

        mean_row = {"model": "official_beats_strong_ft", "fold": "mean"}
        std_row = {"model": "official_beats_strong_ft", "fold": "sample_std"}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            mean_row[metric] = mean(values)
            std_row[metric] = sample_std(values)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    print(f"Saved summary to: {summary_path}")
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        print(f"{metric}: mean={mean(values):.6f}, std={sample_std(values):.6f}")


if __name__ == "__main__":
    main()
