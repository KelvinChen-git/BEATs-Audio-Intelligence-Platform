from __future__ import annotations

import csv
import json
from pathlib import Path


OUTPUT_PATH = Path("outputs") / "final_two_line_summary.csv"
REPRODUCED_METRICS = (
    Path("outputs")
    / "tta_ablation"
    / "fold1_views1"
    / "metrics_official_beats_strong_ft_fold_1.json"
)
IMPROVED_METRICS = (
    Path("outputs")
    / "tta_ablation"
    / "fold1_views3"
    / "metrics_official_beats_strong_ft_fold_1.json"
)
FIELDNAMES = [
    "line_type",
    "setting_name",
    "protocol",
    "fold",
    "train_size",
    "val_size",
    "test_size",
    "epochs",
    "tta_enabled",
    "tta_num_views",
    "test_acc",
    "test_mAUC",
    "checkpoint",
    "output_dir",
    "note",
]


def load_row(path: Path, line_type: str, setting_name: str, note: str) -> dict[str, object] | None:
    if not path.exists():
        print(f"Missing metrics file, skipping: {path}")
        return None

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "line_type": line_type,
        "setting_name": setting_name,
        "protocol": payload.get("protocol", ""),
        "fold": payload.get("fold", ""),
        "train_size": payload.get("train_size", ""),
        "val_size": payload.get("val_size", ""),
        "test_size": payload.get("test_size", ""),
        "epochs": payload.get("epochs", ""),
        "tta_enabled": payload.get("tta_enabled", False),
        "tta_num_views": payload.get("tta_num_views", 1),
        "test_acc": payload.get("test_acc", ""),
        "test_mAUC": payload.get("test_mAUC", ""),
        "checkpoint": payload.get("checkpoint", ""),
        "output_dir": payload.get("output_dir", str(path.parent)),
        "note": note,
    }


def main() -> int:
    candidates = [
        load_row(
            REPRODUCED_METRICS,
            "replication",
            "Reproduced BEATs",
            "Single-view evaluation of reproduced BEATs full fine-tuning.",
        ),
        load_row(
            IMPROVED_METRICS,
            "improvement",
            "Improved BEATs + TTA-3",
            "Selected TTA-3 from view-count ablation for best accuracy/mAUC trade-off.",
        ),
    ]
    rows = [row for row in candidates if row is not None]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} final aligned row(s) to {OUTPUT_PATH}")
    if not rows:
        print("No final replication/TTA metrics found yet. Run the fold-1 scripts before expecting result rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
