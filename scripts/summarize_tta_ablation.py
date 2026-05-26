from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path("outputs") / "tta_ablation"
OUTPUT_PATH = Path("outputs") / "tta_ablation_fold1.csv"
FIELDNAMES = ["setting", "tta_num_views", "test_acc", "test_mAUC", "checkpoint", "output_dir"]


def infer_views(path: Path, payload: dict[str, object]) -> int:
    value = payload.get("tta_num_views")
    if isinstance(value, int):
        return value
    match = re.search(r"views(\d+)", path.parent.name)
    return int(match.group(1)) if match else 1


def load_rows() -> list[dict[str, object]]:
    rows = []
    for path in sorted(ROOT.glob("fold1_views*/metrics_official_beats_strong_ft_fold_1.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "setting": path.parent.name,
                "tta_num_views": infer_views(path, payload),
                "test_acc": payload.get("test_acc", ""),
                "test_mAUC": payload.get("test_mAUC", ""),
                "checkpoint": payload.get("checkpoint", ""),
                "output_dir": payload.get("output_dir", str(path.parent)),
            }
        )
    return sorted(rows, key=lambda row: int(row["tta_num_views"]))


def print_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        print(f"No TTA ablation metrics found under {ROOT}.")
        return

    widths = [
        max(len(str(row[field])) for row in rows + [{field: field}])
        for field in FIELDNAMES
    ]
    print("  ".join(field.ljust(widths[index]) for index, field in enumerate(FIELDNAMES)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(row[field]).ljust(widths[index]) for index, field in enumerate(FIELDNAMES)))


def main() -> int:
    rows = load_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print(f"Wrote {len(rows)} row(s) to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
