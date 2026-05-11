from __future__ import annotations

import json
from pathlib import Path


METRIC_NAMES = ("val_acc", "val_mAUC", "test_acc", "test_mAUC")


def format_metric(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return "missing"


def main() -> int:
    root = Path("outputs") / "strong_ft_priority"
    paths = sorted(root.glob("*/metrics_official_beats_strong_ft_fold_1.json"))

    if not paths:
        print(f"No priority run metrics found under {root}.")
        return 0

    headers = ("config", *METRIC_NAMES, "output_dir")
    rows = []

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            (
                path.parent.name,
                *(format_metric(metrics.get(name)) for name in METRIC_NAMES),
                str(path.parent),
            )
        )

    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]

    print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
