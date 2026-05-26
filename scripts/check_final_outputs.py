from __future__ import annotations

import json
from pathlib import Path


REQUIRED_PATHS = [
    Path("outputs/final_two_line_summary.csv"),
    Path("outputs/final_two_line_summary.png"),
    Path("outputs/replication_standard_cv/full_lr3e6_fold1/metrics_official_beats_strong_ft_fold_1.json"),
    Path("outputs/tta_ablation/fold1_views3/metrics_official_beats_strong_ft_fold_1.json"),
]
REPRODUCED_METRICS = Path("outputs/tta_ablation/fold1_views1/metrics_official_beats_strong_ft_fold_1.json")
IMPROVED_METRICS = Path("outputs/tta_ablation/fold1_views3/metrics_official_beats_strong_ft_fold_1.json")


def load_metric(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        for path in missing:
            print(f"FAIL missing required output: {path}")
        return 1

    reproduced = load_metric(REPRODUCED_METRICS)
    improved = load_metric(IMPROVED_METRICS)
    reproduced_acc = float(reproduced["test_acc"])
    improved_acc = float(improved["test_acc"])
    improvement = improved_acc - reproduced_acc

    print("PASS final required outputs exist.")
    print(f"Reproduced BEATs test_acc: {reproduced_acc:.4f}")
    print(f"Improved BEATs + TTA-3 test_acc: {improved_acc:.4f}")
    print(f"Absolute improvement: {improvement:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
