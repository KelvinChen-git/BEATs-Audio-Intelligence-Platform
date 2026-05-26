from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


OUTPUTS = Path("outputs")
ARCHIVE_ROOT = Path("outputs_archive")
MANIFEST_PATH = Path("outputs_cleanup_manifest.csv")

KEEP_ITEMS = {
    Path("outputs/.gitkeep"): ("keep", "Project placeholder."),
    Path("outputs/final_two_line_summary.csv"): ("final_report_required", "Final two-line summary CSV."),
    Path("outputs/final_two_line_summary.png"): ("final_report_required", "Main final report figure."),
    Path("outputs/final_two_line_summary_metrics.png"): ("final_report_optional", "Optional final metrics figure."),
    Path("outputs/tta_ablation_fold1.csv"): ("final_report_required", "TTA view-count ablation summary."),
    Path("outputs/tta_ablation_fold1.png"): ("final_report_required", "TTA view-count ablation plot."),
    Path("outputs/replication_standard_cv/full_lr3e6_fold1"): (
        "final_result_evidence",
        "Reproduced BEATs fold-1 checkpoint and metrics.",
    ),
    Path("outputs/tta_ablation/fold1_views1"): (
        "final_result_evidence",
        "Single-view reproduced BEATs evaluation evidence.",
    ),
    Path("outputs/tta_ablation/fold1_views3"): (
        "final_result_evidence",
        "Selected TTA-3 improvement evidence.",
    ),
}

ARCHIVE_ITEMS = {
    Path("outputs/tta_ablation/fold1_views5"): (
        "optional_ablation_evidence",
        "Unselected TTA view-count ablation result.",
    ),
    Path("outputs/tta_ablation/fold1_views7"): (
        "optional_ablation_evidence",
        "Unselected TTA view-count ablation result.",
    ),
    Path("outputs/tta_ablation/fold1_views9"): (
        "optional_ablation_evidence",
        "Unselected TTA view-count ablation result.",
    ),
    Path("outputs/improvement_tta"): (
        "historical_unselected_tta",
        "Earlier TTA helper output; final line uses tta_ablation/fold1_views3.",
    ),
    Path("outputs/debug_json_protocol_top8_1epoch"): (
        "development_debug",
        "One-epoch debug run, not a final result.",
    ),
    Path("outputs/debug_standard_cv_top8_1epoch"): (
        "development_debug",
        "One-epoch debug run, not a final result.",
    ),
    Path("outputs/strong_ft_priority/top8_esc50_standard_cv"): (
        "invalid_bad_amp",
        "Invalid pre-fix AMP run with near-random performance.",
    ),
    Path("outputs/strong_ft_priority/top8_esc50_standard_cv_fixed"): (
        "development_experiment",
        "Development strong fine-tuning run.",
    ),
    Path("outputs/strong_ft_priority/top12_esc50_standard_cv_fixed"): (
        "development_experiment",
        "Development strong fine-tuning run.",
    ),
    Path("outputs/strong_ft_priority/full_esc50_standard_cv_fixed"): (
        "development_experiment",
        "Development strong fine-tuning run.",
    ),
    Path("outputs/strong_ft_priority/full_esc50_standard_cv_40ep"): (
        "development_experiment",
        "Development strong fine-tuning run.",
    ),
    Path("outputs/strong_ft_priority/full_esc50_standard_cv_lr3e6"): (
        "development_experiment",
        "Development strong fine-tuning run used for hyperparameter selection.",
    ),
    Path("outputs/strong_ft_priority/full_esc50_standard_cv_lr3e6_ls005_20ep"): (
        "development_experiment",
        "Development strong fine-tuning run.",
    ),
}

HISTORICAL_ROOT_GLOBS = [
    "best_baseline_fold_*.pt",
    "best_official_beats_fold_*.pt",
    "best_official_beats_improved_fold_*.pt",
    "best_official_beats_strong_ft_fold_1.pt",
    "metrics_baseline_fold_*.json",
    "metrics_official_beats_fold_*.json",
    "metrics_official_beats_improved_fold_*.json",
    "metrics_official_beats_strong_ft_fold_1.json",
    "metrics_summary_baseline*.csv",
    "metrics_summary_official_beats.csv",
    "metrics_summary_official_beats_5fold.csv",
    "metrics_summary_official_beats_improved*.csv",
    "metrics_summary_official_beats_strong_ft.csv",
    "model_comparison_5fold.csv",
    "model_comparison_5fold.png",
    "beats_foldwise_comparison.png",
]


def collect_existing_items() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    keep_rows = []
    archive_rows = []

    for path, (category, note) in KEEP_ITEMS.items():
        if path.exists():
            keep_rows.append(row_for(path, "keep", "", category, note))

    archive_map = dict(ARCHIVE_ITEMS)
    for pattern in HISTORICAL_ROOT_GLOBS:
        for path in OUTPUTS.glob(pattern):
            archive_map.setdefault(
                path,
                ("historical_old_experiment", "Historical baseline/frozen/improved output, not final aligned result."),
            )

    for path, (category, note) in sorted(archive_map.items(), key=lambda item: str(item[0])):
        if path.exists():
            archive_rows.append(row_for(path, "archive", "", category, note))

    return keep_rows, archive_rows


def row_for(path: Path, action: str, destination: str, category: str, note: str) -> dict[str, str]:
    return {
        "original_path": str(path),
        "action": action,
        "destination_path": destination,
        "category": category,
        "note": note,
    }


def with_destinations(rows: list[dict[str, str]], timestamp: str) -> list[dict[str, str]]:
    updated = []
    for row in rows:
        original = Path(row["original_path"])
        destination = ARCHIVE_ROOT / timestamp / original
        updated.append({**row, "destination_path": str(destination)})
    return updated


def write_manifest(rows: list[dict[str, str]]) -> None:
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_path", "action", "destination_path", "category", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def move_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        source = Path(row["original_path"])
        destination = Path(row["destination_path"])
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def print_summary(keep_rows: list[dict[str, str]], archive_rows: list[dict[str, str]], apply: bool) -> None:
    print(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    print(f"Keep items: {len(keep_rows)}")
    for row in keep_rows:
        print(f"KEEP    {row['original_path']} [{row['category']}]")
    print(f"Archive candidates: {len(archive_rows)}")
    for row in archive_rows:
        print(f"ARCHIVE {row['original_path']} -> {row['destination_path']} [{row['category']}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely archive non-final output artifacts.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move archive candidates into outputs_archive/final_submission_cleanup_<timestamp>.",
    )
    args = parser.parse_args()

    timestamp = f"final_submission_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    keep_rows, archive_rows = collect_existing_items()
    archive_rows = with_destinations(archive_rows, timestamp)
    manifest_rows = keep_rows + archive_rows

    write_manifest(manifest_rows)
    print_summary(keep_rows, archive_rows, args.apply)
    print(f"Wrote manifest to {MANIFEST_PATH}")

    if args.apply:
        move_rows(archive_rows)
        print(f"Archived outputs under {ARCHIVE_ROOT / timestamp}")
    else:
        print("Dry-run only. Re-run with --apply to move archive candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
