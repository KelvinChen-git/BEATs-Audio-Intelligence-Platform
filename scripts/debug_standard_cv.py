from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import label_to_id, load_esc50_metadata_items, split_esc50_standard_cv
from src.train import get_fold_files


ESC_ROOT = PROJECT_ROOT / "data" / "ESC-50-master"
META_CSV = ESC_ROOT / "meta" / "esc50.csv"
AUDIO_ROOT = ESC_ROOT / "audio"
JSON_DIR = PROJECT_ROOT / "datafiles"


def read_metadata_rows() -> list[dict[str, str]]:
    with META_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def print_split_stats(name: str, items: list[tuple[str, int]]) -> None:
    labels = [label for _, label in items]
    counts = Counter(labels)
    print(f"{name} size: {len(items)}")
    print(f"{name} unique labels: {len(counts)}")
    print(f"{name} min/max label: {min(labels) if labels else '-'} / {max(labels) if labels else '-'}")
    print(f"{name} per-class counts:")
    print(" ".join(f"{label}:{counts.get(label, 0)}" for label in range(50)))


def read_json_items(path: Path) -> list[tuple[str, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    return [(Path(row["wav"]).name, label_to_id(row["labels"])) for row in rows]


def print_overlap(name: str, left: list[tuple[str, int]], right: list[tuple[str, int]]) -> None:
    overlap = {filename for filename, _ in left} & {filename for filename, _ in right}
    print(f"{name} overlap exists: {bool(overlap)} ({len(overlap)} samples)")
    if overlap:
        print(f"{name} overlap examples: {sorted(overlap)[:10]}")


def main() -> int:
    rows = read_metadata_rows()
    meta_items = load_esc50_metadata_items(META_CSV)
    by_filename = {row["filename"]: row for row in rows}
    splits = split_esc50_standard_cv(meta_items=meta_items, fold=1, seed=42)

    print("ESC-50 standard CV fold 1")
    print(f"metadata rows: {len(rows)}")
    print(f"expected test fold: 1")
    print(f"actual test folds: {sorted({by_filename[name]['fold'] for name, _ in splits['test']})}")
    print(f"actual train folds: {sorted({by_filename[name]['fold'] for name, _ in splits['train']})}")
    print(f"actual val folds: {sorted({by_filename[name]['fold'] for name, _ in splits['val']})}")
    print("")

    for split_name in ("train", "val", "test"):
        print_split_stats(split_name, splits[split_name])
        print("")

    print("First 10 metadata samples:")
    for row in rows[:10]:
        path = AUDIO_ROOT / row["filename"]
        print(
            f"{path} | category={row['category']} | target={row['target']} | "
            f"fold={row['fold']} | exists={path.exists()}"
        )
    print("")

    print_overlap("train/test", splits["train"], splits["test"])
    print_overlap("val/test", splits["val"], splits["test"])
    print_overlap("train/val", splits["train"], splits["val"])
    print("")

    all_labels = [int(row["target"]) for row in rows]
    print("Expected-behavior checks:")
    print(f"total dataset is 2000: {len(rows) == 2000}")
    print(f"test set has 400 samples: {len(splits['test']) == 400}")
    print(f"each class appears in test set: {len({label for _, label in splits['test']}) == 50}")
    print(f"labels are integers 0-49: {sorted(set(all_labels)) == list(range(50))}")
    print("")

    fold_files = get_fold_files(JSON_DIR, 1)
    json_splits = {name: read_json_items(path) for name, path in fold_files.items()}
    print("JSON protocol fold 1")
    for split_name, items in json_splits.items():
        print_split_stats(f"json_{split_name}", items)
        print("")

    json_items = [item for items in json_splits.values() for item in items]
    mismatches = []
    for filename, json_label in json_items:
        metadata_label = int(by_filename[filename]["target"])
        if json_label != metadata_label:
            mismatches.append((filename, json_label, metadata_label, by_filename[filename]["category"]))

    print("JSON vs ESC-50 metadata label comparison:")
    print(f"json samples checked: {len(json_items)}")
    print(f"label mismatches: {len(mismatches)}")
    for filename, json_label, metadata_label, category in mismatches[:20]:
        print(f"{filename}: json_label={json_label}, metadata_target={metadata_label}, category={category}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
