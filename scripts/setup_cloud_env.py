from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "BEATs_iter3_plus_AS2M.pt"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=check)


def print_versions() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch: not importable ({exc})")

    try:
        import torchaudio

        print(f"torchaudio: {torchaudio.__version__}")
    except Exception as exc:
        print(f"torchaudio: not importable ({exc})")


def install_requirements() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def clone_beats() -> None:
    third_party = REPO_ROOT / "third_party"
    unilm = third_party / "unilm"
    beats = unilm / "beats" / "BEATs.py"
    third_party.mkdir(exist_ok=True)
    if beats.exists():
        print(f"BEATs source already present: {beats}")
        return
    if unilm.exists():
        raise RuntimeError(f"{unilm} exists but {beats} is missing. Inspect third_party/unilm manually.")
    run(["git", "clone", "--depth", "1", "https://github.com/microsoft/unilm.git", str(unilm)])


def download_checkpoint(url: str, destination: Path = DEFAULT_CHECKPOINT) -> None:
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        print(f"Checkpoint already present: {destination}")
        return
    print(f"Downloading checkpoint to {destination}")
    urllib.request.urlretrieve(url, destination)


def check_assets() -> bool:
    checks = {
        "BEATs source": REPO_ROOT / "third_party" / "unilm" / "beats" / "BEATs.py",
        "BEATs checkpoint": DEFAULT_CHECKPOINT,
        "ESC-50 audio": REPO_ROOT / "data" / "ESC-50-master" / "audio",
        "ESC-50 metadata": REPO_ROOT / "data" / "ESC-50-master" / "meta" / "esc50.csv",
    }
    ok = True
    for name, path in checks.items():
        exists = path.exists()
        print(f"{'PASS' if exists else 'MISSING'}: {name} -> {path}")
        ok = ok and exists
    return ok


def maybe_run_smoke(args: argparse.Namespace) -> None:
    if args.run_tests:
        run([sys.executable, "-m", "pytest", "-q"])

    if args.run_baseline_smoke:
        run(
            [
                sys.executable,
                "-m",
                "src.train",
                "--model",
                "baseline",
                "--fold",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "4",
            ]
        )

    if args.run_official_smoke:
        run(
            [
                sys.executable,
                "-m",
                "src.train",
                "--model",
                "official_beats",
                "--fold",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--checkpoint-path",
                str(DEFAULT_CHECKPOINT),
                "--freeze-backbone",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and verify a Codex/cloud runtime for this project.")
    parser.add_argument("--install", action="store_true", help="Install Python dependencies from requirements.txt.")
    parser.add_argument("--clone-beats", action="store_true", help="Clone official UniLM/BEATs into third_party/unilm.")
    parser.add_argument(
        "--checkpoint-url",
        default=os.environ.get("BEATS_CHECKPOINT_URL", ""),
        help="Optional URL used to download BEATs_iter3_plus_AS2M.pt.",
    )
    parser.add_argument("--run-tests", action="store_true", help="Run python -m pytest -q after setup.")
    parser.add_argument("--run-baseline-smoke", action="store_true", help="Run a 1-epoch baseline smoke test.")
    parser.add_argument("--run-official-smoke", action="store_true", help="Run a 1-epoch frozen BEATs smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_versions()
    if args.install:
        install_requirements()
    if args.clone_beats:
        clone_beats()
    if args.checkpoint_url:
        download_checkpoint(args.checkpoint_url)

    assets_ok = check_assets()
    if not assets_ok:
        print(
            "Some runtime assets are missing. Tests can still run if they do not require audio/checkpoints, "
            "but training needs ESC-50 audio, BEATs source, and the BEATs checkpoint."
        )

    maybe_run_smoke(args)


if __name__ == "__main__":
    main()
