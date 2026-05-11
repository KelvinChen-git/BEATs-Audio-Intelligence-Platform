from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import get_default_device


def main() -> None:
    cuda_available = torch.cuda.is_available()
    print(f"Python version: {platform.python_version()}")
    print(f"torch version: {torch.__version__}")
    print(f"torchaudio version: {torchaudio.__version__}")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA device name: -")
    print(f"train.py selected device: {get_default_device()}")


if __name__ == "__main__":
    main()
