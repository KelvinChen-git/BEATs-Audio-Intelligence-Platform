# Codex Cloud Run Guide

This repository is ready to open in Codex Cloud from GitHub:

```text
https://github.com/sneaker234/beats-esc50
```

## First Cloud Command

Run this first in a fresh Codex cloud environment:

```bash
python scripts/setup_cloud_env.py --install --clone-beats --run-tests
```

This installs Python dependencies, clones the official Microsoft UniLM repository into `third_party/unilm`, checks runtime assets, and runs the project tests.

## Required Assets For Training

The GitHub repository intentionally does not include large runtime assets:

```text
data/ESC-50-master/audio/
checkpoints/BEATs_iter3_plus_AS2M.pt
third_party/unilm/
```

`third_party/unilm/` can be recreated with `--clone-beats`. The ESC-50 audio and BEATs checkpoint must be provided in the cloud environment before training.

If the checkpoint is available through a private or temporary URL, set it as an environment variable and run:

```bash
export BEATS_CHECKPOINT_URL="https://example.com/BEATs_iter3_plus_AS2M.pt"
python scripts/setup_cloud_env.py --install --clone-beats --run-tests
```

The setup script will download it to:

```text
checkpoints/BEATs_iter3_plus_AS2M.pt
```

## Verify Cloud Environment

After setup, run:

```bash
python scripts/check_env.py
python -m pytest -q
```

If `torch` shows `+cpu` or `CUDA available: False`, do not run long strong fine-tuning in that environment.

## Training Smoke Commands

Baseline smoke test:

```bash
python -m src.train --model baseline --fold 1 --epochs 1 --batch-size 4
```

Frozen BEATs smoke test:

```bash
python -m src.train --model official_beats --fold 1 --epochs 1 --batch-size 2 --checkpoint-path checkpoints/BEATs_iter3_plus_AS2M.pt --freeze-backbone
```

Final replication fold 1 requires GPU-class runtime and all assets:

```bash
python -m src.train --model official_beats_strong_ft --protocol esc50_standard_cv --use-trainval-for-final --fold 1 --epochs 17 --batch-size 2 --checkpoint-path checkpoints/BEATs_iter3_plus_AS2M.pt --full-finetune --backbone-lr 3e-6 --head-lr 5e-4 --weight-decay 0.01 --label-smoothing 0.1 --grad-clip 1.0 --scheduler cosine --warmup-ratio 0.05 --amp --output-dir outputs/replication_standard_cv/full_lr3e6_fold1
```

## Recommended Codex Prompt

Use this as the first Codex Cloud task:

```text
Install dependencies, clone official UniLM/BEATs with scripts/setup_cloud_env.py, run python -m pytest -q, then report which assets are missing for baseline and official BEATs training. Do not retrain unless all required assets are present.
```
