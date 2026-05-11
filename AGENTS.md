# Repository instructions for Codex

## Goal
Complete a coursework-ready ESC-50 audio classification project that supports:
- a baseline log-mel Transformer model
- an official BEATs backbone from `third_party/unilm/beats`
- 5-fold training/evaluation using the manifests in `datafiles/`

## Constraints
- Do **not** generate coursework report text.
- Do **not** change the JSON manifest schema.
- Keep all code in Python.
- Prefer small, high-confidence edits.
- Before large edits, explain the plan briefly.
- After edits, run the smallest relevant verification command.

## File map
- `src/dataset.py`: manifest loading and audio path resolution
- `src/model_baseline.py`: baseline model
- `src/beats_adapter.py`: official BEATs wrapper for ESC-50
- `src/train.py`: training and evaluation entry point
- `scripts/setup_beats.ps1`: clone official repo scaffold
- `prompts/`: task prompts for Codex

## Project expectations
- Treat `data/ESC-50-master/audio/` as the audio root.
- Train fold 2 from `esc_trainl_data_2.json` if needed.
- Keep CLI arguments backward compatible when possible.
- Save metrics to `outputs/`.
- If the official BEATs repo or checkpoint is missing, fail with a clear message instead of crashing obscurely.

## Done means
A task is done only when:
1. code changes are applied,
2. the relevant command runs or a clear blocker is reported,
3. changed files stay consistent with the repository structure,
4. new assumptions are written into README comments or docstrings if needed.
