Use this repository as the current project.

Goal:
Integrate the official Microsoft BEATs code from `third_party/unilm/beats` into this ESC-50 coursework project and make the `official_beats` training path run successfully.

Constraints:
- Do not write coursework report text.
- Keep the 5-fold manifest protocol exactly as-is.
- Keep all commands Windows-friendly for VS Code PowerShell.
- Prefer minimal, high-confidence edits.
- If a blocker exists, explain it clearly and propose the smallest next step.

Tasks:
1. Inspect the repo and summarize what is already implemented.
2. Verify that `src/beats_adapter.py` correctly imports and wraps the official BEATs code.
3. Fix any import-path, checkpoint-loading, waveform-shape, or device issues.
4. Keep `python -m src.train --model baseline --fold 1 --epochs 1 --batch-size 4` working.
5. Make `python -m src.train --model official_beats --fold 1 --epochs 1 --batch-size 2 --checkpoint-path .\checkpoints\YOUR_CHECKPOINT.pt --freeze-backbone` work once the checkpoint exists.
6. Add or update small tests if helpful.
7. At the end, tell me exactly which files changed and which command I should run next.

Verification commands:
- `pytest -q`
- `python -m src.train --model baseline --fold 1 --epochs 1 --batch-size 4`
- if checkpoint exists: `python -m src.train --model official_beats --fold 1 --epochs 1 --batch-size 2 --checkpoint-path .\checkpoints\YOUR_CHECKPOINT.pt --freeze-backbone`
