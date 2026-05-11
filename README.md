# BEATs + ESC-50 VS Code Starter

This folder is a **VS Code-ready starter workspace** for your INT306 coursework.

It includes:
- your group's 15 JSON split files under `datafiles/`
- an ESC-50 project layout under `data/ESC-50-master/`
- a baseline training pipeline in `src/`
- an **official BEATs adapter** in `src/beats_adapter.py`
- `AGENTS.md` and prompt files so **Codex** can continue the integration automatically
- `.vscode/` settings and tasks for quick running

## What you still need locally

1. Put the **full ESC-50 audio files** into:
   - `data/ESC-50-master/audio/`
2. Clone the official BEATs code into:
   - `third_party/unilm/`
3. Download a BEATs checkpoint and place it in:
   - `checkpoints/`

## Recommended setup order

1. Open this folder in VS Code.
2. Create/select your Python environment.
3. Run the PowerShell script:
   - `scripts/setup_beats.ps1`
4. Install Python packages:
   - `pip install -r requirements.txt`
5. Run the baseline first:
   - `python -m src.train --model baseline --fold 1`
6. Then ask Codex to finish the BEATs integration using the prompt in:
   - `prompts/codex_task_beats_integration.md`

## Example commands

### Baseline sanity check
```powershell
python -m src.train --model baseline --fold 1 --epochs 1 --batch-size 4
```

### Official BEATs path
```powershell
python -m src.train --model official_beats --fold 1 --epochs 3 --batch-size 4 --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt --freeze-backbone
```

## Notes
- The uploaded ESC-50 zip in chat was trimmed. This starter only includes `meta/esc50.csv`, not the full audio set.
- Fold 2 train manifest is named `esc_trainl_data_2.json`. The code handles this typo automatically.
- This project is for code and experiments only. Write the final report yourself.
