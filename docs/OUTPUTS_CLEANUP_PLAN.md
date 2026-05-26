# Outputs Cleanup Plan

This plan organizes generated artifacts without deleting final evidence. Cleanup must use `scripts/cleanup_outputs.py`; default mode is dry-run and permanent deletion is never used.

## Current Inventory

The `outputs/` folder currently contains:

- Final summary files: `final_two_line_summary.csv`, `final_two_line_summary.png`, `final_two_line_summary_metrics.png`
- TTA ablation files: `tta_ablation_fold1.csv`, `tta_ablation_fold1.png`, and `tta_ablation/fold1_views{1,3,5,7,9}/`
- Final reproduced evidence: `replication_standard_cv/full_lr3e6_fold1/`
- Historical root-level baseline, frozen BEATs, old improved BEATs, and early strong fine-tuning metrics/checkpoints
- Historical figures: `model_comparison_5fold.png`, `beats_foldwise_comparison.png`
- Development/debug folders: `debug_json_protocol_top8_1epoch/`, `debug_standard_cv_top8_1epoch/`, `strong_ft_priority/`
- Earlier TTA helper output: `improvement_tta/fold1/`

## Keep In `outputs/`

Final report required:

- `outputs/final_two_line_summary.csv`
- `outputs/final_two_line_summary.png`
- `outputs/final_two_line_summary_metrics.png` (optional supporting plot)
- `outputs/tta_ablation_fold1.csv`
- `outputs/tta_ablation_fold1.png`

Final result evidence:

- `outputs/replication_standard_cv/full_lr3e6_fold1/`
- `outputs/tta_ablation/fold1_views1/`
- `outputs/tta_ablation/fold1_views3/`

These files support the reproduced BEATs result and the selected TTA-3 improvement.

## Archive Candidates

Optional ablation evidence:

- `outputs/tta_ablation/fold1_views5/`
- `outputs/tta_ablation/fold1_views7/`
- `outputs/tta_ablation/fold1_views9/`

Historical or development outputs:

- Root-level baseline, frozen official BEATs, old improved BEATs, and early strong fine-tuning metrics/checkpoints
- `outputs/model_comparison_5fold.csv`
- `outputs/model_comparison_5fold.png`
- `outputs/beats_foldwise_comparison.png`
- `outputs/debug_json_protocol_top8_1epoch/`
- `outputs/debug_standard_cv_top8_1epoch/`
- `outputs/improvement_tta/`
- Fixed/development folders under `outputs/strong_ft_priority/`

The old frozen-backbone result around 69.45% and the old improved result around 91.55% are historical only. The final project comparison is reproduced BEATs full fine-tuning versus TTA-3 applied to the reproduced checkpoint.

## Invalid Output

Archive and mark as invalid:

- `outputs/strong_ft_priority/top8_esc50_standard_cv/`

This folder came from the pre-fix AMP run with near-random performance and should not be used as evidence.

## Safety Rules

- Do not delete final result files.
- Do not delete datasets, checkpoints, manifests, or vendored BEATs code.
- Move archive candidates only into `outputs_archive/<timestamp>/`.
- Preserve original folder structure under the archive.
- Review `outputs_cleanup_manifest.csv` before applying cleanup.

## Cleanup Commands

Dry-run only:

```powershell
py scripts\cleanup_outputs.py
```

Apply later, only after reviewing the dry-run:

```powershell
py scripts\cleanup_outputs.py --apply
```

Check final required outputs:

```powershell
py scripts\check_final_outputs.py
```
