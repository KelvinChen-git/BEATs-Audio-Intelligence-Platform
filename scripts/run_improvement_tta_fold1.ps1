$ErrorActionPreference = "Stop"

$ReplicationDir = ".\outputs\replication_standard_cv\full_lr3e6_fold1"
$CheckpointPath = Join-Path $ReplicationDir "final_official_beats_strong_ft_fold_1.pt"

if (-not (Test-Path $CheckpointPath)) {
    Write-Host "Missing replicated fold-1 checkpoint: $CheckpointPath" -ForegroundColor Yellow
    Write-Host "Run scripts\run_replication_trainval_fold1.ps1 first."
    exit 1
}

py -m src.train `
    --model official_beats_strong_ft `
    --protocol esc50_standard_cv `
    --use-trainval-for-final `
    --fold 1 `
    --epochs 17 `
    --batch-size 2 `
    --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt `
    --full-finetune `
    --backbone-lr 3e-6 `
    --head-lr 5e-4 `
    --weight-decay 0.01 `
    --label-smoothing 0.1 `
    --grad-clip 1.0 `
    --scheduler cosine `
    --warmup-ratio 0.05 `
    --amp `
    --eval-only-checkpoint $CheckpointPath `
    --tta `
    --tta-num-views 5 `
    --tta-shift-samples 1600 `
    --output-dir .\outputs\improvement_tta\fold1
