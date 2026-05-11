$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptRoot\gpu_preflight_check.ps1"
if ($LASTEXITCODE -ne 0) {
    throw "GPU preflight failed. Stopping before training."
}

py -m src.train `
    --model official_beats_strong_ft `
    --protocol esc50_standard_cv `
    --fold 1 `
    --epochs 20 `
    --batch-size 2 `
    --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt `
    --full-finetune `
    --backbone-lr 5e-6 `
    --head-lr 5e-4 `
    --weight-decay 0.01 `
    --label-smoothing 0.1 `
    --grad-clip 1.0 `
    --scheduler cosine `
    --warmup-ratio 0.05 `
    --amp `
    --output-dir .\outputs\strong_ft_priority\full_esc50_standard_cv
