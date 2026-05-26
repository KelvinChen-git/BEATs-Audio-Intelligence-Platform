$ErrorActionPreference = "Stop"

$CheckpointPath = ".\outputs\replication_standard_cv\full_lr3e6_fold1\final_official_beats_strong_ft_fold_1.pt"
if (-not (Test-Path $CheckpointPath)) {
    Write-Host "Missing reproduced BEATs checkpoint: $CheckpointPath" -ForegroundColor Yellow
    Write-Host "Run scripts\run_replication_trainval_fold1.ps1 first."
    exit 1
}

$ViewCounts = @(1, 3, 5, 7, 9)

foreach ($Views in $ViewCounts) {
    $OutputDir = ".\outputs\tta_ablation\fold1_views$Views"
    $Args = @(
        "-m", "src.train",
        "--model", "official_beats_strong_ft",
        "--protocol", "esc50_standard_cv",
        "--use-trainval-for-final",
        "--fold", "1",
        "--epochs", "17",
        "--batch-size", "2",
        "--checkpoint-path", ".\checkpoints\BEATs_iter3_plus_AS2M.pt",
        "--full-finetune",
        "--backbone-lr", "3e-6",
        "--head-lr", "5e-4",
        "--weight-decay", "0.01",
        "--label-smoothing", "0.1",
        "--grad-clip", "1.0",
        "--scheduler", "cosine",
        "--warmup-ratio", "0.05",
        "--amp",
        "--eval-only-checkpoint", $CheckpointPath,
        "--output-dir", $OutputDir
    )

    if ($Views -gt 1) {
        $Args += @("--tta", "--tta-num-views", "$Views")
    }

    Write-Host "Evaluating fold 1 with TTA views: $Views"
    & py @Args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
