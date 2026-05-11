$ErrorActionPreference = "Stop"

$CudaAvailable = py -c "import torch; print(torch.cuda.is_available())"
if ($CudaAvailable.Trim() -ne "True") {
    Write-Warning "CUDA is not available. Strong fine-tuning will run on CPU."
    Write-Warning "Estimated CPU time: about 5 hours per 20-epoch fold-1 run in this workspace."
    Write-Warning "This script runs three fold-1 experiments, so CPU runtime may be well over 15 hours."
    $Answer = Read-Host "Type YES to continue on CPU"
    if ($Answer -ne "YES") {
        Write-Host "Cancelled. No training was started."
        exit 1
    }
}

$Checkpoint = ".\checkpoints\BEATs_iter3_plus_AS2M.pt"
$CommonArgs = @(
    "-m", "src.train",
    "--model", "official_beats_strong_ft",
    "--protocol", "esc50_standard_cv",
    "--fold", "1",
    "--epochs", "20",
    "--batch-size", "2",
    "--checkpoint-path", $Checkpoint,
    "--head-lr", "5e-4",
    "--weight-decay", "0.01",
    "--label-smoothing", "0.1",
    "--grad-clip", "1.0",
    "--scheduler", "cosine",
    "--warmup-ratio", "0.05",
    "--amp"
)

$Runs = @(
    @{
        Name = "top8_esc50_standard_cv"
        Extra = @("--unfreeze-top-layers", "8", "--backbone-lr", "1e-5")
    },
    @{
        Name = "top12_esc50_standard_cv"
        Extra = @("--unfreeze-top-layers", "12", "--backbone-lr", "1e-5")
    },
    @{
        Name = "full_esc50_standard_cv"
        Extra = @("--full-finetune", "--backbone-lr", "5e-6")
    }
)

foreach ($Run in $Runs) {
    $OutputDir = ".\outputs\strong_ft_priority\$($Run.Name)"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Host "Running fold-1 priority experiment: $($Run.Name)"
    $Args = $CommonArgs + @("--output-dir", $OutputDir) + $Run.Extra
    py @Args
}
