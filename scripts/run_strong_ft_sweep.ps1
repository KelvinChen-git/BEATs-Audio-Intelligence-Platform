$ErrorActionPreference = "Stop"

$Checkpoint = ".\checkpoints\BEATs_iter3_plus_AS2M.pt"
$CommonArgs = @(
    "-m", "src.train",
    "--model", "official_beats_strong_ft",
    "--fold", "1",
    "--epochs", "20",
    "--batch-size", "2",
    "--checkpoint-path", $Checkpoint,
    "--weight-decay", "0.01",
    "--label-smoothing", "0.1",
    "--grad-clip", "1.0",
    "--scheduler", "cosine",
    "--warmup-ratio", "0.05",
    "--protocol", "json_split"
)

$Runs = @(
    @{
        Name = "top4_20ep"
        Extra = @("--unfreeze-top-layers", "4", "--backbone-lr", "5e-6", "--head-lr", "5e-4")
    },
    @{
        Name = "top8_20ep"
        Extra = @("--unfreeze-top-layers", "8", "--backbone-lr", "1e-5", "--head-lr", "5e-4")
    },
    @{
        Name = "top12_20ep"
        Extra = @("--unfreeze-top-layers", "12", "--backbone-lr", "5e-6", "--head-lr", "5e-4")
    },
    @{
        Name = "full_20ep"
        Extra = @("--full-finetune", "--backbone-lr", "2e-6", "--head-lr", "2e-4")
    }
)

foreach ($Run in $Runs) {
    $OutputDir = ".\outputs\strong_ft_sweep\$($Run.Name)"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Host "Running official_beats_strong_ft sweep config: $($Run.Name)"
    $Args = $CommonArgs + @("--output-dir", $OutputDir) + $Run.Extra
    py @Args
}
