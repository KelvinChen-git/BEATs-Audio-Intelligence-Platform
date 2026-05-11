$ErrorActionPreference = "Continue"

$Failures = 0

function Report-Check {
    param(
        [string]$Name,
        [bool]$Passed
    )

    if ($Passed) {
        Write-Host "PASS $Name"
    } else {
        Write-Host "FAIL $Name" -ForegroundColor Red
        $script:Failures += 1
    }
}

Write-Host "Running GPU preflight checks..."
Write-Host ""

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi
    Report-Check "nvidia-smi" ($LASTEXITCODE -eq 0)
} else {
    Report-Check "nvidia-smi" $false
}

Write-Host ""
$EnvOutput = & py scripts\check_env.py
$EnvStatus = $LASTEXITCODE
$EnvText = $EnvOutput -join "`n"
$EnvOutput | ForEach-Object { Write-Host $_ }

Write-Host ""
Report-Check "py scripts\check_env.py" ($EnvStatus -eq 0)
Report-Check "CUDA available is True" ($EnvText -match "CUDA available:\s*True")
Report-Check "train.py selected device is cuda" ($EnvText -match "train\.py selected device:\s*cuda")
Report-Check "checkpoint exists" (Test-Path ".\checkpoints\BEATs_iter3_plus_AS2M.pt")
Report-Check "ESC-50 audio directory exists" (Test-Path ".\data\ESC-50-master\audio")
Report-Check "ESC-50 metadata exists" (Test-Path ".\data\ESC-50-master\meta\esc50.csv")
Report-Check "official BEATs source exists" (Test-Path ".\third_party\unilm\beats\BEATs.py")

if ($Failures -gt 0) {
    Write-Host ""
    Write-Host "GPU preflight failed with $Failures issue(s). Fix these before strong fine-tuning." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "GPU preflight passed."
exit 0
