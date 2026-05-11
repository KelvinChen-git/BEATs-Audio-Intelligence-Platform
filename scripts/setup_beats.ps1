$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ThirdParty = Join-Path $RepoRoot "third_party"
$UniLM = Join-Path $ThirdParty "unilm"

New-Item -ItemType Directory -Force -Path $ThirdParty | Out-Null

if (Test-Path $UniLM) {
    Write-Host "Official UniLM repo already exists at: $UniLM"
} else {
    Write-Host "Cloning official UniLM repo into third_party/unilm ..."
    git clone https://github.com/microsoft/unilm.git $UniLM
}

$BeatsPath = Join-Path $UniLM "beats"
if (-not (Test-Path $BeatsPath)) {
    throw "BEATs folder not found after clone: $BeatsPath"
}

Write-Host "Done. Official BEATs code is now available at: $BeatsPath"
Write-Host "Next: put a BEATs checkpoint under checkpoints/ and run the Codex prompt in prompts/codex_task_beats_integration.md"
