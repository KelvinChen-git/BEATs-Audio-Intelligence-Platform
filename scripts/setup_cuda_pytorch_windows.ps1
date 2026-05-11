$ErrorActionPreference = "Stop"

# Check https://pytorch.org/get-started/locally/ before changing this value.
# The official PyTorch install selector is the source of truth for CUDA wheels.
$CudaIndexUrl = "https://download.pytorch.org/whl/cu126"

Write-Host "Python executable:"
py -c "import sys; print(sys.executable)"

Write-Host ""
Write-Host "Current PyTorch package versions:"
py -c @'
import importlib.util

for name in ("torch", "torchaudio", "torchvision"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"{name}: not installed")
        continue
    module = __import__(name)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")

try:
    import torch
    print(f"torch CPU-only build: {'+cpu' in torch.__version__}")
except Exception as exc:
    print(f"torch CPU-only build: unknown ({exc})")
'@

Write-Host ""
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "nvidia-smi:"
    nvidia-smi
} else {
    Write-Warning "nvidia-smi was not found. Install/check NVIDIA drivers before CUDA PyTorch setup."
}

Write-Host ""
Write-Warning "Before installing, verify the CUDA wheel command with the official PyTorch selector: https://pytorch.org/get-started/locally/"
Write-Host "Default CUDA index URL: $CudaIndexUrl"
Write-Host "This script will uninstall torch, torchaudio, and torchvision before installing CUDA-enabled wheels."

$Confirm = Read-Host "Type YES to continue"
if ($Confirm -ne "YES") {
    Write-Host "Cancelled. No packages were changed."
    exit 1
}

py -m pip uninstall -y torch torchaudio torchvision
py -m pip install torch torchaudio torchvision --index-url $CudaIndexUrl

Write-Host ""
Write-Host "Post-install environment check:"
py scripts\check_env.py

$CudaAvailable = py -c "import torch; print(torch.cuda.is_available())"
if ($CudaAvailable.Trim() -eq "True") {
    Write-Host "CUDA is now available to PyTorch."
} else {
    Write-Warning "CUDA is still not available to PyTorch. Re-check drivers and the official PyTorch CUDA wheel selector."
    exit 1
}
