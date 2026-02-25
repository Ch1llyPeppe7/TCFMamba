# Windows Installation Guide for TCFMamba

## Prerequisites

- Windows 10/11 (64-bit) + NVIDIA GPU
- Python 3.10/3.11 + CUDA 11.8/12.1

## Quick Install

```powershell
# 1. Create conda environment
conda create -n tcfmamba python=3.10 -y
conda activate tcfmamba

# 2. Install PyTorch with CUDA 11.8
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install core dependencies
pip install recbole>=1.2.0 einops numpy pandas scipy tensorboard wandb

# 4. Install mamba-ssm from prebuilt wheel
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/causal_conv1d-1.1.1-cp310-cp310-win_amd64.whl
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/mamba_ssm-1.1.3-cp310-cp310-win_amd64.whl

# 5. Verify
python -c "import torch; from tcfmamba import TCFMamba; print('✓ Ready')"
```

## Prebuilt Wheel Options

| Python | CUDA | Wheel Source |
|--------|------|--------------|
| 3.10/3.11 | 11.8 | [FuouM](https://huggingface.co/FuouM/mamba-ssm-windows-builds) |
| 3.12 | 12.1 | [kurogane](https://huggingface.co/kurogane/mamba-causal-conv1d-win-build-torch2.9.1-cu128) |
| 3.10 | 11.8 | [divertingpan](https://github.com/divertingPan/mamba-for-windows) |

## Troubleshooting

### CUDA_HOME not set
```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
```

### CUDA/PyTorch mismatch
```powershell
python -c "import torch; print(torch.version.cuda)"
nvcc --version
# Ensure versions match (both 11.8 or both 12.1)
```

### Alternative: WSL2
If native Windows fails, use WSL2:
```powershell
wsl --install -d Ubuntu-22.04
# Then follow Linux install in INSTALL.md
```

## Full Install Script

```powershell
# Save as install_windows.ps1 and run
conda create -n tcfmamba python=3.10 -y
conda activate tcfmamba
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install recbole einops numpy pandas scipy tensorboard wandb
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/causal_conv1d-1.1.1-cp310-cp310-win_amd64.whl
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/mamba_ssm-1.1.3-cp310-cp310-win_amd64.whl
python -c "import torch; from tcfmamba import TCFMamba; print('✓ Ready')"
```

---

**Tested On**: Windows 11, Python 3.10, CUDA 11.8, PyTorch 2.2.0
