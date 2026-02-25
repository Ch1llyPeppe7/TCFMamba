# TCFMamba Installation Guide

Installation guide for Linux, macOS, and Windows.

## Quick Start (Conda - Recommended)

```bash
# Clone and setup
git clone https://github.com/yourusername/TCFMamba.git
cd TCFMamba
conda env create -f environment.yml
conda activate tcfmamba

# Verify
python -c "from tcfmamba import TCFMamba; print('✓ TCFMamba ready')"
```

## Platform-Specific Installation

### Linux (Ubuntu/Debian)

```bash
# 1. System dependencies
sudo apt update && sudo apt install -y build-essential git

# 2. Create conda environment
conda create -n tcfmamba python=3.10 -y
conda activate tcfmamba

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify
python -c "import torch; from tcfmamba import TCFMamba; print('✓ Ready')"
```

### macOS

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Create conda environment
conda create -n tcfmamba python=3.10 -y
conda activate tcfmamba

# 3. Install PyTorch (CPU version for Mac)
pip install torch==2.2.0

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify
python -c "import torch; from tcfmamba import TCFMamba; print('✓ Ready')"
```

### Windows

⚠️ **Important**: Windows requires special handling for mamba-ssm.

See detailed guide: [WINDOWS_INSTALL.md](WINDOWS_INSTALL.md)

```powershell
# Quick install (PowerShell)
conda create -n tcfmamba python=3.10 -y
conda activate tcfmamba
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install mamba-ssm from prebuilt wheel
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/causal_conv1d-1.1.1-cp310-cp310-win_amd64.whl
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/mamba_ssm-1.1.3-cp310-cp310-win_amd64.whl

# Install other dependencies
pip install recbole einops numpy pandas scipy tensorboard wandb

# Verify
python -c "import torch; from tcfmamba import TCFMamba; print('✓ Ready')"
```

## Troubleshooting

### CUDA Version Mismatch

```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# Reinstall PyTorch with matching CUDA
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

### mamba-ssm Import Error (Windows)

- Use prebuilt wheels (see WINDOWS_INSTALL.md)
- Or use WSL2 for native Linux environment

### Dataset Not Found

```bash
python utils/prepare_datasets.py --verify
python utils/convert_gowalla.py --input raw_data/gowalla --output dataset/gowalla
```

## Docker (Optional)

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
CMD ["python", "utils/train.py", "--help"]
```

Build and run:
```bash
docker build -t tcfmamba .
docker run --gpus all -it -v $(pwd)/dataset:/workspace/dataset tcfmamba
```

## Next Steps

1. Prepare datasets: `python utils/prepare_datasets.py --verify`
2. Quick test: `python utils/train.py --model=TCFMamba --dataset=gowalla --config=config/tcfmamba_gowalla.yaml --epochs=10`
3. Full training: See [README.md](README.md)

## Getting Help

- Windows issues: [WINDOWS_INSTALL.md](WINDOWS_INSTALL.md)
- General issues: Open GitHub issue with OS, Python, PyTorch, CUDA versions and error message
