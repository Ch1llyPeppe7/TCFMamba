# Windows 环境配置（TCFMamba）

Linux / macOS 建议使用项目根目录的 **`scripts/setup_env.sh`** 一键配置，见 [INSTALL.md](INSTALL.md)。  
本文档仅针对 **Windows** 下无法使用该脚本时的环境配置。

> **Python**：Windows 下 mamba-ssm 预编译 wheel 多为 **Python 3.10**（`cp310`），部分来源支持 3.11/3.12，请以所选 wheel 为准。

---

## 前置

- Windows 10/11 (64-bit)，NVIDIA GPU
- 已安装 Miniconda/Anaconda
- Python 3.10（或与所选 wheel 一致）
- CUDA 11.8 或 12.1（与 PyTorch 版本匹配）

---

## 快速安装

```powershell
# 1. 创建环境
conda create -n tcfmamba python=3.10 -y
conda activate tcfmamba

# 2. PyTorch（按本机 CUDA 选）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 其他依赖
pip install -r requirements.txt

# 4. mamba-ssm / causal-conv1d：使用预编译 wheel（见下表）
# 示例（Python 3.10 + CUDA 11.8）：
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/causal_conv1d-1.1.1-cp310-cp310-win_amd64.whl
pip install https://huggingface.co/FuouM/mamba-ssm-windows-builds/resolve/main/mamba_ssm-1.1.3-cp310-cp310-win_amd64.whl

# 5. 可编辑安装
pip install -e .

# 6. 验证
python -c "import torch; from tcfmamba import TCFMamba; print('OK')"
```

---

## 预编译 Wheel 来源（参考）

| Python | CUDA | 来源说明 |
|--------|------|----------|
| 3.10   | 11.8 | [FuouM / mamba-ssm-windows-builds](https://huggingface.co/FuouM/mamba-ssm-windows-builds) |
| 3.12   | 12.1 | 部分仓库提供 torch2.x+cu121 的 Windows wheel，需自行搜索 |
| 3.10   | 11.8 | [divertingPan / mamba-for-windows](https://github.com/divertingPan/mamba-for-windows) |

具体 URL 可能随版本更新，请以各仓库最新说明为准。

---

## 常见问题

### CUDA_HOME

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
```

### PyTorch 与 CUDA 版本不一致

```powershell
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

保证 PyTorch 的 CUDA 与驱动/本机 CUDA 一致（如均为 11.8 或 12.1）。

### 仍无法安装 mamba-ssm

- 使用 WSL2，在 Linux 下按 [INSTALL.md](INSTALL.md) 用 `scripts/setup_env.sh` 配置。
- 或在 Windows 下仅安装除 mamba-ssm 外的依赖，运行时会回退到项目内纯 PyTorch 的 Mamba 实现（若已实现 fallback）。

---

## 实验与日志

- 实验配置：`config/experiment/experiment.yaml`
- 日志级别：命令行 `--state DEBUG` 或 `--state ERROR` 覆盖 yaml 中的 `state`。

---

**测试环境示例**：Windows 11，Python 3.10，CUDA 11.8，PyTorch 2.2.0
