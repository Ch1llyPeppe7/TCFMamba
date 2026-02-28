# TCFMamba 环境配置

适用于 Linux、macOS；Windows 见 [WINDOWS_INSTALL.md](WINDOWS_INSTALL.md)。

**环境要求**：Python 3.10+（推荐 3.12），PyTorch 2.0+，RecBole 1.2+。

---

## 一键配置（推荐）

项目提供 `scripts/setup_env.sh`，会创建 conda 环境、按系统/显卡选择 PyTorch 并安装依赖与可编辑包：

```bash
git clone https://github.com/yourusername/TCFMamba.git
cd TCFMamba
bash scripts/setup_env.sh
conda activate tcfmamba
```

脚本会：

- 创建 conda 环境 **`tcfmamba`**（默认 Python 3.12）
- 根据是否检测到 NVIDIA GPU、驱动 CUDA 版本、是否 50 系/Blackwell 等，自动选择：
  - PyTorch 稳定版（cu118/cu121/cu124）或 **nightly cu128**（新卡/高版本 CUDA）
  - 可选通过 conda 安装 **cuda-nvcc**（12.8 或 11.8，便于部分库编译）
- 安装 `requirements.txt` 及 **causal-conv1d**、**mamba-ssm**
- 执行 `pip install -e .`

**可选环境变量**（在运行 `bash scripts/setup_env.sh` 前设置）：

| 变量 | 说明 |
|------|------|
| `PYTHON_VERSION` | Python 版本，如 `3.12` |
| `CUDA_VERSION` | cuda-nvcc 版本：`12.8` 或 `11.8` |
| `USE_PYTORCH_NIGHTLY=1` | 强制使用 PyTorch nightly (cu128) |
| `TORCH_INDEX` | 强制指定 PyTorch index-url |
| `INSTALL_CUDA_NVCC=1` | 强制安装 conda 的 cuda-nvcc |
| `RECREATE_ENV=1` | 非交互下也删除并重建环境 |

**验证**：

```bash
python -c "import torch; from tcfmamba import TCFMamba; print('✓ TCFMamba ready')"
```

---

## 手动安装

未使用一键脚本时，可按以下步骤操作。

### Linux（示例）

```bash
conda create -n tcfmamba python=3.12 -y
conda activate tcfmamba

# PyTorch：按本机 CUDA 选择，见 https://pytorch.org
# 例如 CUDA 12.1：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 或 50 系 / CUDA 12.8 用 nightly：
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt
pip install -e .
```

### macOS

```bash
conda create -n tcfmamba python=3.12 -y
conda activate tcfmamba
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
```

macOS 下 `mamba-ssm` 可能需从源码编译，若安装失败可暂时跳过（仅影响 TCFMamba 的 Mamba 层）。

---

## 实验与日志配置

- 实验配置**仅一份**：`config/experiment/experiment.yaml`（含 `state`、`RUN_ID`、`monitoring_metrics`、`hparams` 等）。
- 日志级别用 **`state`** 控制：`INFO`（默认）、`DEBUG`（详细）、`ERROR`（安静）。可在命令行覆盖：
  - `python utils/train.py --model=TCFMamba --dataset=gowalla --state DEBUG`
  - `python utils/train.py --model=TCFMamba --dataset=gowalla --state ERROR`

---

## 常见问题

### CUDA / PyTorch 版本不一致

- 用 `nvidia-smi` 看驱动支持的 CUDA，再选对应 PyTorch index（cu118/cu121/cu124 或 nightly/cu128）。
- 若使用 50 系或 CUDA 12.8，优先用 nightly cu128；必要时设 `CUDA_VERSION=12.8` 并让脚本安装 cuda-nvcc。

### mamba-ssm 导入失败

- Linux/macOS：确认已安装 `causal-conv1d` 与 `mamba-ssm`（`pip install -r requirements.txt` 或一键脚本已包含）。
- Windows：需单独安装预编译 wheel，见 [WINDOWS_INSTALL.md](WINDOWS_INSTALL.md)。

### 数据集未找到

```bash
python utils/prepare_datasets.py --verify
python utils/prepare_datasets.py --dataset gowalla
```

---

## 可选：Docker

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
CMD ["python", "utils/train.py", "--help"]
```

```bash
docker build -t tcfmamba .
docker run --gpus all -it -v $(pwd)/dataset:/workspace/dataset tcfmamba
```

---

## 下一步

1. 准备数据：`python utils/prepare_datasets.py --dataset gowalla`
2. 训练：`python utils/train.py --model=TCFMamba --dataset=gowalla`
3. 更多用法见 [README.md](../README.md)
