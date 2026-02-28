#!/usr/bin/env bash
# TCFMamba 一键环境配置（从创建 conda 环境到可运行）
# 用法: bash scripts/setup_env.sh  或  ./scripts/setup_env.sh
# 依赖: 已安装 Miniconda/Anaconda，且 conda 已初始化（conda init bash）
#
# 环境变量（可选）:
#   TCFMAMBA_ENV_NAME     conda 环境名，默认 mamba
#   PYTHON_VERSION        Python 版本，默认 3.12
#   RECREATE_ENV=1        非交互下也强制删除并重建环境
#   CUDA_VERSION          指定 cuda-nvcc 版本：12.8 或 11.8（mamba_ssm/causal_conv1d 支持）
#   TORCH_INDEX           强制指定 PyTorch index-url，覆盖自动检测
#   USE_PYTORCH_NIGHTLY=1 强制使用 PyTorch nightly（cu128）
#   INSTALL_CUDA_NVCC=1   在 Linux 下通过 conda 安装 cuda-nvcc（适配新架构如 RTX 5090）

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${TCFMAMBA_ENV_NAME:-tcfmamba}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RECREATE_ENV="${RECREATE_ENV:-0}"

echo "[TCFMamba] 项目目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 若已存在同名环境：非交互下保留；交互下询问是否重建
if conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "[TCFMamba] 检测到已存在 conda 环境: $ENV_NAME"
  if [[ "$RECREATE_ENV" == "1" ]]; then
    echo "[TCFMamba] RECREATE_ENV=1，删除并重建环境。"
    conda env remove -n "$ENV_NAME" -y
  elif [[ -t 0 ]]; then
    read -p "是否删除并重建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      conda env remove -n "$ENV_NAME" -y
    else
      echo "[TCFMamba] 保留现有环境，仅安装/更新依赖。"
    fi
  else
    echo "[TCFMamba] 非交互模式，保留现有环境并更新依赖。"
  fi
fi

# 创建环境（不存在时）
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "[TCFMamba] 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "[TCFMamba] 激活环境并安装依赖..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---------- 平台与显卡检测 ----------
OS="$(uname -s)"
ARCH="$(uname -m)"
GPU_NAME=""
CUDA_DRIVER=""
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  CUDA_DRIVER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]\+\.[0-9]*\).*/\1/p' | head -1 || true)"
fi

echo "[TCFMamba] 系统: $OS, 架构: $ARCH"
if [[ -n "$GPU_NAME" ]]; then
  echo "[TCFMamba] 检测到 GPU: $GPU_NAME (驱动 CUDA: ${CUDA_DRIVER:-未知})"
fi

# ---------- 1) 选择 PyTorch 安装方式 ----------
install_cuda_nvcc=0
if [[ -n "$TORCH_INDEX" ]]; then
  PYTORCH_INDEX="$TORCH_INDEX"
  echo "[TCFMamba] 使用指定 TORCH_INDEX: $PYTORCH_INDEX"
elif [[ "$USE_PYTORCH_NIGHTLY" == "1" ]]; then
  PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
  echo "[TCFMamba] 使用 PyTorch nightly (cu128)"
  [[ "$OS" == "Linux" && -n "$GPU_NAME" ]] && install_cuda_nvcc=1
elif [[ "$OS" == "Linux" && -n "$GPU_NAME" ]]; then
  # 新架构（50 系 / Blackwell）或驱动 CUDA >= 12.6 → nightly cu128 + 可选 cuda-nvcc
  if [[ "$GPU_NAME" =~ 50[0-9]{2} ]] || [[ "$GPU_NAME" =~ Blackwell ]] || \
     [[ -n "$CUDA_DRIVER" && "$(echo -e "${CUDA_DRIVER}\n12.6" | sort -V | head -1)" == "12.6" ]]; then
    PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
    echo "[TCFMamba] 检测到新架构/高版本 CUDA，使用 PyTorch nightly (cu128)"
    install_cuda_nvcc=1
  else
    # 稳定版：按驱动 CUDA 版本选 cu124 / cu121 / cu118
    if [[ -n "$CUDA_DRIVER" ]]; then
      if [[ "$(echo -e "${CUDA_DRIVER}\n12.4" | sort -V | head -1)" == "12.4" ]]; then
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
      elif [[ "$(echo -e "${CUDA_DRIVER}\n12.1" | sort -V | head -1)" == "12.1" ]]; then
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
      else
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
      fi
    else
      PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
    fi
    echo "[TCFMamba] 使用 PyTorch 稳定版: $PYTORCH_INDEX"
  fi
elif [[ "$OS" == "Darwin" ]]; then
  PYTORCH_INDEX=""
  echo "[TCFMamba] macOS：安装 PyTorch（默认含 MPS 支持）"
else
  PYTORCH_INDEX=""
  echo "[TCFMamba] 未检测到 NVIDIA GPU，安装 CPU 版 PyTorch"
fi

# 可选：通过 conda 安装 cuda-nvcc（mamba_ssm/causal_conv1d 常用 12.8 或 11.8，5090 等用 12.8）
if [[ "$INSTALL_CUDA_NVCC" == "1" ]] || [[ "$install_cuda_nvcc" -eq 1 ]]; then
  if [[ -n "$CUDA_VERSION" ]]; then
    CUDA_VER="$CUDA_VERSION"
  else
    # 按驱动选 12.8 或 11.8（仅支持这两种，与 mamba_ssm/causal_conv1d wheel 一致）
    if [[ -n "$CUDA_DRIVER" && "$CUDA_DRIVER" =~ ^11\. ]]; then
      CUDA_VER="11.8"
    else
      CUDA_VER="12.8"
    fi
  fi
  # 统一为 X.Y.Z 供 nvidia channel（12.8 -> 12.8.0, 11.8 -> 11.8.0）
  if [[ "$CUDA_VER" =~ ^[0-9]+\.[0-9]+$ ]]; then
    CUDA_VER="${CUDA_VER}.0"
  fi
  echo "[TCFMamba] 通过 conda 安装 cuda-nvcc ($CUDA_VER)..."
  if conda install -c "nvidia/label/cuda-${CUDA_VER}" cuda-nvcc -y 2>/dev/null; then
    echo "[TCFMamba] cuda-nvcc 安装完成"
  else
    echo "[TCFMamba] 跳过 cuda-nvcc（可用 CUDA_VERSION=12.8 或 11.8，或 INSTALL_CUDA_NVCC=0）"
  fi
fi

# 安装 PyTorch（nightly 用 --pre，稳定版不用）
if [[ -n "$PYTORCH_INDEX" ]]; then
  if [[ "$PYTORCH_INDEX" == *nightly* ]]; then
    pip install --pre torch torchvision torchaudio --index-url "$PYTORCH_INDEX"
  else
    pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"
  fi
else
  pip install torch torchvision torchaudio
fi

# ---------- 2) 项目 requirements ----------
echo "[TCFMamba] 安装 requirements.txt..."
pip install -r requirements.txt

# ---------- 3) Mamba SSM（仅 Linux；Windows 需另装预编译 wheel）----------
if [[ "$OS" == "Linux" ]]; then
  echo "[TCFMamba] 安装 causal-conv1d 与 mamba-ssm..."
  pip install "causal-conv1d>=1.1.0" "mamba-ssm>=1.1.0"
elif [[ "$OS" == "Darwin" ]]; then
  echo "[TCFMamba] macOS：尝试安装 mamba-ssm（部分版本可能需从源码编译）..."
  pip install "causal-conv1d>=1.1.0" "mamba-ssm>=1.1.0" 2>/dev/null || echo "[TCFMamba] 跳过 mamba-ssm（可选）"
else
  echo "[TCFMamba] 非 Linux，跳过 mamba-ssm（Windows 请见 WINDOWS_INSTALL.md）"
fi

# ---------- 4) 可编辑安装本项目 ----------
echo "[TCFMamba] 可编辑安装当前项目..."
pip install -e .

echo "[TCFMamba] 环境就绪。激活命令: conda activate $ENV_NAME"
echo "[TCFMamba] 运行示例: python utils/train.py --model=TCFMamba --dataset=gowalla"
