# 🚀 TCFMamba

**Trajectory Collaborative Filtering Mamba for Debiased POI Recommendation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![RecBole 1.2+](https://img.shields.io/badge/RecBole-1.2+-green.svg)](https://recbole.io/)
[![CIKM 2025](https://img.shields.io/badge/CIKM-2025-orange.svg)](https://doi.org/10.1145/3746252.3761175)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 📄 **Official implementation** of our paper at **[CIKM 2025](https://doi.org/10.1145/3746252.3761175)**.

POI sequential recommendation with **Mamba SSM**: geographic & temporal encoding (**JLSDR**) and **PSMN** (Preference State Mamba Network) for debiased recommendation. Built on [RecBole](https://recbole.io/).

| 🗺️ Spatial-temporal | ⚡ Linear-time Mamba | 🔧 RecBole |
|----------------------|----------------------|------------|

---

## ⚙️ Installation

#### 🚀 One-line setup (recommended)

```bash
git clone https://github.com/yourusername/TCFMamba.git && cd TCFMamba
bash scripts/setup_env.sh
conda activate tcfmamba
```

The script creates the `tcfmamba` conda env, installs PyTorch (auto-detects OS/GPU, including nightly for 50-series / CUDA 12.8), dependencies, and the project in editable mode.

**Optional env vars:** `PYTHON_VERSION=3.12` · `CUDA_VERSION=12.8.0` · `USE_PYTORCH_NIGHTLY=1` · `TORCH_INDEX=...`

#### 📦 Manual

```bash
conda create -n tcfmamba python=3.12 -y && conda activate tcfmamba
# Install PyTorch for your platform: https://pytorch.org
pip install -r requirements.txt && pip install -e .
```

💡 **Windows:** install `mamba-ssm` from prebuilt wheels separately (see project docs).

---

## 🚀 Quick start

```bash
# Train (config: config/dataset/*.yaml + config/model/tcfmamba.yaml)
python utils/train.py --model=TCFMamba --dataset=gowalla

# Options: log level (--state DEBUG/ERROR), TensorBoard, rebuild dataset cache
python utils/train.py --model=TCFMamba --dataset=gowalla --state DEBUG
python utils/train.py --model=TCFMamba --dataset=gowalla --rebuild-dataset
```

📂 Data: place `{name}.inter` and `{name}.item` under `dataset/{name}/` (RecBole atomic format). Missing? Run `python utils/prepare_datasets.py --dataset gowalla` to download or generate.

---

## 📁 Structure

| Directory | Description |
|-----------|-------------|
| `tcfmamba/` | Model & Mamba layers |
| `config/` | Experiment, dataset, model configs |
| `utils/` | `train.py`, `prepare_datasets.py` |
| `scripts/` | `setup_env.sh` |
| `dataset/` | Dataset folders |

---

## 📖 Citation

```bibtex
@inproceedings{qian2025tcfmamba,
  author    = {Qian, Jin and Song, Shiyu and Zhang, Xin and Wang, Dongjing and Weng, He and Zhang, Haiping and Yu, Dongjin},
  title     = {TCFMamba: Trajectory Collaborative Filtering Mamba for Debiased Point-of-Interest Recommendation},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025},
  url       = {https://doi.org/10.1145/3746252.3761175}
}
```

---

[MIT License](LICENSE) · 🙏 [RecBole](https://recbole.io/) & [mamba-ssm](https://github.com/state-spaces/mamba)
