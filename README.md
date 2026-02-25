# 🚀 TCFMamba：Trajectory Collaborative Filtering Mamba for Debiased POI Recommendation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![RecBole](https://img.shields.io/badge/RecBole-1.2+-green.svg)](https://recbole.io/)
[![CIKM 2025](https://img.shields.io/badge/CIKM-2025-orange.svg)](https://doi.org/10.1145/3746252.3761175)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 📄 **Official implementation** of our paper published at **[CIKM 2025](https://doi.org/10.1145/3746252.3761175)** 
---

## 📋 Overview

**TCFMamba** is a novel POI (Point-of-Interest) recommendation model published at **[CIKM 2025](https://doi.org/10.1145/3746252.3761175)**. It addresses **popularity bias**, **exposure bias**, and **limited representational capacity** issues in location-based social networks ([LBSNs](https://en.wikipedia.org/wiki/Location-based_service)).

### 🧩 Core Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| **JLSDR** | Joint Learning of Static and Dynamic Representations | 🗺️ Geographic coordinates + 👤 Dynamic user preferences |
| **PSMN** | Preference State Mamba Network | ⚡ Linear-time sequential encoder |

### ✨ Key Features

- 🎯 **Debiased Recommendation** - Mitigates popularity bias through JLSDR's balanced representation learning
- 🗺️ **Spatial-Temporal Awareness** - Captures geographic and temporal patterns
- ⚡ **Linear-Time Complexity** - Leverages [Mamba SSM](https://github.com/state-spaces/mamba) for efficient modeling
- 🔧 **RecBole Integration** - Seamless integration with [RecBole](https://recbole.io/) framework

## ⚙️ Installation

### 🚀 Quick Install (Conda - Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/TCFMamba.git
cd TCFMamba

# Create conda environment
conda env create -f environment.yml
conda activate tcfmamba

# Or use pip
pip install -r requirements.txt
```

### Platform-Specific Instructions

- **Linux/macOS**: See [docs/INSTALL.md](docs/INSTALL.md#linux-ubuntudebian)
- **Windows**: See [docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md) (requires special handling for mamba-ssm)
- **Docker**: See [docs/INSTALL.md](docs/INSTALL.md#docker-setup)

### Prerequisites

- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 or 12.1 (for GPU)
- **PyTorch**: 2.0+

### Key Dependencies

| Package | Version | Purpose | Platform Notes |
|---------|---------|---------|----------------|
| `torch` | ≥2.0.0 | Deep Learning | Match CUDA version |
| `recbole` | ≥1.2.0 | Recommendation Framework | All platforms |
| `causal-conv1d` | ≥1.1.0 | Mamba dependency | Linux/macOS native; Windows: prebuilt wheels |
| `mamba-ssm` | ≥1.1.0 | State Space Model | Linux/macOS native; Windows: prebuilt wheels |

**Note for Windows Users**: mamba-ssm requires prebuilt wheels on Windows. See [docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md) for detailed instructions and download links.

## 🚀 Quick Start

### 1️⃣ Train TCFMamba on a Single Dataset

```bash
# Train on Gowalla dataset
python utils/train.py \
    --model=TCFMamba \
    --dataset=gowalla \
    --config=config/tcfmamba_gowalla.yaml \
    --epochs=100

# Train on Foursquare Tokyo
python utils/train.py \
    --model=TCFMamba \
    --dataset=foursquare_TKY \
    --config=config/tcfmamba_tky.yaml

# Train on Foursquare NYC
python utils/train.py \
    --model=TCFMamba \
    --dataset=foursquare_NYC \
    --config=config/tcfmamba_nyc.yaml
```

### 2️⃣ Run Batch Experiments

```bash
# Run all experiments (TCFMamba on all datasets)
python utils/run_experiments.py --all

# Run specific dataset with multiple seeds
python utils/run_experiments.py \
    --dataset gowalla \
    --seeds 42 2023 2024

# Compare with baseline models
python utils/run_experiments.py \
    --all \
    --baselines BERT4Rec GRU4Rec SRGNN
```

### 3️⃣ Monitor Training with TensorBoard

```bash
# Enable TensorBoard logging
python utils/train.py \
    --model=TCFMamba \
    --dataset=gowalla \
    --config=config/tcfmamba_gowalla.yaml \
    --tensorboard

# View logs
tensorboard --logdir=saved/
```

## 📁 Project Structure

Following [RecBole's SOP](https://recbole.io/docs/user_guide/data/atomic_files.html) (Standard Operating Procedure):

```
TCFMamba/
├── tcfmamba/                    # Core model package (high cohesion)
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── tcfmamba.py          # Main TCFMamba model
│   └── layers/
│       ├── __init__.py
│       └── mamba_ssm.py         # Mamba SSM layer (model component)
├── config/                     # Configuration files
│   ├── tcfmamba_gowalla.yaml    # Gowalla config
│   ├── tcfmamba_tky.yaml        # Foursquare TKY config
│   ├── tcfmamba_nyc.yaml        # Foursquare NYC config
│   └── *.yaml                   # Baseline model configs (RecBole native)
├── utils/                       # Python utility scripts
│   ├── train.py                 # Command-line training entry
│   ├── run_experiments.py       # Batch experiment runner
│   └── prepare_datasets.py      # Dataset preparation helper
├── scripts/                     # Shell scripts
│   ├── quick_start.sh           # Linux quick start
│   └── quick_start.ps1          # Windows quick start
├── dataset/                     # Datasets (following RecBole SOP)
│   ├── gowalla/
│   │   ├── gowalla.inter        # User-POI interactions
│   │   └── gowalla.item         # POI features (lat, lon)
│   ├── foursquare_TKY/
│   └── foursquare_NYC/
├── requirements.txt
└── README.md
```

## 📊 Datasets

Following [RecBole's SOP](https://recbole.io/docs/user_guide/data/atomic_files.html), we use atomic file format for datasets.

### Supported Datasets

| Dataset | Users | POIs | Interactions | Type | Source |
|---------|-------|------|--------------|------|--------|
| Gowalla | ~107K | ~128K | ~1.3M | Check-in | [SNAP](https://snap.stanford.edu/data/loc-gowalla.html) |
| Foursquare TKY | ~2.3K | ~6.2K | ~574K | Check-in | [UoM](https://archive.org/details/201309_foursquare_dataset_umn) |
| Foursquare NYC | ~1.1K | ~4K | ~227K | Check-in | [UoM](https://archive.org/details/201309_foursquare_dataset_umn) |

### Dataset Preparation (RecBole SOP)

#### Step 1: Verify / Download Datasets

```bash
# Check if datasets are ready
python utils/prepare_datasets.py --verify

# Show download instructions
python utils/prepare_datasets.py --dataset gowalla
```

#### Step 2: Dataset Format (RecBole Atomic Files)

Datasets must follow RecBole's atomic file format:

**`.inter` file** (user-item interactions):
```
user_id:token	venue_id:token	timestamp:float
0	420315	1287411463.0
1	123456	1287411500.0
```

**`.item` file** (POI features with latitude/longitude):
```
venue_id:token	latitude:float	longitude:float
420315	30.2691029532	-97.7493953705
123456	40.74137425	-73.9881052167
```

**Required column suffixes** (RecBole convention):
- `:token` - Discrete/categorical features
- `:float` - Continuous numerical features

#### Step 3: Directory Structure (RecBole SOP)

```
dataset/
├── gowalla/                    # Dataset name must match config
│   ├── gowalla.inter           # Interactions file
│   └── gowalla.item            # Item features file
├── foursquare_TKY/
│   ├── foursquare_TKY.inter
│   └── foursquare_TKY.item
└── foursquare_NYC/
    ├── foursquare_NYC.inter
    └── foursquare_NYC.item
```

#### Step 4: Dataset Conversion

Use RecBole's official conversion tool ([RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets)):

```bash
# 1. Clone the conversion tools
git clone https://github.com/RUCAIBox/RecSysDatasets
cd RecSysDatasets/conversion_tools
pip install -r requirements.txt

# 2. Convert Gowalla
python run.py --dataset gowalla \
    --input_path /path/to/raw/gowalla \
    --output_path /path/to/output \
    --convert_inter --duplicate_removal

# 3. Copy converted files to project
mv /path/to/output/gowalla.inter /path/to/output/gowalla.item \
   your_project/dataset/gowalla/
```

For Foursquare and other datasets, see [RecSysDatasets usage guides](https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools/usage).

#### Quick Dataset Setup

```bash
# Create directory structure
python utils/prepare_datasets.py --all

# Verify datasets are ready
python utils/prepare_datasets.py --verify
```

### Why Not Include Datasets?

Following RecBole's best practices:
1. **Legal**: Dataset redistribution may violate terms of use
2. **Size**: Datasets are large (100MB+ each)
3. **Freshness**: Raw sources may be updated
4. **Transparency**: Users should download from official sources

We rely on RecBole's official [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) tools for dataset conversion.

## 🏗️ Model Architecture

### TCFMamba Components (CIKM 2025)

```python
# 1. JLSDR (Joint Learning of Static and Dynamic Representations)
#    - Static: Geographic coordinates (latitude/longitude)
#    - Dynamic: User preference evolution through CF signals
#    - Fusion: Unified representation for debiased recommendation

# 2. PSMN (Preference State Mamba Network)
#    - Mamba-based sequential encoder with linear complexity
#    - Captures long-range dependencies in user trajectories
#    - Residual connections + Feed-forward network
```

### Configuration Options

Key hyperparameters in config files:

```yaml
# Model Architecture
hidden_size: 64              # Embedding dimension
num_layers: 2                # Number of PSMN layers
dropout_prob: 0.2            # Dropout rate
loss_type: 'CE'              # 'CE' or 'BPR'

# Mamba Parameters
d_state: 32                  # SSM state dimension
d_conv: 4                    # Convolution width
expand: 2                    # Expansion factor

# Time Encoding
Term: 'day'                  # 'day', 'week', or 'month'

# Location Fields (auto-detected)
LATITUDE_FIELD: latitude
LONGITUDE_FIELD: longitude
```

## 🛠️ Advanced Usage

### Custom Training Parameters

```bash
python utils/train.py \
    --model=TCFMamba \
    --dataset=gowalla \
    --config=config/tcfmamba_gowalla.yaml \
    --learning_rate=0.0005 \
    --train_batch_size=2048 \
    --epochs=150 \
    --gpu_id=0,1
```

### Resume from Checkpoint

```bash
python utils/train.py \
    --model=TCFMamba \
    --dataset=gowalla \
    --config=config/tcfmamba_gowalla.yaml \
    --checkpoint=saved/TCFMamba-gowalla.pth
```

### Run Baseline Models

```bash
# BERT4Rec
python utils/train.py \
    --model=BERT4Rec \
    --dataset=gowalla \
    --config=config/BERT4Rec.yaml

# GRU4Rec
python utils/train.py \
    --model=GRU4Rec \
    --dataset=gowalla \
    --config=config/GRU4Rec.yaml
```


## 🔧 Troubleshooting


### Issue: Mamba SSM Import Error

**Solution**: Ensure `causal-conv1d` and `mamba-ssm` are installed:
```bash
pip install causal-conv1d>=1.1.0 mamba-ssm>=1.1.0
```

### Issue: Dataset Not Found

**Solution**: Verify dataset structure:
```bash
# Check dataset exists
ls dataset/gowalla/gowalla.inter
ls dataset/gowalla/gowalla.item
```

## 📖 Citation

If you use TCFMamba in your research, please cite:

```bibtex
@inproceedings{qian2025tcfmamba,
  author = {Qian, Jin and Song, Shiyu and Zhang, Xin and Wang, Dongjing and Weng, He and Zhang, Haiping and Yu, Dongjin},
  title = {TCFMamba: Trajectory Collaborative Filtering Mamba for Debiased Point-of-Interest Recommendation},
  year = {2025},
  isbn = {9798400720406},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746252.3761175},
  doi = {10.1145/3746252.3761175},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  pages = {2409–2419},
  numpages = {11},
  keywords = {debias, mamba, poi recommendation, spatial-temporal aware, trajectory collaborative filtering},
  location = {Seoul, Republic of Korea},
  series = {CIKM '25}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- 🎯 Built on [RecBole](https://recbole.io/) framework
- 🚀 Mamba implementation from [mamba-ssm](https://github.com/state-spaces/mamba)
- 📚 Dataset conversion tools from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets)

## 💬 Contact

- 🐛 **Bug Reports & Issues**: [GitHub Issues](../../issues)
- 💡 **Questions & Discussions**: [GitHub Discussions](../../discussions)
- 📧 **Email**: Contact authors through [ACM Digital Library](https://doi.org/10.1145/3746252.3761175)

---

<p align="center">
  ⭐ <b>Star this repo if you find it helpful!</b> ⭐
</p>

<p align="center">
  <b>Happy Researching! 🚀</b>
</p>
