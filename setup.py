"""
TCFMamba: Trajectory Collaborative Filtering Mamba for Debiased POI Recommendation

Setup script for pip install -e . (可编辑安装).
完整依赖建议用: pip install -r requirements.txt，再根据平台安装 PyTorch / mamba-ssm，见 scripts/setup_env.sh
"""

from setuptools import setup, find_packages
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read_me():
    path = os.path.join(_HERE, "README.md")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "TCFMamba: Trajectory Collaborative Filtering Mamba for Debiased POI Recommendation"


def _parse_requirements(filename):
    path = os.path.join(_HERE, filename)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("--"):
                lines.append(line)
        return lines


# 与 requirements.txt 保持一致，作为单一来源
_install_requires = _parse_requirements("requirements.txt")
if not _install_requires:
    _install_requires = [
        "torch>=2.0.0",
        "recbole>=1.2.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "tabulate>=0.9.0",
        "tensorboard>=2.14.0",
        "wandb>=0.16.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
    ]

setup(
    name="tcfmamba",
    version="1.0.0",
    author="TCFMamba Authors",
    author_email="your.email@example.com",
    description="Trajectory Collaborative Filtering Mamba for Debiased POI Recommendation",
    long_description=_read_me(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TCFMamba",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=_install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "tcfmamba-train=utils.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
