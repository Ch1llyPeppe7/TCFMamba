"""
TCFMamba: Trajectory Collaborative Filtering Mamba for Debiased POI Recommendation

Setup script for pip installation.
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and platform-specific markers
            if line and not line.startswith("#") and not line.startswith("--"):
                lines.append(line)
        return lines

# Core requirements (without platform-specific mamba-ssm)
core_requirements = [
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TCFMamba",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "linux": [
            "causal-conv1d>=1.1.0",
            "mamba-ssm>=1.1.0",
        ],
        "windows": [
            # Windows users should install prebuilt wheels manually
            # See WINDOWS_INSTALL.md
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "tcfmamba-train=scripts.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
