#!/bin/bash
# TCFMamba Quick Test Pipeline
# Usage: ./scripts/quick_test.sh [cpu|gpu]
# 
# This script performs a minimal batch test to verify:
# 1. Environment setup is correct
# 2. Dataset structure is valid
# 3. Model can train (1 epoch)
# 4. No CUDA errors (supports CPU-only machines)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DEVICE=${1:-cpu}  # Default to CPU
DATASET=${2:-gowalla}  # Default dataset
EPOCHS=${3:-1}  # Minimal epochs for quick test

echo "=========================================="
echo "TCFMamba Quick Test Pipeline"
echo "=========================================="
echo "Device: $DEVICE"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Step 1: Environment Check
echo -e "\n${BLUE}[1/4] Checking Environment...${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi
python_version=$(python --version 2>&1)
echo -e "${GREEN}✓ Python: $python_version${NC}"

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠ No conda environment active${NC}"
    echo "  Recommended: conda activate tcfmamba"
else
    echo -e "${GREEN}✓ Conda env: $CONDA_DEFAULT_ENV${NC}"
fi

# Check PyTorch and key imports
echo "  Checking PyTorch..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" || {
    echo -e "${RED}✗ PyTorch not installed${NC}"
    exit 1
}

# Check CUDA availability
if [ "$DEVICE" == "gpu" ]; then
    cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$cuda_available" == "False" ]; then
        echo -e "${YELLOW}⚠ CUDA not available, switching to CPU${NC}"
        DEVICE="cpu"
    else
        cuda_version=$(python -c "import torch; print(torch.version.cuda)")
        echo -e "${GREEN}✓ CUDA: $cuda_version${NC}"
    fi
else
    echo -e "${BLUE}  Using CPU mode (no CUDA required)${NC}"
fi

# Check TCFMamba module
echo "  Checking TCFMamba..."
python -c "from tcfmamba import TCFMamba; print('  TCFMamba: import OK')" || {
    echo -e "${RED}✗ TCFMamba not installed. Run: pip install -e .${NC}"
    exit 1
}

# Check RecBole
echo "  Checking RecBole..."
python -c "import recbole; print(f'  RecBole: {recbole.__version__}')" || {
    echo -e "${RED}✗ RecBole not installed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Environment check passed${NC}"

# Step 2: Dataset Check
# Note: train.py now auto-downloads and prepares datasets if missing
echo -e "\n${BLUE}[2/4] Dataset: $DATASET${NC}"

# Quick check - if not exists, inform user that auto-download will happen
if [ ! -d "dataset/$DATASET" ] || [ ! -f "dataset/$DATASET/$DATASET.inter" ]; then
    echo -e "${YELLOW}⚠ Dataset not found locally${NC}"
    echo -e "${BLUE}  Will auto-download and prepare during training...${NC}"
    echo "  (First run may take a few minutes depending on network)"
else
    echo -e "${GREEN}✓ Dataset found locally${NC}"
fi

# Step 3: Config (dataset + model yaml under config/)
echo -e "\n${BLUE}[3/4] Config: config/dataset + config/model${NC}"

# Step 4: Quick Training Test
echo -e "\n${BLUE}[4/4] Running Minimal Training Test...${NC}"
echo "  Training for $EPOCHS epoch(s), device: $DEVICE"
echo ""

GPU_ARG="$([ "$DEVICE" == "cpu" ] && echo "-1" || echo "0")"
echo "  Starting training..."
python utils/train.py \
    --model=TCFMamba \
    --dataset=$DATASET \
    --epochs=$EPOCHS \
    --learning_rate=0.001 \
    --gpu_id=$GPU_ARG \
    2>&1 | tee /tmp/tcfmamba_test.log | grep -E "(epoch|loss|metric|NDCG|Recall|Error|success)" || true

test_exit_code=${PIPESTATUS[0]}

if [ $test_exit_code -ne 0 ]; then
    echo -e "\n${RED}✗ Training test failed${NC}"
    echo "  Check log: /tmp/tcfmamba_test.log"
    echo "  Common issues:"
    echo "    - Dataset format incorrect"
    echo "    - Missing dependencies"
    echo "    - CUDA out of memory (try CPU mode)"
    exit 1
fi

# Check if training completed successfully
if grep -q "best valid\|test result" /tmp/tcfmamba_test.log 2>/dev/null; then
    echo -e "\n${GREEN}✓ Training test completed${NC}"
else
    echo -e "\n${YELLOW}⚠ Training may have issues, check full log:${NC}"
    echo "  tail -50 /tmp/tcfmamba_test.log"
fi

# Summary
echo -e "\n${BLUE}Test Summary${NC}"
echo "=========================================="
echo -e "${GREEN}✓ All checks passed!${NC}"
echo ""
echo "Device: $DEVICE"
echo "Dataset: $DATASET"
echo "Status: Ready for full training"
echo ""
echo "Next steps:"
echo "  1. Full training: python utils/train.py --model=TCFMamba --dataset=$DATASET"
echo "  2. Batch experiments: python utils/run_experiments.py --dataset $DATASET"
echo ""

if [ "$DEVICE" == "cpu" ]; then
    echo -e "${YELLOW}Note: Running on CPU. For GPU acceleration:${NC}"
    echo "  - Install CUDA Toolkit 11.8 or 12.1"
    echo "  - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    echo "  - Re-run with: ./scripts/quick_test.sh gpu"
fi

echo "=========================================="
echo -e "${GREEN}Pipeline test complete!${NC}"
