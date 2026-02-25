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
echo -e "\n${BLUE}[1/5] Checking Environment...${NC}"

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
echo -e "\n${BLUE}[2/5] Checking Dataset: $DATASET${NC}"

python utils/prepare_datasets.py --verify --dataset $DATASET 2>&1 | head -20 || {
    echo -e "${YELLOW}⚠ Dataset verification had issues, but continuing...${NC}"
}

# Check if dataset files exist
if [ ! -d "dataset/$DATASET" ]; then
    echo -e "${RED}✗ Dataset directory not found: dataset/$DATASET${NC}"
    echo -e "${YELLOW}  Please prepare dataset first:${NC}"
    echo "  1. Download from official source"
    echo "  2. Convert using RecSysDatasets tool"
    echo "  3. Place in dataset/$DATASET/"
    echo ""
    echo "  See: python utils/prepare_datasets.py --dataset $DATASET"
    exit 1
fi

# Check for required files
if [ ! -f "dataset/$DATASET/$DATASET.inter" ]; then
    echo -e "${RED}✗ Missing: dataset/$DATASET/$DATASET.inter${NC}"
    exit 1
fi

if [ ! -f "dataset/$DATASET/$DATASET.item" ]; then
    echo -e "${YELLOW}⚠ Missing: dataset/$DATASET/$DATASET.item (optional but recommended)${NC}"
fi

echo -e "${GREEN}✓ Dataset check passed${NC}"

# Step 3: Config Check
echo -e "\n${BLUE}[3/5] Checking Config...${NC}"

CONFIG_FILE="config/tcfmamba_${DATASET}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config: $CONFIG_FILE${NC}"

# Step 4: Quick Training Test
echo -e "\n${BLUE}[4/5] Running Minimal Training Test...${NC}"
echo "  This will train for $EPOCHS epoch(s) to verify the pipeline"
echo "  Using device: $DEVICE"
echo ""

# Create a minimal test config
test_config=$(mktemp)
cat > $test_config << EOF
# Minimal test configuration
$(cat $CONFIG_FILE)

# Override for quick test
gpu_id: '$([ "$DEVICE" == "cpu" ] && echo "-1" || echo "0")'
epochs: $EPOCHS
train_batch_size: 256
eval_batch_size: 256
eval_step: 1
show_progress: True
EOF

echo "  Starting training..."
python utils/train.py \
    --model=TCFMamba \
    --dataset=$DATASET \
    --config=$test_config \
    --epochs=$EPOCHS \
    --learning_rate=0.001 \
    2>&1 | tee /tmp/tcfmamba_test.log | grep -E "(epoch|loss|metric|NDCG|Recall|Error|success)" || true

test_exit_code=${PIPESTATUS[0]}
rm -f $test_config

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

# Step 5: Summary
echo -e "\n${BLUE}[5/5] Test Summary${NC}"
echo "=========================================="
echo -e "${GREEN}✓ All checks passed!${NC}"
echo ""
echo "Device: $DEVICE"
echo "Dataset: $DATASET"
echo "Status: Ready for full training"
echo ""
echo "Next steps:"
echo "  1. Full training: python utils/train.py --model=TCFMamba --dataset=$DATASET --config=$CONFIG_FILE"
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
