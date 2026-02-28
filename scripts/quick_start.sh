#!/bin/bash
# TCFMamba Quick Start Script
# This script provides quick commands to reproduce TCFMamba experiments

set -e

echo "=========================================="
echo "TCFMamba Quick Start"
echo "=========================================="

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment detected."
    echo "Please activate your environment: conda activate tcfmamba"
    echo ""
fi

# Function to train TCFMamba
train_tcfmamba() {
    local dataset=$1
    echo "Training TCFMamba on ${dataset}..."
    python utils/train.py --model=TCFMamba --dataset=${dataset} --epochs=100
}

# Function to run all experiments
run_all() {
    echo "Running all experiments..."
    python utils/run_experiments.py --all
}

# Parse command line arguments
case "$1" in
    gowalla)
        train_tcfmamba "gowalla"
        ;;
    tky|foursquare_TKY)
        train_tcfmamba "foursquare_TKY"
        ;;
    nyc|foursquare_NYC)
        train_tcfmamba "foursquare_NYC"
        ;;
    all)
        run_all
        ;;
    test)
        echo "Running quick test (10 epochs on gowalla)..."
        python utils/train.py --model=TCFMamba --dataset=gowalla --epochs=10
        ;;
    help|--help|-h)
        echo "Usage: ./scripts/quick_start.sh [command]"
        echo ""
        echo "Commands:"
        echo "  gowalla          Train on Gowalla dataset"
        echo "  tky              Train on Foursquare Tokyo dataset"
        echo "  nyc              Train on Foursquare NYC dataset"
        echo "  all              Run all experiments"
        echo "  test             Quick test (10 epochs)"
        echo "  help             Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./scripts/quick_start.sh gowalla"
        echo "  ./scripts/quick_start.sh test"
        echo "  ./scripts/quick_start.sh all"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './scripts/quick_start.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "Done!"
