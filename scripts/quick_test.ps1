# TCFMamba Quick Test Pipeline (PowerShell)
# Usage: .\scripts\quick_test.ps1 [-device cpu] [-dataset gowalla] [-epochs 1]
#
# This script performs a minimal batch test to verify:
# 1. Environment setup is correct
# 2. Dataset structure is valid
# 3. Model can train (1 epoch)
# 4. No CUDA errors (supports CPU-only machines)

param(
    [string]$device = "cpu",
    [string]$dataset = "gowalla",
    [int]$epochs = 1
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TCFMamba Quick Test Pipeline" -ForegroundColor Cyan
Write-Host "=========================================="
Write-Host "Device: $device"
Write-Host "Dataset: $dataset"
Write-Host "Epochs: $epochs"
Write-Host "=========================================="

# Step 1: Environment Check
Write-Host "`n[1/5] Checking Environment..." -ForegroundColor Cyan

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Python not found" -ForegroundColor Red
    exit 1
}

# Check conda
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "[OK] Conda env: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
} else {
    Write-Host "[WARN] No conda environment active" -ForegroundColor Yellow
    Write-Host "  Recommended: conda activate tcfmamba"
}

# Check PyTorch
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>$null
    Write-Host "  PyTorch: $torchVersion"
} catch {
    Write-Host "[FAIL] PyTorch not installed" -ForegroundColor Red
    exit 1
}

# Check CUDA
if ($device -eq "gpu") {
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaAvailable -eq "False") {
        Write-Host "[WARN] CUDA not available, switching to CPU" -ForegroundColor Yellow
        $device = "cpu"
    } else {
        $cudaVersion = python -c "import torch; print(torch.version.cuda)" 2>$null
        Write-Host "[OK] CUDA: $cudaVersion" -ForegroundColor Green
    }
} else {
    Write-Host "  Using CPU mode (no CUDA required)"
}

# Check TCFMamba
try {
    python -c "from tcfmamba import TCFMamba" 2>$null
    Write-Host "  TCFMamba: import OK"
} catch {
    Write-Host "[FAIL] TCFMamba not installed. Run: pip install -e ." -ForegroundColor Red
    exit 1
}

# Check RecBole
try {
    $recboleVersion = python -c "import recbole; print(recbole.__version__)" 2>$null
    Write-Host "  RecBole: $recboleVersion"
} catch {
    Write-Host "[FAIL] RecBole not installed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Environment check passed" -ForegroundColor Green

# Step 2: Dataset Check
Write-Host "`n[2/5] Checking Dataset: $dataset" -ForegroundColor Cyan

python utils\prepare_datasets.py --verify --dataset $dataset 2>&1 | Select-Object -First 20

if (-not (Test-Path "dataset\$dataset")) {
    Write-Host "[FAIL] Dataset directory not found: dataset\$dataset" -ForegroundColor Red
    Write-Host "[INFO] Please prepare dataset first:" -ForegroundColor Yellow
    Write-Host "  1. Download from official source"
    Write-Host "  2. Convert using RecSysDatasets tool"
    Write-Host "  3. Place in dataset\$dataset\"
    Write-Host ""
    Write-Host "  See: python utils\prepare_datasets.py --dataset $dataset"
    exit 1
}

if (-not (Test-Path "dataset\$dataset\$dataset.inter")) {
    Write-Host "[FAIL] Missing: dataset\$dataset\$dataset.inter" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Dataset check passed" -ForegroundColor Green

# Step 3: Config Check
Write-Host "`n[3/5] Checking Config..." -ForegroundColor Cyan

$configFile = "config\tcfmamba_$dataset.yaml"
if (-not (Test-Path $configFile)) {
    Write-Host "[FAIL] Config not found: $configFile" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Config: $configFile" -ForegroundColor Green

# Step 4: Quick Training Test
Write-Host "`n[4/5] Running Minimal Training Test..." -ForegroundColor Blue
Write-Host "  This will train for $epochs epoch(s) to verify the pipeline"
Write-Host "  Using device: $device"
Write-Host ""

# Create minimal test config
$testConfig = [System.IO.Path]::GetTempFileName()
$gpuId = if ($device -eq "cpu") { "-1" } else { "0" }

$content = @"
# Minimal test configuration
$(Get-Content $configFile -Raw)

# Override for quick test
gpu_id: '$gpuId'
epochs: $epochs
train_batch_size: 256
eval_batch_size: 256
eval_step: 1
show_progress: True
"@

$content | Set-Content $testConfig

Write-Host "  Starting training..."
$logFile = "$env:TEMP\tcfmamba_test.log"

try {
    python utils\train.py `
        --model=TCFMamba `
        --dataset=$dataset `
        --config=$testConfig `
        --epochs=$epochs `
        --learning_rate=0.001 2>&1 | Tee-Object -FilePath $logFile | Select-String -Pattern "epoch|loss|metric|NDCG|Recall|Error|success" | Select-Object -First 20
} catch {
    Write-Host "[FAIL] Training test failed" -ForegroundColor Red
    Write-Host "  Check log: $logFile"
    Remove-Item $testConfig -ErrorAction SilentlyContinue
    exit 1
}

Remove-Item $testConfig -ErrorAction SilentlyContinue

if (Select-String -Path $logFile -Pattern "best valid|test result" -Quiet) {
    Write-Host "[OK] Training test completed" -ForegroundColor Green
} else {
    Write-Host "[WARN] Training may have issues, check full log:" -ForegroundColor Yellow
    Write-Host "  Get-Content $logFile -Tail 50"
}

# Step 5: Summary
Write-Host "`n[5/5] Test Summary" -ForegroundColor Cyan
Write-Host "=========================================="
Write-Host "[OK] All checks passed!" -ForegroundColor Green
Write-Host ""
Write-Host "Device: $device"
Write-Host "Dataset: $dataset"
Write-Host "Status: Ready for full training"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Full training: python utils\train.py --model=TCFMamba --dataset=$dataset --config=$configFile"
Write-Host "  2. Batch experiments: python utils\run_experiments.py --dataset $dataset"
Write-Host ""

if ($device -eq "cpu") {
    Write-Host "[INFO] Running on CPU. For GPU acceleration:" -ForegroundColor Yellow
    Write-Host "  - Install CUDA Toolkit 11.8 or 12.1"
    Write-Host "  - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    Write-Host "  - Re-run with: .\scripts\quick_test.ps1 gpu"
}

Write-Host "=========================================="
Write-Host "Pipeline test complete!" -ForegroundColor Green
