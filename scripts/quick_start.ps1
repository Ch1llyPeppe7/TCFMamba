# TCFMamba Quick Start Script for PowerShell
# This script provides quick commands to reproduce TCFMamba experiments

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Usage: .\scripts\quick_start.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  gowalla          Train on Gowalla dataset"
    Write-Host "  tky              Train on Foursquare Tokyo dataset"
    Write-Host "  nyc              Train on Foursquare NYC dataset"
    Write-Host "  all              Run all experiments"
    Write-Host "  test             Quick test (10 epochs)"
    Write-Host "  help             Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\quick_start.ps1 gowalla"
    Write-Host "  .\scripts\quick_start.ps1 test"
    Write-Host "  .\scripts\quick_start.ps1 all"
}

function Train-TCFMamba {
    param($Dataset, $Config)
    
    Write-Host "Training TCFMamba on ${Dataset}..."
    python utils/train.py `
        --model=TCFMamba `
        --dataset=$Dataset `
        --config=$Config `
        --epochs=100
}

function Run-All {
    Write-Host "Running all experiments..."
    python utils/run_experiments.py --all
}

Write-Host "=========================================="
Write-Host "TCFMamba Quick Start"
Write-Host "=========================================="

switch ($Command.ToLower()) {
    "gowalla" {
        Train-TCFMamba -Dataset "gowalla" -Config "config/tcfmamba_gowalla.yaml"
    }
    "tky" {
        Train-TCFMamba -Dataset "foursquare_TKY" -Config "config/tcfmamba_tky.yaml"
    }
    "nyc" {
        Train-TCFMamba -Dataset "foursquare_NYC" -Config "config/tcfmamba_nyc.yaml"
    }
    "all" {
        Run-All
    }
    "test" {
        Write-Host "Running quick test (10 epochs on gowalla)..."
        python utils/train.py `
            --model=TCFMamba `
            --dataset=gowalla `
            --config=config/tcfmamba_gowalla.yaml `
            --epochs=10
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "Unknown command: $Command"
        Show-Help
    }
}

Write-Host ""
Write-Host "Done!"
