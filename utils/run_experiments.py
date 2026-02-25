"""
Batch experiment runner for TCFMamba.

This script enables running multiple experiments across different datasets
and configurations with a single command. Results are automatically
aggregated and saved for easy comparison.

Usage:
    # Run all experiments (TCFMamba on all datasets)
    python utils/run_experiments.py --all

    # Run specific dataset
    python utils/run_experiments.py --dataset gowalla

    # Run multiple datasets
    python utils/run_experiments.py --dataset gowalla foursquare_TKY

    # Run with custom seeds for reproducibility
    python utils/run_experiments.py --all --seeds 42 2023 2024

    # Run baseline models for comparison
    python utils/run_experiments.py --all --baselines BERT4Rec GRU4Rec

    # Dry run (show what would be executed)
    python utils/run_experiments.py --all --dry-run

Results are saved to:
    - results/experiments_TIMESTAMP/ (individual results)
    - results/summary_TIMESTAMP.csv (aggregated results)
    - results/summary_TIMESTAMP.md (formatted report)
"""

import os
import sys
import argparse
import json
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Experiment configurations
DATASETS = ["gowalla", "foursquare_TKY", "foursquare_NYC"]

CONFIG_MAP = {
    "TCFMamba": {
        "gowalla": "config/tcfmamba_gowalla.yaml",
        "foursquare_TKY": "config/tcfmamba_tky.yaml",
        "foursquare_NYC": "config/tcfmamba_nyc.yaml",
    },
    "BERT4Rec": {
        "gowalla": "config/baseline/bert4rec_gowalla.yaml",
        "foursquare_TKY": "config/baseline/bert4rec_tky.yaml",
        "foursquare_NYC": "config/baseline/bert4rec_nyc.yaml",
    },
    "GRU4Rec": {
        "gowalla": "config/baseline/gru4rec_gowalla.yaml",
        "foursquare_TKY": "config/baseline/gru4rec_tky.yaml",
        "foursquare_NYC": "config/baseline/gru4rec_nyc.yaml",
    },
    "SRGNN": {
        "gowalla": "config/baseline/srgnn_gowalla.yaml",
        "foursquare_TKY": "config/baseline/srgnn_tky.yaml",
        "foursquare_NYC": "config/baseline/srgnn_nyc.yaml",
    },
}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch experiments")

    # Experiment selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments (TCFMamba on all datasets)",
    )
    group.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS,
        help="Run on specific dataset(s)",
    )
    group.add_argument(
        "--config",
        type=str,
        help="Run a single specific config file",
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=["TCFMamba"],
        choices=["TCFMamba", "BERT4Rec", "GRU4Rec", "SRGNN"],
        help="Models to evaluate (default: TCFMamba)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        help="Additional baseline models to run for comparison",
    )

    # Experiment settings
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds to use (default: [42])",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID(s) to use",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (if multiple GPUs available)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_on_error",
        action="store_true",
        help="Continue on experiment failure",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging for all experiments",
    )

    return parser.parse_args()


def generate_experiments(args) -> List[Dict[str, Any]]:
    """Generate list of experiments to run."""
    experiments = []

    # Determine datasets to run on
    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = args.dataset
    else:
        datasets = []

    # Determine models to run
    models = args.models.copy()
    if args.baselines:
        models.extend(args.baselines)

    # Generate experiment combinations
    for model in models:
        for dataset in datasets:
            for seed in args.seeds:
                # Get config file
                if model in CONFIG_MAP and dataset in CONFIG_MAP[model]:
                    config_file = CONFIG_MAP[model][dataset]
                else:
                    # Fallback: construct config path
                    config_file = f"config/{model.lower()}_{dataset}.yaml"

                # Skip if config doesn't exist
                if not os.path.exists(config_file):
                    print(f"Warning: Config file not found: {config_file}")
                    continue

                exp = {
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "config": config_file,
                    "name": f"{model}_{dataset}_seed{seed}",
                }
                experiments.append(exp)

    return experiments


def run_single_experiment(exp: Dict[str, Any], args, output_dir: str) -> Dict[str, Any]:
    """
    Run a single experiment.

    Args:
        exp: Experiment configuration
        args: Command line arguments
        output_dir: Output directory

    Returns:
        Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"{'='*60}")
    print(f"Model: {exp['model']}")
    print(f"Dataset: {exp['dataset']}")
    print(f"Seed: {exp['seed']}")
    print(f"Config: {exp['config']}")

    # Build command
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model", exp["model"],
        "--dataset", exp["dataset"],
        "--config", exp["config"],
        "--seed", str(exp["seed"]),
        "--gpu_id", args.gpu_id,
    ]

    # Add output directory
    exp_output_dir = os.path.join(output_dir, exp["name"])
    os.makedirs(exp_output_dir, exist_ok=True)
    cmd.extend(["--output_dir", exp_output_dir])

    # Add TensorBoard
    if args.tensorboard:
        cmd.append("--tensorboard")

    print(f"\nCommand: {' '.join(cmd)}")

    if args.dry_run:
        return {"status": "dry_run", "name": exp["name"]}

    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed_time = time.time() - start_time

        # Parse results from output
        results = parse_experiment_output(result.stdout)
        results["status"] = "success"
        results["time"] = elapsed_time
        results["name"] = exp["name"]

        # Save individual results
        result_file = os.path.join(exp_output_dir, "results.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Success in {elapsed_time:.1f}s")
        return results

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed after {elapsed_time:.1f}s")
        print(f"Error: {e.stderr}")

        results = {
            "status": "failed",
            "name": exp["name"],
            "time": elapsed_time,
            "error": str(e),
        }

        if not args.continue_on_error:
            raise

        return results


def parse_experiment_output(output: str) -> Dict[str, Any]:
    """Parse experiment results from training output."""
    results = {}

    # Parse key metrics from output
    lines = output.split("\n")
    for line in lines:
        # Look for metric lines
        if "NDCG@" in line or "Recall@" in line or "MRR@" in line:
            # Extract metrics
            parts = line.split(":")
            if len(parts) >= 2:
                metric_name = parts[0].strip()
                metric_value = parts[1].strip()
                try:
                    results[metric_name] = float(metric_value)
                except ValueError:
                    results[metric_name] = metric_value

    return results


def save_summary(all_results: List[Dict], output_dir: str, timestamp: str):
    """Save experiment summary in multiple formats."""
    # Create summary directory
    summary_dir = os.path.join(output_dir, f"summary_{timestamp}")
    os.makedirs(summary_dir, exist_ok=True)

    # Save as JSON
    json_file = os.path.join(summary_dir, "results.json")
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save as CSV
    csv_file = os.path.join(summary_dir, "results.csv")
    if all_results:
        # Get all unique keys
        keys = set()
        for r in all_results:
            keys.update(r.keys())
        keys = sorted(keys)

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

    # Generate Markdown report
    md_file = os.path.join(summary_dir, "report.md")
    generate_markdown_report(all_results, md_file, timestamp)

    print(f"\nSummary saved to: {summary_dir}")
    return summary_dir


def generate_markdown_report(results: List[Dict], output_file: str, timestamp: str):
    """Generate a formatted Markdown report."""
    with open(output_file, "w") as f:
        f.write(f"# TCFMamba Experiment Results\n\n")
        f.write(f"Generated: {timestamp}\n\n")

        # Group by model and dataset
        from collections import defaultdict
        grouped = defaultdict(lambda: defaultdict(list))

        for r in results:
            if r.get("status") == "success":
                model = r.get("name", "").split("_")[0]
                dataset = r.get("name", "").split("_")[1] if "_" in r.get("name", "") else "unknown"
                grouped[model][dataset].append(r)

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Dataset | Status | NDCG@10 | Recall@10 | Time (s) |\n")
        f.write("|-------|---------|--------|---------|-----------|----------|\n")

        for model in sorted(grouped.keys()):
            for dataset in sorted(grouped[model].keys()):
                for r in grouped[model][dataset]:
                    ndcg = r.get("NDCG@10", r.get("test_result", {}).get("NDCG@10", "N/A"))
                    recall = r.get("Recall@10", r.get("test_result", {}).get("Recall@10", "N/A"))
                    time_val = r.get("time", "N/A")
                    status = "✓" if r.get("status") == "success" else "✗"

                    f.write(f"| {model} | {dataset} | {status} | {ndcg} | {recall} | {time_val:.1f} |\n")

        f.write("\n## Detailed Results\n\n")
        for r in results:
            f.write(f"### {r.get('name', 'Unknown')}\n\n")
            f.write(f"```json\n")
            f.write(json.dumps(r, indent=2))
            f.write(f"\n```\n\n")


def main():
    """Main entry point."""
    args = get_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"experiments_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"TCFMamba Batch Experiment Runner")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("Mode: DRY RUN (no experiments will be executed)")
    print(f"{'='*60}\n")

    # Generate experiments
    experiments = generate_experiments(args)

    if not experiments:
        print("No experiments to run!")
        return

    print(f"Total experiments: {len(experiments)}")
    print("\nExperiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")

    if args.dry_run:
        return

    # Run experiments
    print(f"\n{'='*60}")
    print("Starting experiments...")
    print(f"{'='*60}\n")

    all_results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        result = run_single_experiment(exp, args, output_dir)
        all_results.append(result)

    # Save summary
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}\n")

    summary_dir = save_summary(all_results, output_dir, timestamp)

    # Print summary
    print("\nResults Summary:")
    print("-" * 60)
    for r in all_results:
        status_icon = "✓" if r.get("status") == "success" else "✗"
        print(f"{status_icon} {r['name']}: {r.get('status', 'unknown')}")
    print("-" * 60)

    print(f"\nResults saved to: {summary_dir}")


if __name__ == "__main__":
    main()
