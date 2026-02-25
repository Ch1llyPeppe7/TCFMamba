"""
Dataset preparation script for TCFMamba.

This script helps prepare datasets following RecBole's SOP:
1. Download raw datasets from official sources
2. Convert to RecBole atomic file format (.inter, .item)
3. Verify dataset structure

Usage:
    python scripts/prepare_datasets.py --all
    python scripts/prepare_datasets.py --dataset gowalla
    python scripts/prepare_datasets.py --verify

Dataset Sources:
- Gowalla: https://snap.stanford.edu/data/loc-gowalla.html
- Foursquare: https://archive.org/details/201309_foursquare_dataset_umn

Expected Directory Structure (following RecBole SOP):
    dataset/
    ├── gowalla/
    │   ├── gowalla.inter       # User-POI interactions
    │   └── gowalla.item        # POI features (lat, lon)
    ├── foursquare_TKY/
    │   ├── foursquare_TKY.inter
    │   └── foursquare_TKY.item
    └── foursquare_NYC/
        ├── foursquare_NYC.inter
        └── foursquare_NYC.item
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare datasets for TCFMamba")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gowalla", "foursquare_TKY", "foursquare_NYC", "all"],
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing datasets"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="raw_data",
        help="Directory for raw downloaded data"
    )
    
    return parser.parse_args()


def check_dataset_exists(dataset_name: str) -> bool:
    """Check if dataset already exists in correct format."""
    dataset_dir = Path(f"dataset/{dataset_name}")
    
    if not dataset_dir.exists():
        return False
    
    inter_file = dataset_dir / f"{dataset_name}.inter"
    item_file = dataset_dir / f"{dataset_name}.item"
    
    return inter_file.exists() and item_file.exists()


def verify_dataset(dataset_name: str) -> dict:
    """Verify dataset structure and print statistics."""
    dataset_dir = Path(f"dataset/{dataset_name}")
    result = {"name": dataset_name, "exists": False, "valid": False}
    
    if not dataset_dir.exists():
        print(f"[FAIL] Dataset directory not found: {dataset_dir}")
        return result
    
    result["exists"] = True
    inter_file = dataset_dir / f"{dataset_name}.inter"
    item_file = dataset_dir / f"{dataset_name}.item"
    
    print(f"\n[VERIFY] Verifying {dataset_name}...")
    print(f"   Directory: {dataset_dir.absolute()}")
    
    if not inter_file.exists():
        print(f"   [FAIL] Missing: {inter_file.name}")
        return result
    
    if not item_file.exists():
        print(f"   [FAIL] Missing: {item_file.name}")
        return result
    
    # Try to read and validate
    try:
        inter_df = pd.read_csv(inter_file, sep='\t')
        item_df = pd.read_csv(item_file, sep='\t')
        
        # Check required columns for POI recommendation
        inter_cols = set(inter_df.columns)
        item_cols = set(item_df.columns)
        
        # Expected columns
        required_inter = {'user_id:token', 'venue_id:token', 'timestamp:float'}
        required_item = {'venue_id:token', 'latitude:float', 'longitude:float'}
        
        missing_inter = required_inter - inter_cols
        missing_item = required_item - item_cols
        
        if missing_inter:
            print(f"   [WARN]  Missing interaction columns: {missing_inter}")
        if missing_item:
            print(f"   [WARN]  Missing item columns: {missing_item}")
        
        print(f"   [OK] Interactions: {len(inter_df):,} records")
        print(f"   [OK] POIs: {len(item_df):,} items")
        print(f"   [OK] Users: {inter_df['user_id:token'].nunique():,}")
        
        # Check if latitude/longitude exists
        if 'latitude:float' in item_cols and 'longitude:float' in item_cols:
            print(f"   [OK] Geographic coordinates available")
        
        result["valid"] = True
        result["stats"] = {
            "interactions": len(inter_df),
            "items": len(item_df),
            "users": inter_df['user_id:token'].nunique()
        }
        
    except Exception as e:
        print(f"   [FAIL] Error reading dataset: {e}")
    
    return result


def print_download_instructions(dataset_name: str):
    """Print instructions for downloading and converting datasets."""
    instructions = {
        "gowalla": """
[INFO] Gowalla Dataset Download & Conversion:

1. Download from: https://snap.stanford.edu/data/loc-gowalla.html
   Files: loc-gowalla_totalCheckins.txt.gz

2. Extract and convert using RecBole's official tool:
   git clone https://github.com/RUCAIBox/RecSysDatasets
   cd RecSysDatasets/conversion_tools
   pip install -r requirements.txt
   
   python run.py --dataset gowalla \\
       --input_path /path/to/gowalla_data \\
       --output_path /path/to/output \\
       --convert_inter --duplicate_removal

3. Copy output files to dataset/gowalla/
   Docs: https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools/usage
        """,
        "foursquare_TKY": """
[INFO] Foursquare Tokyo Dataset Download & Conversion:

1. Download from: https://archive.org/details/201309_foursquare_dataset_umn
   File: dataset_TSMC2014_TKY.csv

2. Convert using RecBole's official tool:
   git clone https://github.com/RUCAIBox/RecSysDatasets
   cd RecSysDatasets/conversion_tools
   
   python run.py --dataset foursquare \\
       --input_path /path/to/foursquare_TKY.csv \\
       --output_path /path/to/output \\
       --convert_inter

3. Copy output files to dataset/foursquare_TKY/
        """,
        "foursquare_NYC": """
[INFO] Foursquare NYC Dataset Download & Conversion:

1. Download from: https://archive.org/details/201309_foursquare_dataset_umn
   File: dataset_TSMC2014_NYC.csv

2. Convert using RecBole's official tool:
   git clone https://github.com/RUCAIBox/RecSysDatasets
   cd RecSysDatasets/conversion_tools
   
   python run.py --dataset foursquare \\
       --input_path /path/to/foursquare_NYC.csv \\
       --output_path /path/to/output \\
       --convert_inter

3. Copy output files to dataset/foursquare_NYC/
        """
    }
    
    print(instructions.get(dataset_name, "Unknown dataset"))


def create_dataset_structure():
    """Create the standard dataset directory structure."""
    datasets = ["gowalla", "foursquare_TKY", "foursquare_NYC"]
    
    for dataset in datasets:
        dataset_dir = Path(f"dataset/{dataset}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("[OK] Created dataset directory structure")
    print("   dataset/")
    print("   ├── gowalla/")
    print("   ├── foursquare_TKY/")
    print("   └── foursquare_NYC/")


def main():
    """Main entry point."""
    args = get_args()
    
    print("=" * 60)
    print("TCFMamba Dataset Preparation")
    print("=" * 60)
    
    # Verify mode
    if args.verify:
        print("\n[VERIFY] Verifying existing datasets...")
        
        datasets = ["gowalla", "foursquare_TKY", "foursquare_NYC"]
        all_valid = True
        
        for dataset in datasets:
            result = verify_dataset(dataset)
            if not result["exists"] or not result["valid"]:
                all_valid = False
        
        print("\n" + "=" * 60)
        if all_valid:
            print("[OK] All datasets are ready!")
        else:
            print("[WARN] Some datasets are missing or invalid.")
            print("   Run: python utils/prepare_datasets.py --dataset <name>")
        
        return
    
    # Create directory structure
    create_dataset_structure()
    
    # Check specific dataset
    if args.dataset:
        if args.dataset == "all":
            datasets = ["gowalla", "foursquare_TKY", "foursquare_NYC"]
        else:
            datasets = [args.dataset]
        
        for dataset in datasets:
            print(f"\n[CHECK] Checking {dataset}...")
            
            if check_dataset_exists(dataset):
                print(f"   [OK] Dataset already exists!")
                verify_dataset(dataset)
            else:
                print(f"   [WARN] Dataset not found in RecBole format")
                print_download_instructions(dataset)
    
    print("\n" + "=" * 60)
    print("For more information, see:")
    print("  - README.md#datasets")
    print("  - https://github.com/RUCAIBox/RecSysDatasets")
    print("=" * 60)


if __name__ == "__main__":
    main()
