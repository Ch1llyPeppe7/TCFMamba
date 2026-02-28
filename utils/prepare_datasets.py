"""
Minimal dataset preparation wrapper using RecBole's built-in download.

RecBole automatically downloads datasets from AWS S3 when create_dataset() is called
and the dataset is registered in url.yaml.

Usage:
    python utils/prepare_datasets.py --verify
    python utils/prepare_datasets.py --dataset gowalla --auto
"""

import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed, init_logger
from logging import getLogger


# Map user names to RecBole's url.yaml names
NAME_MAP = {
    "gowalla": "gowalla-merged",
    "foursquare_TKY": "foursquare-tky-merged",
    "foursquare_NYC": "foursquare-nyc-merged",
}


def check_exists(name):
    """Check if dataset exists in correct format."""
    d = Path(f"dataset/{name}")
    return (d / f"{name}.inter").exists() and (d / f"{name}.item").exists()


def verify(name):
    """Verify dataset and print stats."""
    d = Path(f"dataset/{name}")
    inter = d / f"{name}.inter"
    item = d / f"{name}.item"

    print(f"\n[VERIFY] {name}")
    if not inter.exists() or not item.exists():
        print("   [FAIL] Missing files")
        return False

    try:
        i_df = pd.read_csv(inter, sep='\t')
        m_df = pd.read_csv(item, sep='\t')
        print(f"   [OK] {len(i_df):,} interactions, {len(m_df):,} items, {i_df['user_id:token'].nunique():,} users")
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


def prepare(name):
    """Download dataset using RecBole's url utilities (no Config/Dataset)."""
    if check_exists(name):
        return True

    # If .inter exists but .item missing (e.g. merged format), build .item and exit
    inter_path = Path(f"dataset/{name}/{name}.inter")
    item_path = Path(f"dataset/{name}/{name}.item")
    if inter_path.exists() and not item_path.exists():
        _build_item_from_inter(inter_path, item_path, name)
        return True

    recbole_name = NAME_MAP.get(name, name)
    print(f"\n[PREPARE] {name} (RecBole: {recbole_name})")
    print("   Downloading from RecBole S3 (may take 1-2 minutes)...")

    try:
        import yaml
        from recbole.utils.url import decide_download, download_url, extract_zip, makedirs, rename_atomic_files

        # Load url.yaml from RecBole package
        import recbole
        url_file = Path(recbole.__file__).parent / "properties" / "dataset" / "url.yaml"
        if not url_file.exists():
            raise FileNotFoundError(f"RecBole url.yaml not found: {url_file}")

        with open(url_file) as f:
            dataset2url = yaml.load(f.read(), Loader=yaml.FullLoader)
        if recbole_name not in dataset2url:
            raise ValueError(f"Dataset {recbole_name} not in RecBole url.yaml")
        url = dataset2url[recbole_name]

        if not decide_download(url):
            print("   Download cancelled.")
            return False

        dataset_path = f"dataset/{recbole_name}"
        makedirs(dataset_path)
        path = download_url(url, dataset_path)
        basename = os.path.splitext(os.path.basename(path))[0]
        extract_zip(path, dataset_path)
        # 保留下载的 zip，不删除
        rename_atomic_files(dataset_path, basename, recbole_name)
        print("   Download done.")

        # Copy to user-facing path dataset/{name}/
        src_dir = Path(f"dataset/{recbole_name}")
        dst_dir = Path(f"dataset/{name}")
        if src_dir != dst_dir:
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    new_name = f.name.replace(recbole_name, name)
                    shutil.copy2(f, dst_dir / new_name)
            shutil.rmtree(src_dir)
        else:
            dst_dir = Path(f"dataset/{name}")

        # If .item missing (e.g. RecBole merged has only .inter with lat/lon), build it
        inter_path = dst_dir / f"{name}.inter"
        item_path = dst_dir / f"{name}.item"
        if inter_path.exists() and not item_path.exists():
            _build_item_from_inter(inter_path, item_path, name)
        if item_path.exists():
            _ensure_item_header_venue_id(item_path)

        print(f"   [OK] {name} ready")
        return True

    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def _ensure_item_header_venue_id(item_path):
    """若 .item 表头是 item_id，改为 venue_id，以便 config 使用 ITEM_ID_FIELD: venue_id。仅改表头不改数据。"""
    with open(item_path, "r", encoding="utf-8") as f:
        first = f.readline()
    if "item_id" not in first or "venue_id" in first:
        return
    first = first.replace("item_id", "venue_id")
    with open(item_path, "r", encoding="utf-8") as f:
        f.readline()
        rest = f.read()
    with open(item_path, "w", encoding="utf-8") as f:
        f.write(first)
        f.write(rest)
    print(f"   [{item_path.name}] header: item_id -> venue_id")


def _build_item_from_inter(inter_path, item_path, name):
    """Build .item from .inter when lat/lon are in inter (e.g. RecBole merged). 保留原字段名（item_id 或 venue_id）。"""
    df = pd.read_csv(inter_path, sep="\t")
    id_col = None
    for c in ["item_id:token", "venue_id:token", "item_id", "venue_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"inter file has no item_id or venue_id column: {list(df.columns)}")
    lat_col = "latitude:float" if "latitude:float" in df.columns else "latitude"
    lon_col = "longitude:float" if "longitude:float" in df.columns else "longitude"
    item_df = df.groupby(id_col, as_index=False).agg({lat_col: "first", lon_col: "first"})
    item_df.to_csv(item_path, sep="\t", index=False)
    print(f"   Built {item_path.name} from inter ({len(item_df):,} items)")


def prepare_if_needed(name):
    """Called by train.py before training."""
    if check_exists(name):
        item_path = Path(f"dataset/{name}/{name}.item")
        if item_path.exists():
            _ensure_item_header_venue_id(item_path)
        return True
    return prepare(name)


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gowalla", "foursquare_TKY", "foursquare_NYC", "all"])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("Dataset Preparation")
    print("=" * 50)

    datasets = ["gowalla", "foursquare_TKY", "foursquare_NYC"]

    if args.verify:
        all_ok = all(verify(d) for d in datasets)
        print("\n" + ("[OK] All ready" if all_ok else "[WARN] Some missing"))
    elif args.dataset:
        todo = datasets if args.dataset == "all" else [args.dataset]
        for d in todo:
            if args.auto:
                ok = prepare(d)
                print(f"  {d}: {'[OK]' if ok else '[FAIL]'}")
            else:
                exists = "[OK]" if check_exists(d) else "[MISSING]"
                print(f"  {d}: {exists}")
        if not args.auto:
            print("\nUse --auto to download")
    else:
        print("\nStatus:")
        for d in datasets:
            exists = "[OK]" if check_exists(d) else "[MISSING]"
            print(f"  {d}: {exists}")
        print("\nUsage: python utils/prepare_datasets.py --dataset gowalla --auto")

    print("=" * 50)
