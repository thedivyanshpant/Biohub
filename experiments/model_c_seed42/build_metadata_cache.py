import json
import time
from pathlib import Path
import sys
import torch
from tqdm import tqdm

import os

PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", Path(__file__).resolve().parents[2]))
REPO_DIR = Path(os.environ.get("BIOHUB_TRACKING_REPO",
    PROJECT_ROOT / "kaggle_source" / "tracking_repo" if (PROJECT_ROOT / "kaggle_source" / "tracking_repo").exists()
    else PROJECT_ROOT / "code" / "leader_reproduction" / "working" / "tracking_repo"
))
EXP_DIR = PROJECT_ROOT / "experiments" / "model_c_seed42"

sys.path.insert(0, str(REPO_DIR / "scripts"))
sys.path.insert(0, str(REPO_DIR / "src"))

from train_unet_transformer import load_dataset_windows

def main():
    splits_path = EXP_DIR / "dataset_splits_model_c.json"
    splits = json.loads(splits_path.read_text())[0]

    train_stems = splits["train"]
    val_stems = splits["test"]
    train_dir = PROJECT_ROOT / "database" / "train"

    train_files = [train_dir / f"{s}.zarr" for s in train_stems]
    val_files = [train_dir / f"{s}.zarr" for s in val_stems]

    print(f"Loading metadata for {len(train_files)} train datasets...")
    t0 = time.time()
    train_data = []
    for f in tqdm(train_files, desc="Train metadata"):
        vm, ws = load_dataset_windows(f, window_size=2, downsample=(1, 4, 4))
        train_data.append((vm, ws))
    n_train_w = sum(len(w) for _, w in train_data)
    print(f"Train metadata loaded: {n_train_w} windows in {time.time()-t0:.1f}s")

    print(f"\nLoading metadata for {len(val_files)} val datasets...")
    t0 = time.time()
    val_data = []
    for f in tqdm(val_files, desc="Val metadata"):
        vm, ws = load_dataset_windows(f, window_size=2, downsample=(1, 4, 4))
        val_data.append((vm, ws))
    n_val_w = sum(len(w) for _, w in val_data)
    print(f"Val metadata loaded: {n_val_w} windows in {time.time()-t0:.1f}s")

    all_windows = [w for _, ws in train_data + val_data for w in ws]
    max_nodes = max(max(w.node_counts) for w in all_windows)
    print(f"\nmax_nodes across all windows: {max_nodes}")

    cache_path = EXP_DIR / "metadata_cache.pt"
    print(f"Saving cache to {cache_path}...")
    torch.save({
        "train_data": train_data,
        "val_data": val_data,
        "max_nodes": max_nodes,
        "n_train_windows": n_train_w,
        "n_val_windows": n_val_w,
    }, cache_path)
    print("Done! Cache saved successfully.")

if __name__ == "__main__":
    main()
