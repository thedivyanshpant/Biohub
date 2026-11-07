"""
package_submission.py

Prepares and verifies the Submission1 package for Kaggle upload:
1. Validates that notebook/submission1.ipynb exists and is valid JSON.
2. Checks that required checkpoints exist in local kaggle_artifacts.
3. Checks that the dedicated Kaggle Support Dataset package exists and is complete.
4. Generates a summary checklist for uploading to Kaggle.
"""

import json
import os
from pathlib import Path

SUBMISSION_DIR = Path(os.environ.get("BIOHUB_SUBMISSION_DIR", Path(__file__).resolve().parent.parent))
NOTEBOOK_PATH = SUBMISSION_DIR / "notebook/submission1.ipynb"
PROJECT_ROOT = SUBMISSION_DIR.parent
ARTIFACTS_DIR = Path(os.environ.get("BIOHUB_ARTIFACTS_DIR", PROJECT_ROOT / "kaggle_artifacts"))
SUPPORT_PACK_DIR = SUBMISSION_DIR / "kaggle_support_pack"

print("=" * 78)
print("SUBMISSION 1 KAGGLE PACKAGING & READINESS CHECK")
print("=" * 78)

# 1. Validate Notebook
assert NOTEBOOK_PATH.exists(), f"Notebook missing: {NOTEBOOK_PATH}"
nb = json.load(NOTEBOOK_PATH.open())
print(f"✓ Notebook verified: {NOTEBOOK_PATH}")
print(f"  Format: nbformat {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"  Cells : {len(nb['cells'])} code cells")

# 2. Check Checkpoints
checkpoints = [
    ("Model A (Primary 50ep)", ARTIFACTS_DIR / "biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"),
    ("Model B (Seed 314159)", ARTIFACTS_DIR / "biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"),
    ("DeepCenter Veto", ARTIFACTS_DIR / "biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"),
]

for name, path in checkpoints:
    assert path.exists(), f"Missing checkpoint: {path}"
    print(f"✓ Checkpoint verified: {name:<25} ({path.stat().st_size / 1e6:.1f} MB)")

# 3. Check Dedicated Kaggle Support Dataset
assert SUPPORT_PACK_DIR.exists(), f"Missing support dataset: {SUPPORT_PACK_DIR}"
repo_dir = SUPPORT_PACK_DIR / "repo"
wheels_dir = SUPPORT_PACK_DIR / "wheels"
assert repo_dir.is_dir(), f"Missing repo/ in {SUPPORT_PACK_DIR}"
assert wheels_dir.is_dir(), f"Missing wheels/ in {SUPPORT_PACK_DIR}"

repo_files = list(repo_dir.rglob("*.py"))
wheel_files = list(wheels_dir.glob("*.whl"))
print(f"✓ Dedicated Support Dataset verified: {SUPPORT_PACK_DIR}")
print(f"  Repository Python files: {len(repo_files)}")
print(f"  Offline wheels         : {len(wheel_files)} wheels ({sum(w.stat().st_size for w in wheel_files) / 1e6:.1f} MB)")

print("\n" + "=" * 78)
print("REQUIRED KAGGLE ATTACHMENTS (5 INPUTS TOTAL):")
print("1. Competition Data  : biohub-cell-tracking-during-development")
print("2. Model A (Primary) : thedivyanshpant/biohub-tracking-support-pack-50ep-v1")
print("3. Model B (Seed)    : thedivyanshpant/biohub-temporal-unet3d-seed314159-v1")
print("4. DeepCenter Prior  : thedivyanshpant/biohub-deepcenter-unet3d-center-prior-v1")
print(f"5. Support Dataset   : upload {SUPPORT_PACK_DIR}")
print("                       (as Kaggle Dataset: biohub-kaggle-support)")
print(f"\nNotebook upload: {NOTEBOOK_PATH}")
print("Ensure GPU accelerator is enabled and Internet is toggled OFF.")
print("=" * 78)
