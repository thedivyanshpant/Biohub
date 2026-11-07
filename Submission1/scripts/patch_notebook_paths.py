"""
patch_notebook_paths.py

Fixes submission1.ipynb so all /kaggle/input paths match the actual
Kaggle model mount layout for thedivyanshpant's account:

  /kaggle/input/models/thedivyanshpant/<slug>/pytorch/default/1/

Also applies PPSWEEP winning overrides and adds
BIOHUB_KERNEL_START_TS / repair deadline / ILP timeout / cache-dir
environment variables that are required by the reference notebook but
were missing from our Cell 0.
"""

import json
import os
import re
from pathlib import Path

NOTEBOOK_PATH = Path(os.environ.get("BIOHUB_NOTEBOOK_PATH", Path(__file__).resolve().parent.parent / "notebook/submission1.ipynb"))
BACKUP_PATH   = NOTEBOOK_PATH.with_suffix(".ipynb.bak")

# ─────────────────────────────────────────────────────────────────
# CANONICAL MODEL PATHS  (the actual Kaggle mount points)
# ─────────────────────────────────────────────────────────────────
PRIMARY_SLUG      = "biohub-tracking-support-pack-50ep-v1"
SECONDARY_SLUG    = "biohub-temporal-unet3d-seed314159-v1"
DEEPCENTER_SLUG   = "biohub-deepcenter-unet3d-center-prior-v1"
USERNAME          = "thedivyanshpant"

# Kaggle Model mounts
MODEL_BASE        = f"/kaggle/input/models/{USERNAME}"
PRIMARY_MOUNT     = f"{MODEL_BASE}/{PRIMARY_SLUG}/pytorch/default/1"
SECONDARY_MOUNT   = f"{MODEL_BASE}/{SECONDARY_SLUG}/pytorch/default/1"
DEEPCENTER_MOUNT  = f"{MODEL_BASE}/{DEEPCENTER_SLUG}/pytorch/default/1"

# Competition data
COMP_DATA_PATH    = "/kaggle/input/competitions/biohub-cell-tracking-during-development"

def model_mount(slug: str) -> str:
    return f"{MODEL_BASE}/{slug}/pytorch/default/1"


# ─────────────────────────────────────────────────────────────────
# REPLACEMENTS TO MAKE
# ─────────────────────────────────────────────────────────────────

# All old-style flat /kaggle/input/<slug>  and
#          pilkwang dataset paths
# → new thedivyanshpant model paths

GLOBAL_REPLACEMENTS = [
    # pilkwang dataset paths → thedivyanshpant model paths
    (
        f"/kaggle/input/datasets/pilkwang/{PRIMARY_SLUG}",
        PRIMARY_MOUNT,
    ),
    (
        f"/kaggle/input/datasets/pilkwang/{SECONDARY_SLUG}",
        SECONDARY_MOUNT,
    ),
    (
        f"/kaggle/input/datasets/pilkwang/{DEEPCENTER_SLUG}",
        DEEPCENTER_MOUNT,
    ),
    # flat /kaggle/input/<slug> paths → model mounts
    (
        f"/kaggle/input/{PRIMARY_SLUG}",
        PRIMARY_MOUNT,
    ),
    (
        f"/kaggle/input/{SECONDARY_SLUG}",
        SECONDARY_MOUNT,
    ),
    (
        f"/kaggle/input/{DEEPCENTER_SLUG}",
        DEEPCENTER_MOUNT,
    ),
    # pilkwang username → thedivyanshpant
    (
        "/kaggle/input/datasets/pilkwang/",
        f"{MODEL_BASE}/",
    ),
]

# ─────────────────────────────────────────────────────────────────
# CELL 0 ADDITIONS  (env vars missing from our notebook)
# ─────────────────────────────────────────────────────────────────

# Lines to ADD to Cell 0 (appended after existing env vars, before print)
CELL0_EXTRA_ENVVARS = """
# --- Runtime hardening (required for 9-hour Kaggle execution) ---
import time as _t0_time
os.environ["BIOHUB_KERNEL_START_TS"] = str(_t0_time.time())
os.environ["BIOHUB_ILP_TIMEOUT_S"] = "1200"           # 20-min ILP cap per dataset
os.environ["BIOHUB_REPAIR_DEADLINE_S"] = "27000"       # 7.5h: degrade post-processing to guarantee CSV
os.environ["BIOHUB_FRAME_CACHE_MAX_FRAMES"] = "48"     # RAM cap for video cache (~30 GB Kaggle)
os.environ["BIOHUB_CACHE_DIR"] = "/kaggle/working/edge_cache"
os.environ["BIOHUB_LOWDET_THRESHOLD"] = "0.3"
os.environ["BIOHUB_CACHE_EDGE_THRESHOLD"] = "1.0"
os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"            # No local validation on Kaggle (no internet)
os.environ["BIOHUB_MOTION_RELINK_FLOW_MODE"] = "seed"
os.environ["BIOHUB_MOTION_RELINK_FLOW_K"] = "12"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RADIUS_UM"] = "40.0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_EXCLUDE_UM"] = "1.5"
os.environ["BIOHUB_MOTION_RELINK_FLOW_MIN_SAMPLES"] = "4"
os.environ["BIOHUB_MOTION_RELINK_FLOW_GATE"] = "1"
os.environ["BIOHUB_MOTION_RELINK_FLOW_ITER"] = "1"
os.environ["BIOHUB_MOTION_RELINK_FLOW_SEED_GATE_UM"] = "0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RAW_ADMIT"] = "1"
os.environ["BIOHUB_MOTION_RELINK_FLOW_Z_WEIGHT"] = "1.0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RAW_COST"] = "0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_TIGHT_UM"] = "7.0"
os.environ["BIOHUB_MOTION_RELINK_FLOW_RELAXED_UM"] = "0"
os.environ["BIOHUB_READMIT_RADIUS_UM"] = "4"
os.environ["BIOHUB_READMIT_MIN_SCORE"] = "0.965"
os.environ["BIOHUB_GAPFILL_MAX_GAP"] = "3"
os.environ["BIOHUB_GAPFILL_MIN_SCORE"] = "0.5"
os.environ["BIOHUB_GAPFILL_STEP_UM"] = "5.0"
os.environ["BIOHUB_GAPFILL_PEAK_RADIUS_UM"] = "3.5"
os.environ["BIOHUB_GAPFILL_EXCLUDE_UM"] = "2.0"
os.environ["BIOHUB_GAPFILL_ALLOW_SYNTHETIC"] = "0"
os.environ["BIOHUB_GAPFILL_CONTEXT"] = "1"
os.environ["BIOHUB_GAPFILL_MAX_ADDED_FRAC"] = "0.03"
"""

# ─────────────────────────────────────────────────────────────────
# CELL 3 FIX: The artifact discovery function needs to check the
# correct thedivyanshpant model mount paths.  We add a new search
# path candidate that uses the model-mount layout.
# ─────────────────────────────────────────────────────────────────

# The old default search list in cell 3 has:
#   Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
#   Path(f"/kaggle/input/{slug}"),
#   Path(f"/kaggle/input/{slug}/{slug}"),
# We add a NEW first-priority entry:
#   Path(f"/kaggle/input/models/thedivyanshpant/{slug}/pytorch/default/1"),

CELL3_OLD_SLUG_SEARCH = (
    '        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),\n'
    '        Path(f"/kaggle/input/{slug}"),\n'
    '        Path(f"/kaggle/input/{slug}/{slug}"),'
)
CELL3_NEW_SLUG_SEARCH = (
    f'        Path(f"/kaggle/input/models/{USERNAME}/{{slug}}/pytorch/default/1"),  # primary\n'
    f'        Path(f"/kaggle/input/models/{USERNAME}/{{slug}}/pytorch/default/1/{{slug}}"),\n'
    '        Path(f"/kaggle/input/datasets/thedivyanshpant/{slug}"),\n'
    '        Path(f"/kaggle/input/{slug}"),\n'
    '        Path(f"/kaggle/input/{slug}/{slug}"),'
)

# For the deepcenter manifest/checkpoint fallback lines
OLD_DC_MANIFEST_DEFAULT = (
    '    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json",'
)
NEW_DC_MANIFEST_DEFAULT = (
    f'    "{DEEPCENTER_MOUNT}/ARTIFACT_MANIFEST.json",'
)

OLD_DC_CHECKPOINT_DEFAULT = (
    '    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt",'
)
NEW_DC_CHECKPOINT_DEFAULT = (
    f'    "{DEEPCENTER_MOUNT}/weights/full_frame_center/best.pt",'
)

OLD_PRIMARY_MANIFEST_DEFAULT = (
    '    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",'
)
NEW_PRIMARY_MANIFEST_DEFAULT = (
    f'    "{PRIMARY_MOUNT}/ARTIFACT_MANIFEST.json",'
)

OLD_SECONDARY_MANIFEST_DEFAULT = (
    '    "/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json",'
)
NEW_SECONDARY_MANIFEST_DEFAULT = (
    f'    "{SECONDARY_MOUNT}/ARTIFACT_MANIFEST.json",'
)

# For secondary slug discovery
OLD_SECONDARY_SLUG_PATH = (
    '        Path(f"/kaggle/input/datasets/pilkwang/{_secondary_slug}/ARTIFACT_MANIFEST.json"),'
)
NEW_SECONDARY_SLUG_PATH = (
    f'        Path(f"/kaggle/input/models/{USERNAME}/{{_secondary_slug}}/pytorch/default/1/ARTIFACT_MANIFEST.json"),'
)

# For deepcenter checkpoint discovery lines in cell 5 / 11
OLD_DC_SHORT = (
    '        Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),'
)
NEW_DC_SHORT = (
    f'        Path("{DEEPCENTER_MOUNT}"),'
)

OLD_DC_PILKWANG = (
    '        Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),'
)
NEW_DC_PILKWANG = (
    f'        Path("{DEEPCENTER_MOUNT}"),'
)


# ─────────────────────────────────────────────────────────────────
# MAIN PATCHING LOGIC
# ─────────────────────────────────────────────────────────────────

def patch_source(src: str, cell_index: int) -> str:
    """Apply all targeted patches to a single cell source string."""
    original = src

    # 1. Global find-replace for all path patterns (order matters: longest first)
    for old, new in GLOBAL_REPLACEMENTS:
        src = src.replace(old, new)

    # 2. Cell 0: inject missing env vars before final print
    if cell_index == 0:
        # Insert before the final print statements
        insert_before = 'print("BIOHUB_PRESET:'
        if insert_before in src and CELL0_EXTRA_ENVVARS.strip() not in src:
            src = src.replace(insert_before, CELL0_EXTRA_ENVVARS + insert_before)

    # 3. Cell-specific path fixes for pilkwang → thedivyanshpant model mounts
    if cell_index in (2, 3):
        # Primary artifact manifest default
        src = src.replace(OLD_PRIMARY_MANIFEST_DEFAULT, NEW_PRIMARY_MANIFEST_DEFAULT)
        src = src.replace(OLD_DC_MANIFEST_DEFAULT, NEW_DC_MANIFEST_DEFAULT)
        src = src.replace(OLD_DC_CHECKPOINT_DEFAULT, NEW_DC_CHECKPOINT_DEFAULT)

        # Slug-based search paths
        src = src.replace(CELL3_OLD_SLUG_SEARCH, ''.join(CELL3_NEW_SLUG_SEARCH))
        src = src.replace(
            '        Path(f"/kaggle/input/{_secondary_slug}/ARTIFACT_MANIFEST.json"),',
            f'        Path(f"/kaggle/input/models/{USERNAME}/{{_secondary_slug}}/pytorch/default/1/ARTIFACT_MANIFEST.json"),',
        )
        src = src.replace(OLD_SECONDARY_MANIFEST_DEFAULT, NEW_SECONDARY_MANIFEST_DEFAULT)

    if cell_index in (5, 11):
        src = src.replace(OLD_DC_SHORT, NEW_DC_SHORT)
        src = src.replace(OLD_DC_PILKWANG, NEW_DC_PILKWANG)

    return src


def main():
    print(f"Loading: {NOTEBOOK_PATH}")
    nb = json.load(NOTEBOOK_PATH.open())

    # Backup original
    import shutil
    shutil.copy2(NOTEBOOK_PATH, BACKUP_PATH)
    print(f"Backup saved: {BACKUP_PATH}")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src_list = cell["source"]
        src = "".join(src_list)
        patched = patch_source(src, i)
        if patched != src:
            print(f"  Cell {i:02d}: PATCHED ({len(src)} → {len(patched)} chars)")
        # Store back as list of lines
        if isinstance(src_list, list):
            lines = patched.splitlines(keepends=True)
            cell["source"] = lines
        else:
            cell["source"] = patched

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"\nWritten: {NOTEBOOK_PATH}")
    print(f"Size: {NOTEBOOK_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
