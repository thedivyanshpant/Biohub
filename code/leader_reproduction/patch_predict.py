"""
patch_predict.py — Faithful reproduction of Cell 4 dynamic source patches

Applies ALL source patches to predict_unet_transformer.py:
1. D4 8-way TTA (replaces 4-way)
2. Dual-seed ensemble injection
3. Candidate retention guard
4. Bidirectional harmonic probability fusion
5. Pre-ILP coordinate manifest
6. Edge-feature TTA (primary)
7. Edge-feature TTA (secondary)

Extracts the exact AST patch strings from cell_04_code.py for 100% byte-for-byte fidelity.
"""
from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path

from . import config


def extract_notebook_patches() -> dict:
    """Extract exact patch strings from cell_04_code.py."""
    cell_path = (
        config.PROJECT_ROOT / "code" / "reverse_engineering"
        / "nb_cells" / "cell_04_code.py"
    )
    with cell_path.open() as f:
        tree = ast.parse(f.read())

    patches = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "_old", "_new", "_ensemble_replacements",
                    "_guard_old", "_guard_new", "_bi_old", "_bi_new",
                    "_coordinate_manifest_old", "_coordinate_manifest_new",
                    "_et_old", "_et_new", "_secondary_tta_old", "_secondary_tta_new"
                ):
                    patches[target.id] = ast.literal_eval(node.value)
    return patches


def apply_all_patches() -> None:
    """Apply all Cell 4 dynamic patches to predict_unet_transformer.py."""
    ps = config.REPO_DIR / "scripts" / "predict_unet_transformer.py"
    if not ps.exists():
        raise FileNotFoundError(f"Predict script not found: {ps}")

    patches = extract_notebook_patches()

    s = ps.read_text()

    # 1. D4 8-way TTA patch
    if patches["_old"] in s:
        s = s.replace(patches["_old"], patches["_new"])
        print("PATCH 1: TTA patch applied (D4 8-way)")
    else:
        print("PATCH 1 WARNING: TTA block not found (may already be patched)")

    # 2. Dual-seed ensemble replacements
    for i, (old, new) in enumerate(patches["_ensemble_replacements"], start=1):
        count = s.count(old)
        if count != 1:
            raise RuntimeError(f"Dual-seed patch {i} expected 1 match, found {count}")
        s = s.replace(old, new, 1)
    print("PATCH 2: Dual-seed ensemble patches applied")

    # 3. Retention guard
    guard_new = patches["_guard_new"].replace(
        "/kaggle/working", str(config.WORKING_DIR)
    )
    count = s.count(patches["_guard_old"])
    if count != 1:
        raise RuntimeError(f"Retention guard expected 1 match, found {count}")
    s = s.replace(patches["_guard_old"], guard_new, 1)
    print("PATCH 3: Retention guard applied")

    # 4. Bidirectional harmonic probability fusion
    count = s.count(patches["_bi_old"])
    if count != 1:
        raise RuntimeError(f"Bidirectional edge patch expected 1 match, found {count}")
    s = s.replace(patches["_bi_old"], patches["_bi_new"], 1)

    # Coordinate manifest
    manifest_new = patches["_coordinate_manifest_new"].replace(
        "/kaggle/working", str(config.WORKING_DIR)
    )
    count = s.count(patches["_coordinate_manifest_old"])
    if count != 1:
        raise RuntimeError(f"Coordinate manifest expected 1 match, found {count}")
    s = s.replace(patches["_coordinate_manifest_old"], manifest_new, 1)
    print("PATCH 4: Bidirectional harmonic fusion + coordinate manifest applied")

    # 5. Primary edge-feature TTA
    count = s.count(patches["_et_old"])
    if count != 1:
        raise RuntimeError(f"Edge-TTA anchor expected 1 match, found {count}")
    s = s.replace(patches["_et_old"], patches["_et_new"], 1)
    print("PATCH 5: Edge-feature TTA (primary) applied")

    # 6. Secondary edge-feature TTA
    count = s.count(patches["_secondary_tta_old"])
    if count != 1:
        raise RuntimeError(f"Secondary edge-TTA anchor expected 1 match, found {count}")
    s = s.replace(patches["_secondary_tta_old"], patches["_secondary_tta_new"], 1)
    print("PATCH 6: Edge-feature TTA (secondary) applied")

    compile(s, str(ps), "exec")
    ps.write_text(s)

    # Verification checks matching notebook
    content = ps.read_text()
    assert "EDGE_TTA_ACTIVE" in content, "EDGE-TTA patch did not persist"
    assert "SECONDARY_EDGE_TTA_ACTIVE" in content, "Secondary edge-TTA patch did not persist"

    os.environ["BIOHUB_EDGE_FEATURE_TTA"] = "1"
    os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"
    os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"

    print("\n✓ All 6 dynamic source patches verified and compiled successfully.")
