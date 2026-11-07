import os
import sys
from pathlib import Path
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import config

config.set_all_env_vars()
cfg = config.parse_config()

print("Checking config overrides...")
expected = {
    "BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT": "0.25",
    "BIOHUB_MOTION_RELINK_LEARNED_BONUS": "1.25",
    "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.10",
    "BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB": "0.30",
    "BIOHUB_MOTION_RELINK_TIGHT_UM": "5.5",
    "BIOHUB_SAFE_DIV_MAX_UM": "11.0",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "16.0",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "12.0",
    "BIOHUB_SAFE_DIV_DIVERGE_UM": "1.5",
    "BIOHUB_MOTION_RELINK_RELAXED_UM": "9.0"
}
for k, v in expected.items():
    if os.environ.get(k) != v:
        print(f"FAIL: {k} = {os.environ.get(k)} != {v}")
        sys.exit(1)

print("Checking checkpoints...")
checkpoints = [
    Path(os.environ["BIOHUB_MODEL_ARTIFACTS"]) / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth",
    Path(os.environ["BIOHUB_SECONDARY_WEIGHTS"]),
    Path(os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"])
]

for ckpt in checkpoints:
    if not ckpt.exists():
        print(f"FAIL: Missing {ckpt}")
        sys.exit(1)
    print(f"Found: {ckpt}")

print("PASS: Smoke test completed successfully.")
