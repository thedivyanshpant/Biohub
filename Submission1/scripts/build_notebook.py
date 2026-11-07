import json
import os
import re
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", _SCRIPT_DIR.parents[1]))
nb_cells_dir = str(_PROJECT_ROOT / "code/reverse_engineering/nb_cells")
output_ipynb = str(_SCRIPT_DIR.parent / "notebook/submission1.ipynb")

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_ipynb), exist_ok=True)

cells = []

# Order of files
files = [
    "cell_00_code.py",
    "cell_01_code.py",
    "cell_02_code.py",
    "cell_03_code.py",
    "cell_04_code.py",
    "cell_05_code.py",
    "cell_06_code.py",
    "cell_07_code.py",
    "cell_08_code.py",
    "cell_09_code.py",
    "cell_10_code.py",
    "cell_11_code.py"
]

overrides = {
    "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.1,
    "LEAF_PRUNE_MIN_EDGE_PROB": 0.3,
    "MOTION_RELINK_LEARNED_BONUS": 1.25,
    "MOTION_RELINK_RELAXED_UM": 9.0,
    "MOTION_RELINK_TIGHT_UM": 5.5,
    "MOTION_RELINK_VELOCITY_WEIGHT": 0.25,
    "SAFE_DIV_DIVERGE_UM": 1.5,
    "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0,
    "SAFE_DIV_MAX_UM": 11.0,
    "SAFE_DIV_SISTER_MAX_UM": 16.0,
}

extra_settings = {
    "BIOHUB_SECONDARY_WEIGHTS": "/kaggle/input/biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/edge_predictor_best.pth",
    "BIOHUB_SECONDARY_EDGE_WEIGHT": "0.15",
    "BIOHUB_SECONDARY_DETECTION_WEIGHT": "0.80",
    "BIOHUB_SECONDARY_LINK_MODE": "low_margin_consensus",
    "BIOHUB_SECONDARY_MIX_TEMPERATURE": "1",
    "BIOHUB_SECONDARY_LOW_MARGIN_MAX": "0.35",
    "BIOHUB_DUAL_SEED_EDGE_THRESHOLD": "0.48",
}

def create_cell(source):
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }

for filename in files:
    filepath = os.path.join(nb_cells_dir, filename)
    with open(filepath, "r") as f:
        content = f.read()
    
    if filename == "cell_00_code.py":
        # Add comment at the top
        content = "# Submission 1 — PPSWEEP winning config pre-applied (score: 0.90017)\n" + content
        
        # Override values
        for k, v in overrides.items():
            env_key = f'BIOHUB_{k}'
            val_str = str(v)
            # Regex to replace existing assignment or add to end if not exists
            pattern = re.compile(rf'os\.environ\["{env_key}"\]\s*=\s*".*?"')
            if pattern.search(content):
                content = pattern.sub(f'os.environ["{env_key}"] = "{val_str}"', content)
            else:
                content += f'\nos.environ["{env_key}"] = "{val_str}"\n'
                
        # Extra settings
        for env_key, val_str in extra_settings.items():
            pattern = re.compile(rf'os\.environ\["{env_key}"\]\s*=\s*".*?"')
            if pattern.search(content):
                content = pattern.sub(f'os.environ["{env_key}"] = "{val_str}"', content)
            else:
                content += f'\nos.environ["{env_key}"] = "{val_str}"\n'

    cells.append(create_cell(content.strip()))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(output_ipynb, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook written to {output_ipynb}")
print(f"Total cells: {len(cells)}")
print(f"File size: {os.path.getsize(output_ipynb)} bytes")
