"""
generate_best_submission.py — Generates submission CSV for the winning dense mode (0.90017)
"""
import copy
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", Path(__file__).resolve().parents[2]))
CODE_DIR = Path(os.environ.get("BIOHUB_CODE_DIR", PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODE_DIR))

from leader_reproduction import config, evaluate, postprocess
from experiments.dense_volume.run_exp2 import (
    apply_overrides,
    restore_overrides,
    make_smooth_density_gap_fn,
    CONTROL_OVERRIDES,
)
from experiments.dense_volume.run_exp1 import load_graph_from_geff

config.set_all_env_vars()
_cfg = config.parse_config()

val_stems = config.VALIDATION_STEMS
pred_dir = config.REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer" / "split_0"
pred_th968_dir = config.REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer_th968" / "split_0"

# Graphs: 3 base + 1 dense th=0.968
active_graphs = {}
for stem in ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b"]:
    active_graphs[stem] = load_graph_from_geff(pred_dir / f"{stem}.geff")
active_graphs["6bba_05db0fb1"] = load_graph_from_geff(pred_th968_dir / "6bba_05db0fb1.geff")

out_sub_path = PROJECT_ROOT / "experiments" / "dense_volume" / "submission_dense_best.csv"

saved = apply_overrides(CONTROL_OVERRIDES)
orig_gap_fn = postprocess.close_single_frame_gaps
orig_motion_fn = postprocess.motion_relink_edges

postprocess.close_single_frame_gaps = make_smooth_density_gap_fn(scale_factor=3.0, dense_volume_only=True)

def custom_motion_fn(nodes_by_id, stats, learned_edge_probs=None):
    if len(nodes_by_id) >= 50000:
        saved_relaxed = postprocess.MOTION_RELINK_RELAXED_UM
        try:
            postprocess.MOTION_RELINK_RELAXED_UM = 7.0
            return orig_motion_fn(nodes_by_id, stats, learned_edge_probs)
        finally:
            postprocess.MOTION_RELINK_RELAXED_UM = saved_relaxed
    return orig_motion_fn(nodes_by_id, stats, learned_edge_probs)

postprocess.motion_relink_edges = custom_motion_fn

CSV_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"
]

row_id = 0
total_nodes = 0
total_edges = 0

deepcenter_bundle = postprocess.load_deepcenter_veto_detector()
shared_heatmap_cache = {}
shared_frame_cache = {}

try:
    with out_sub_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for stem in val_stems:
            raw_nodes_by_id, raw_edges = active_graphs[stem]
            nodes_copy = copy.deepcopy(raw_nodes_by_id)
            edges_copy = copy.deepcopy(raw_edges)

            p_nodes, p_edges, stats = postprocess.filter_output_graph(
                nodes_copy,
                edges_copy,
                dataset=stem,
                deepcenter_bundle=deepcenter_bundle,
                deepcenter_cache=shared_heatmap_cache,
                frame_cache=shared_frame_cache,
            )

            for node_id in sorted(p_nodes):
                node = p_nodes[node_id]
                writer.writerow({
                    "id": row_id,
                    "dataset": stem,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": max(0, int(round(float(node["z"])))),
                    "y": max(0, int(round(float(node["y"])))),
                    "x": max(0, int(round(float(node["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                })
                row_id += 1
                total_nodes += 1

            for edge in p_edges:
                writer.writerow({
                    "id": row_id,
                    "dataset": stem,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(edge["source_id"]),
                    "target_id": int(edge["target_id"]),
                })
                row_id += 1
                total_edges += 1
finally:
    postprocess.close_single_frame_gaps = orig_gap_fn
    postprocess.motion_relink_edges = orig_motion_fn
    restore_overrides(saved)

print(f"Submission written to {out_sub_path}")
print(f"Total rows: {row_id} (Nodes: {total_nodes}, Edges: {total_edges})")
