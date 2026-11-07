"""
run_exp1.py — Experiment 1: Dense-Volume Detection Threshold Sweep

Tests detection thresholds: 0.965 (baseline), 0.968, 0.970, 0.972
Condition: Activated only when volume/frame satisfies density criterion (predicted nodes > 50,000).
6bba_05db0fb1 is the only volume with > 50,000 nodes.
The other 3 volumes retain their known-good baseline graphs (det_threshold=0.965).

For each threshold:
- predicted nodes
- edge TP
- edge FP
- edge FN
- adjusted edge Jaccard
- overall score across all 4 volumes
"""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", Path(__file__).resolve().parents[2]))
CODE_DIR = Path(os.environ.get("BIOHUB_CODE_DIR", PROJECT_ROOT / "code"))
sys.path.insert(0, str(CODE_DIR))

from leader_reproduction import config, evaluate, postprocess

config.set_all_env_vars()
_cfg = config.parse_config()

CONTROL_OVERRIDES = {
    "MOTION_RELINK_VELOCITY_WEIGHT": 0.25,
    "MOTION_RELINK_LEARNED_BONUS": 1.25,
    "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10,
    "LEAF_PRUNE_MIN_EDGE_PROB": 0.30,
    "MOTION_RELINK_TIGHT_UM": 5.5,
    "SAFE_DIV_MAX_UM": 11.0,
    "SAFE_DIV_SISTER_MAX_UM": 16.0,
    "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0,
    "SAFE_DIV_DIVERGE_UM": 1.5,
    "MOTION_RELINK_RELAXED_UM": 9.0,
}

PP_SWEEP_KEYS = [
    "SAFE_DIV_MAX_UM", "SAFE_DIV_SISTER_MAX_UM", "SAFE_DIV_DIVERGE_UM",
    "SAFE_DIV_SISTER_SYMMETRY_TAU", "SAFE_DIV_EXISTING_CHILD_MAX_UM",
    "SAFE_DIV_FRAME_FRAC_CAP", "SAFE_DIV_GLOBAL_FRAC_CAP",
    "DEEPCENTER_SAFE_DIV_THRESHOLD", "DEEPCENTER_GAP_THRESHOLD",
    "GAP_CLOSE_UM", "OUTPUT_MIN_TRACK_LEN",
    "SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "MOTION_RELINK_TIGHT_UM",
    "MOTION_RELINK_RELAXED_UM", "GAP2_MAX_STEP_UM", "GAP2_MAX_TOTAL_UM",
    "MOTION_RELINK_LEARNED_BONUS", "MOTION_RELINK_VELOCITY_WEIGHT",
    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM", "LEAF_PRUNE_MIN_EDGE_PROB",
]
PP_BASE_CONFIG = {key: getattr(postprocess, key, None) for key in PP_SWEEP_KEYS}


def apply_overrides(overrides: dict):
    saved = {}
    for k, v in overrides.items():
        if hasattr(postprocess, k):
            saved[k] = getattr(postprocess, k)
            orig_val = PP_BASE_CONFIG.get(k)
            orig_type = type(orig_val) if orig_val is not None else float
            setattr(postprocess, k, orig_type(v))
    return saved


def restore_overrides(saved: dict):
    for k, v in saved.items():
        setattr(postprocess, k, v)


def run_prediction_for_threshold(th: float, gpu_id: int = 0) -> Path:
    """Run predict_unet_transformer.py for 6bba_05db0fb1 at threshold th."""
    repo_dir = config.REPO_DIR
    th_tag = f"th{int(round(th * 1000))}"
    method_name = f"unet_transformer_{th_tag}"
    out_geff = repo_dir / "predictions" / "deepanthpcu" / method_name / "split_0" / "6bba_05db0fb1.geff"

    if out_geff.exists():
        print(f"Prediction for {method_name} already exists: {out_geff}")
        return out_geff

    print(f"\n>>> Running neural prediction for 6bba_05db0fb1 with det_threshold={th} on GPU {gpu_id}...")
    splits_path = repo_dir / f"split_6bba_{th_tag}.json"
    splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": ["6bba_05db0fb1"]}], indent=2))

    cmd = [
        sys.executable,
        "scripts/predict_unet_transformer.py",
        "--data-dir", str(config.TEST_DIR),
        "--splits", str(splits_path.resolve()),
        "--split", "0",
        "--weights", str(repo_dir / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"),
        "--method", method_name,
        "--unet-batch-size", str(_cfg["unet_batch_size"]),
        "--det-threshold", str(th),
        "--ilp-edge-weight", str(_cfg["ilp_edge_weight"]),
        "--ilp-appearance-weight", str(_cfg["ilp_appearance_weight"]),
        "--ilp-disappearance-weight", str(_cfg["ilp_disappearance_weight"]),
        "--ilp-division-weight", str(_cfg["ilp_division_weight"]),
        "--use-ilp",
    ]

    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(repo_dir), env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"Prediction error (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
        raise RuntimeError(f"Prediction failed for threshold {th}")

    print(f"Prediction complete for threshold {th} in {dt:.1f}s. Output: {out_geff}")
    return out_geff


def load_graph_from_geff(pred_path: Path):
    pred_graph = evaluate.graph_from_geff(pred_path)
    raw_nodes_by_id = {}
    for row in pred_graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        raw_nodes_by_id[nid] = {
            "node_id": nid,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
    raw_edges = []
    for row in pred_graph.edge_attrs().iter_rows(named=True):
        ep = row.get("edge_prob") if hasattr(row, "get") else None
        raw_edges.append({
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "edge_prob": None if ep is None else float(ep),
        })
    return raw_nodes_by_id, raw_edges


def evaluate_combo(
    label: str,
    val_stems: list[str],
    val_raw_graphs: dict,
    val_gt: dict,
    shared_heatmap_cache: dict,
    shared_frame_cache: dict,
):
    saved = apply_overrides(CONTROL_OVERRIDES)
    rows = []
    t0 = time.time()
    try:
        for stem in val_stems:
            raw_nodes_by_id, raw_edges = val_raw_graphs[stem]
            nodes_copy = copy.deepcopy(raw_nodes_by_id)
            edges_copy = copy.deepcopy(raw_edges)

            buf = io.StringIO()
            with redirect_stdout(buf):
                p_nodes, p_edges, stats = postprocess.filter_output_graph(
                    nodes_copy,
                    edges_copy,
                    dataset=stem,
                    deepcenter_bundle=postprocess.DEEPCENTER_VETO_DETECTOR,
                    deepcenter_cache=shared_heatmap_cache,
                    frame_cache=shared_frame_cache,
                )

            gt_nodes_plain, gt_edges_plain, t_true = val_gt[stem]
            pred_nodes_plain = {
                nid: (int(n["t"]), float(n["z"]), float(n["y"]), float(n["x"]))
                for nid, n in p_nodes.items()
            }
            pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in p_edges]
            row = evaluate.score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
            row["stem"] = stem
            row["config"] = label
            row["final_pred_nodes"] = len(p_nodes)
            row["raw_pred_nodes"] = len(raw_nodes_by_id)
            rows.append(row)
    finally:
        restore_overrides(saved)

    summary = evaluate.aggregate_official(rows)
    summary["seconds"] = time.time() - t0
    summary["config"] = label
    summary["edge_tp"] = sum(r["edge_tp"] for r in rows)
    summary["edge_fp"] = sum(r["edge_fp"] for r in rows)
    summary["edge_fn"] = sum(r["edge_fn"] for r in rows)
    summary["edge_precision"] = summary["edge_tp"] / (summary["edge_tp"] + summary["edge_fp"]) if (summary["edge_tp"] + summary["edge_fp"]) > 0 else 0.0
    summary["edge_recall"] = summary["edge_tp"] / (summary["edge_tp"] + summary["edge_fn"]) if (summary["edge_tp"] + summary["edge_fn"]) > 0 else 0.0

    per_stem = {}
    for r in rows:
        per_stem[r["stem"]] = {
            "adj_jaccard": r["adjusted_edge_jaccard"],
            "tp": r["edge_tp"],
            "fp": r["edge_fp"],
            "fn": r["edge_fn"],
            "final_pred_nodes": r["final_pred_nodes"],
            "raw_pred_nodes": r["raw_pred_nodes"],
        }
    summary["per_stem"] = per_stem
    return summary


def main():
    val_stems = config.VALIDATION_STEMS
    pred_dir = config.REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer" / "split_0"

    # Load GT for all 4 stems
    val_gt = {}
    for stem in val_stems:
        gt_path = config.TRAIN_DIR / f"{stem}.geff"
        gt_graph = evaluate.graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = evaluate.graph_to_plain(gt_graph)
        t_true = evaluate.read_estimated_true_node_count(gt_path)
        val_gt[stem] = (gt_nodes_plain, gt_edges_plain, t_true)

    # Load standard graphs for 3 non-dense stems
    base_val_raw_graphs = {}
    for stem in ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b"]:
        base_val_raw_graphs[stem] = load_graph_from_geff(pred_dir / f"{stem}.geff")

    shared_heatmap_cache = {}
    shared_frame_cache = {}

    thresholds = [0.965, 0.968, 0.970, 0.972]
    results = []

    control_score = None
    control_6bba_score = None

    for th in thresholds:
        th_label = f"th_{th:.3f}"
        if abs(th - 0.965) < 1e-5:
            # Baseline graph
            dense_geff = pred_dir / "6bba_05db0fb1.geff"
        else:
            dense_geff = run_prediction_for_threshold(th, gpu_id=0)

        # Assemble the 4 graphs: 3 standard + 1 threshold-specific dense graph
        active_graphs = dict(base_val_raw_graphs)
        active_graphs["6bba_05db0fb1"] = load_graph_from_geff(dense_geff)

        res = evaluate_combo(
            label=th_label,
            val_stems=val_stems,
            val_raw_graphs=active_graphs,
            val_gt=val_gt,
            shared_heatmap_cache=shared_heatmap_cache,
            shared_frame_cache=shared_frame_cache,
        )

        overall_score = res["proxy_score"]
        dense_stats = res["per_stem"]["6bba_05db0fb1"]
        dense_score = dense_stats["adj_jaccard"]

        if th == 0.965:
            control_score = overall_score
            control_6bba_score = dense_score

        delta_overall = overall_score - control_score
        delta_6bba = dense_score - control_6bba_score

        print(
            f"\n[Threshold {th:.3f}] Overall: {overall_score:.5f} (d={delta_overall:+.5f}) | "
            f"6bba: {dense_score:.5f} (d={delta_6bba:+.5f}) | "
            f"RawNodes: {dense_stats['raw_pred_nodes']} -> FinalNodes: {dense_stats['final_pred_nodes']} | "
            f"TP: {dense_stats['tp']} | FP: {dense_stats['fp']} | FN: {dense_stats['fn']}"
        )

        results.append({
            "threshold": th,
            "condition": "dense_volume_only (nodes > 50,000)",
            "6bba_raw_nodes": dense_stats["raw_pred_nodes"],
            "6bba_final_nodes": dense_stats["final_pred_nodes"],
            "6bba_tp": dense_stats["tp"],
            "6bba_fp": dense_stats["fp"],
            "6bba_fn": dense_stats["fn"],
            "6bba_adj_jaccard": dense_score,
            "6bba_delta": delta_6bba,
            "overall_score": overall_score,
            "overall_delta": delta_overall,
            "overall_tp": res["edge_tp"],
            "overall_fp": res["edge_fp"],
            "overall_fn": res["edge_fn"],
            "overall_precision": res["edge_precision"],
            "overall_recall": res["edge_recall"],
            "44b6_0113": res["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
            "44b6_0b24": res["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
            "6bba_05b6": res["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
            "6bba_05db": dense_score,
        })

    out_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "experiments" / "dense_volume" / "exp1_results.csv"
    out_df.to_csv(out_path, index=False)
    print("\nExperiment 1 complete! Results saved to:", out_path)


if __name__ == "__main__":
    main()
