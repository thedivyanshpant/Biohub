"""
ppsweep_worker.py — Distributed PPSWEEP worker pinned to a single GPU.

Evaluates a designated shard of post-processing configurations.
Maintains persistent in-memory DeepCenter heatmap cache for ultra-fast evaluations.
"""
from __future__ import annotations

import argparse
import copy as _copy
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

args = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPSWEEP Worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--candidates", type=str, required=True, help="Comma-separated candidate names")
    parser.add_argument("--shard-out", type=Path, required=True)
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import numpy as np
import pandas as pd
import tracksdata as td
import torch

PROJECT_CODE = Path(__file__).resolve().parent.parent
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from leader_reproduction import config, postprocess, evaluate

config.set_all_env_vars()
_cfg = config.parse_config()

REPO_DIR = config.REPO_DIR
TRAIN_DIR = config.TRAIN_DIR

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

PP_BASE_CONFIG = {key: getattr(postprocess, key) for key in PP_SWEEP_KEYS}

# Master dictionary of candidate configs
ALL_CANDIDATES: dict[str, dict] = {
    "gap45": {"GAP_CLOSE_UM": 4.5},
    "tight55": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "relaxed9": {"MOTION_RELINK_RELAXED_UM": 9.0},
    "bonus125": {"MOTION_RELINK_LEARNED_BONUS": 1.25},
    "gap2step40": {"GAP2_MAX_STEP_UM": 4.0},
    "reuse28": {"GAP_CLOSE_REUSE_UM": 2.8},
    "dcgap035": {"DEEPCENTER_GAP_THRESHOLD": 0.35},
    "dcsd015": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.15},
    "dcsd010": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10},
    "divwide": {
        "SAFE_DIV_MAX_UM": 11.0,
        "SAFE_DIV_SISTER_MAX_UM": 16.0,
        "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0,
    },
    "sym075": {"SAFE_DIV_SISTER_SYMMETRY_TAU": 0.75},
    "diverge150": {"SAFE_DIV_DIVERGE_UM": 1.5},
    "t55_dcsd015": {"MOTION_RELINK_TIGHT_UM": 5.5, "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.15},
    "t55_dcsd010": {"MOTION_RELINK_TIGHT_UM": 5.5, "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10},
    "t55_divwide": {
        "MOTION_RELINK_TIGHT_UM": 5.5,
        "SAFE_DIV_MAX_UM": 11.0,
        "SAFE_DIV_SISTER_MAX_UM": 16.0,
        "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0,
    },
    "t55_diverge150": {"MOTION_RELINK_TIGHT_UM": 5.5, "SAFE_DIV_DIVERGE_UM": 1.5},
    "leaf030": {"LEAF_PRUNE_MIN_EDGE_PROB": 0.30},
    "leaf040": {"LEAF_PRUNE_MIN_EDGE_PROB": 0.40},
    "t55_leaf030": {"MOTION_RELINK_TIGHT_UM": 5.5, "LEAF_PRUNE_MIN_EDGE_PROB": 0.30},
    "minlen5": {"OUTPUT_MIN_TRACK_LEN": 5},
    "rescue085": {"SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": 0.85},
    "dcgap018": {"DEEPCENTER_GAP_THRESHOLD": 0.18},
    "vel075": {"MOTION_RELINK_VELOCITY_WEIGHT": 0.75},
    "vel025": {"MOTION_RELINK_VELOCITY_WEIGHT": 0.25},
}

VALIDATOR_DIVISION_WEIGHT = 0.1


def pp_apply(candidate_config: dict) -> dict:
    saved = {key: getattr(postprocess, key) for key in candidate_config}
    for key, value in candidate_config.items():
        if key not in PP_SWEEP_KEYS:
            raise KeyError(f"{key} is not a sweepable post-process constant")
        orig_type = type(PP_BASE_CONFIG[key])
        setattr(postprocess, key, orig_type(value))
    return saved


def pp_restore(saved: dict) -> None:
    for key, value in saved.items():
        setattr(postprocess, key, value)


def nodes_by_id_to_plain(nodes_by_id: dict[int, dict[str, object]]) -> dict[int, tuple]:
    return {
        nid: (int(n["t"]), float(n["z"]), float(n["y"]), float(n["x"]))
        for nid, n in nodes_by_id.items()
    }


def load_raw_and_gt(stems: list[str]) -> tuple[dict, dict]:
    val_raw_graphs: dict[str, tuple[dict, list]] = {}
    val_gt: dict[str, tuple[dict, list, float]] = {}

    pred_dir = REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer" / "split_0"
    for stem in stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        gt_graph = evaluate.graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = evaluate.graph_to_plain(gt_graph)
        t_true = evaluate.read_estimated_true_node_count(gt_path)
        val_gt[stem] = (gt_nodes_plain, gt_edges_plain, t_true)

        pred_path = pred_dir / f"{stem}.geff"
        pred_graph = evaluate.graph_from_geff(pred_path)
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            raw_nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })
        val_raw_graphs[stem] = (raw_nodes_by_id, raw_edges)

    return val_raw_graphs, val_gt


def evaluate_config(
    candidate_config: dict,
    label: str,
    val_stems: list[str],
    val_raw_graphs: dict,
    val_gt: dict,
    shared_heatmap_cache: dict,
    shared_frame_cache: dict,
) -> dict:
    saved = pp_apply(candidate_config)
    rows = []
    t0 = time.time()
    try:
        for stem in val_stems:
            raw_nodes_by_id, raw_edges = val_raw_graphs[stem]
            nodes_copy = _copy.deepcopy(raw_nodes_by_id)
            edges_copy = _copy.deepcopy(raw_edges)

            buf = io.StringIO()
            with redirect_stdout(buf):
                processed_nodes, processed_edges, _stage_stats = postprocess.filter_output_graph(
                    nodes_copy,
                    edges_copy,
                    dataset=stem,
                    deepcenter_bundle=postprocess.DEEPCENTER_VETO_DETECTOR,
                    deepcenter_cache=shared_heatmap_cache,
                    frame_cache=shared_frame_cache,
                )

            gt_nodes_plain, gt_edges_plain, t_true = val_gt[stem]
            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
            pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]
            row = evaluate.score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
            row["stem"] = stem
            row["config"] = label
            rows.append(row)
    finally:
        pp_restore(saved)

    summary = evaluate.aggregate_official(rows)
    summary["seconds"] = time.time() - t0
    summary["config"] = label
    summary["overrides"] = json.dumps(candidate_config, sort_keys=True)

    summary["edge_tp"] = sum(r["edge_tp"] for r in rows)
    summary["edge_fp"] = sum(r["edge_fp"] for r in rows)
    summary["edge_fn"] = sum(r["edge_fn"] for r in rows)
    summary["edge_precision"] = (
        summary["edge_tp"] / (summary["edge_tp"] + summary["edge_fp"])
        if (summary["edge_tp"] + summary["edge_fp"]) > 0 else 0.0
    )
    summary["edge_recall"] = (
        summary["edge_tp"] / (summary["edge_tp"] + summary["edge_fn"])
        if (summary["edge_tp"] + summary["edge_fn"]) > 0 else 0.0
    )
    summary["raw_jaccard"] = (
        summary["edge_tp"] / (summary["edge_tp"] + summary["edge_fp"] + summary["edge_fn"])
        if (summary["edge_tp"] + summary["edge_fp"] + summary["edge_fn"]) > 0 else 0.0
    )

    summary["node_fn"] = sum(r["missed_gt_nodes"] for r in rows)
    summary["node_fp"] = sum(r["spurious_pred_nodes"] for r in rows)
    summary["node_tp"] = sum(len(val_gt[r["stem"]][0]) - r["missed_gt_nodes"] for r in rows)

    # Prefix proxy
    prefix_rows: dict[str, list[dict]] = {}
    for row in rows:
        prefix_rows.setdefault(str(row["stem"]).split("_")[0], []).append(row)
    summary["prefix_proxy"] = {}
    for prefix, prows in prefix_rows.items():
        w = sum(r["weight"] for r in prows) or 1
        p_tp = sum(r["div_tp"] for r in prows)
        p_fp = sum(r["div_fp"] for r in prows)
        p_fn = sum(r["div_fn"] for r in prows)
        p_adj = sum(r["adjusted_edge_jaccard"] * r["weight"] for r in prows) / w
        summary["prefix_proxy"][prefix] = (
            p_adj + VALIDATOR_DIVISION_WEIGHT * evaluate.edge_jaccard(p_tp, p_fp, p_fn)
        )

    return summary


def main():
    val_stems = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
    candidate_names = [c.strip() for c in args.candidates.split(",") if c.strip()]

    print(f"[Worker {args.worker_id}] Initializing on GPU {args.gpu_id}...")
    print(f"[Worker {args.worker_id}] Assigned {len(candidate_names)} candidates: {candidate_names}")

    val_raw_graphs, val_gt = load_raw_and_gt(val_stems)

    # Shared caches across runs on this GPU worker
    shared_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    shared_frame_cache: dict[int, np.ndarray] = {}

    results = []

    # First, run base configuration to pre-warm the cache and compute baseline
    print(f"[Worker {args.worker_id}] Evaluating base configuration (warmup)...")
    base_summary = evaluate_config({}, "base", val_stems, val_raw_graphs, val_gt, shared_heatmap_cache, shared_frame_cache)
    base_summary["prefix_proxy_json"] = json.dumps(base_summary["prefix_proxy"], sort_keys=True)
    results.append(base_summary)
    print(
        f"[Worker {args.worker_id}] BASE: proxy={base_summary['proxy_score']:.5f} "
        f"adj={base_summary['adjusted_edge_jaccard']:.5f} [{base_summary['seconds']:.1f}s] "
        f"(cached {len(shared_heatmap_cache)} heatmaps)"
    )

    for i, name in enumerate(candidate_names, 1):
        if name not in ALL_CANDIDATES:
            print(f"[Worker {args.worker_id}] Warning: Unknown candidate {name}")
            continue
        cfg = ALL_CANDIDATES[name]
        summary = evaluate_config(cfg, name, val_stems, val_raw_graphs, val_gt, shared_heatmap_cache, shared_frame_cache)
        summary["prefix_proxy_json"] = json.dumps(summary["prefix_proxy"], sort_keys=True)
        results.append(summary)
        delta = summary["proxy_score"] - base_summary["proxy_score"]
        print(
            f"[Worker {args.worker_id}] [{i}/{len(candidate_names)}] {name:<16} "
            f"proxy={summary['proxy_score']:.5f} ({delta:+.5f}) adj={summary['adjusted_edge_jaccard']:.5f} "
            f"rec={summary['edge_recall']*100:.2f}% [{summary['seconds']:.1f}s]"
        )

    # Save shard dataframe
    df = pd.DataFrame(results)
    args.shard_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.shard_out, index=False)
    print(f"[Worker {args.worker_id}] Wrote {len(df)} configurations to {args.shard_out}")


if __name__ == "__main__":
    main()
