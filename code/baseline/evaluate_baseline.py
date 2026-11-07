#!/usr/bin/env python3
"""
Biohub Baseline: Local Evaluation Interface.
Evaluates predicted nodes and edges against ground truth GEFF graphs using the official competition metric.
"""

import os
import sys
import json
import zarr
from typing import Dict, List, Tuple

# Add code directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evaluation.evaluator import evaluate_dataset, evaluate_run, EvaluationResult, DatasetsResult


def load_ground_truth_geff(geff_path: str):
    """
    Loads ground truth nodes, edges, and estimated_number_of_nodes from a GEFF Zarr v3 store.
    """
    if not os.path.exists(geff_path):
        raise FileNotFoundError(f"Ground truth GEFF store not found: {geff_path}")

    root = zarr.open_group(geff_path, mode="r")
    with open(os.path.join(geff_path, "zarr.json")) as f:
        meta = json.load(f)
    n_total = meta.get("attributes", {}).get("geff", {}).get("extra", {}).get("estimated_number_of_nodes")

    node_ids = root["nodes/ids"][:]
    t_vals = root["nodes/props/t/values"][:]
    z_vals = root["nodes/props/z/values"][:]
    y_vals = root["nodes/props/y/values"][:]
    x_vals = root["nodes/props/x/values"][:]
    edge_ids = root["edges/ids"][:]

    gt_nodes = {}
    for nid, t, z, y, x in zip(node_ids, t_vals, z_vals, y_vals, x_vals):
        gt_nodes[int(nid)] = (int(t), int(z), int(y), int(x))

    gt_edges = [(int(src), int(tgt)) for src, tgt in edge_ids]
    return gt_nodes, gt_edges, n_total


def evaluate_single_dataset(
    pred_nodes: Dict[int, Tuple[int, int, int, int]],
    pred_edges: List[Tuple[int, int]],
    geff_path: str,
    scale: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    match_distance: float = 7.0
) -> EvaluationResult:
    """Evaluates a single dataset against its ground truth GEFF."""
    gt_nodes, gt_edges, n_total = load_ground_truth_geff(geff_path)
    return evaluate_dataset(
        pred_nodes=pred_nodes,
        pred_edges=pred_edges,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        n_total=n_total,
        scale=scale,
        max_distance=match_distance
    )


def evaluate_multiple_datasets(
    results_dict: Dict[str, EvaluationResult]
) -> DatasetsResult:
    """Aggregates multiple per-dataset evaluation results using official micro-averaging."""
    return evaluate_run(results_dict)
