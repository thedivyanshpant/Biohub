#!/usr/bin/env python3
"""
Biohub Classical Detector V2: Pipeline Evaluation Harness.
Connects any V2 detector with the UNCHANGED baseline Hungarian tracker and official evaluator.
"""

import os
import sys
import time
import json
import zarr
import numpy as np
import scipy.spatial.distance as sp_dist
from scipy.optimize import linear_sum_assignment

# Add code directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from evaluation.evaluator import evaluate_dataset, evaluate_run, EvaluationResult, match_nodes_bipartite

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MAX_LINK_DISTANCE = 15.0  # µm
MATCH_DISTANCE = 7.0      # µm


def load_gt_geff(geff_path: str):
    """Loads ground truth nodes, edges, and estimated_number_of_nodes."""
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

    gt_nodes = {int(nid): (int(t), int(z), int(y), int(x)) for nid, t, z, y, x in zip(node_ids, t_vals, z_vals, y_vals, x_vals)}
    gt_edges = [(int(src), int(tgt)) for src, tgt in edge_ids]
    return gt_nodes, gt_edges, n_total


def evaluate_node_precision_recall_f1(pred_nodes, gt_nodes, scale=SCALE, match_distance=MATCH_DISTANCE):
    """
    Evaluates node centroid Precision, Recall, F1 using bipartite matching at 7.0 µm physical cutoff.
    """
    pred_by_t = {}
    for pid, (t, z, y, x) in pred_nodes.items():
        pred_by_t.setdefault(t, []).append((pid, z, y, x))

    gt_by_t = {}
    for gid, (t, z, y, x) in gt_nodes.items():
        gt_by_t.setdefault(t, []).append((gid, z, y, x))

    total_tp = 0
    all_t = set(pred_by_t.keys()) | set(gt_by_t.keys())

    for t in all_t:
        plist = pred_by_t.get(t, [])
        glist = gt_by_t.get(t, [])
        if not plist or not glist:
            continue

        p_ids = np.array([p[0] for p in plist], dtype=np.int64)
        p_coords = np.array([[p[1], p[2], p[3]] for p in plist], dtype=np.float64)

        g_ids = np.array([g[0] for g in glist], dtype=np.int64)
        g_coords = np.array([[g[1], g[2], g[3]] for g in glist], dtype=np.float64)

        p2g, _ = match_nodes_bipartite(p_coords, p_ids, g_coords, g_ids, scale=tuple(scale), max_distance=match_distance)
        total_tp += len(p2g)

    num_pred = len(pred_nodes)
    num_gt = len(gt_nodes)
    tp = total_tp
    fp = num_pred - tp
    fn = num_gt - tp

    prec = (tp / num_pred) if num_pred > 0 else 0.0
    rec = (tp / num_gt) if num_gt > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "node_tp": tp,
        "node_fp": fp,
        "node_fn": fn,
        "node_precision": prec,
        "node_recall": rec,
        "node_f1": f1,
    }


def run_pipeline_on_volume(
    dataset_name: str,
    zarr_path: str,
    geff_path: str,
    detector_fn,
    detector_kwargs: dict = None,
    refinement_fn=None,
    refinement_kwargs: dict = None,
    filter_fn=None,
    filter_kwargs: dict = None,
):
    """
    Runs detection on every 3D frame, executes optional sub-voxel refinement and confidence filtering,
    executes unchanged Hungarian tracking, and returns comprehensive evaluation results.
    """
    if detector_kwargs is None:
        detector_kwargs = {}
    if refinement_kwargs is None:
        refinement_kwargs = {}
    if filter_kwargs is None:
        filter_kwargs = {}

    zg = zarr.open_group(zarr_path, mode="r")
    arr = zg["0"]
    num_t = arr.shape[0]

    t_start = time.time()
    pred_nodes = {}  # nid -> (t, z, y, x)
    pred_edges = []  # (src_id, tgt_id)

    node_counter = 1
    prev_centroids = {}
    all_shifts = []

    for t in range(num_t):
        vol = arr[t][:]  # Native shape (64, 256, 256), uint16

        # 1. Detection (integer peaks + confidence)
        raw_dets = detector_fn(vol, **detector_kwargs)

        # 2. Sub-voxel refinement (if specified)
        if refinement_fn is not None and len(raw_dets) > 0:
            # Check if detector returns smoothed or normalized volume, or pass raw vol
            refined_dets = refinement_fn(vol, raw_dets, **refinement_kwargs)
            # Record shifts
            raw_coords = np.array([[d[0], d[1], d[2]] for d in raw_dets], dtype=np.float64)
            ref_coords = np.array([[d[0], d[1], d[2]] for d in refined_dets], dtype=np.float64)
            diff = (ref_coords - raw_coords) * SCALE
            dists = np.linalg.norm(diff, axis=1)
            all_shifts.extend(dists.tolist())
            curr_dets = refined_dets
        else:
            curr_dets = raw_dets

        # 3. Confidence / feature filtering (if specified)
        if filter_fn is not None and len(curr_dets) > 0:
            curr_dets = filter_fn(curr_dets, **filter_kwargs)

        curr_centroids = {}
        for d in curr_dets:
            nid = node_counter
            node_counter += 1
            curr_centroids[nid] = (d[0], d[1], d[2])
            pred_nodes[nid] = (t, d[0], d[1], d[2])

        # Tracking (UNCHANGED baseline Hungarian matching)
        if prev_centroids and curr_centroids:
            prev_ids = list(prev_centroids.keys())
            curr_ids = list(curr_centroids.keys())

            prev_coords = np.array([prev_centroids[pid] for pid in prev_ids], dtype=np.float64)
            curr_coords = np.array([curr_centroids[cid] for cid in curr_ids], dtype=np.float64)

            prev_phys = prev_coords * SCALE
            curr_phys = curr_coords * SCALE

            dist_mat = sp_dist.cdist(prev_phys, curr_phys)
            row_ind, col_ind = linear_sum_assignment(dist_mat)

            for r, c in zip(row_ind, col_ind):
                if dist_mat[r, c] <= MAX_LINK_DISTANCE:
                    pred_edges.append((prev_ids[r], curr_ids[c]))

        prev_centroids = curr_centroids

    runtime = time.time() - t_start

    # Ground Truth Evaluation
    gt_nodes, gt_edges, n_total = load_gt_geff(geff_path)

    node_metrics = evaluate_node_precision_recall_f1(pred_nodes, gt_nodes)
    edge_eval = evaluate_dataset(
        pred_nodes=pred_nodes,
        pred_edges=pred_edges,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
        n_total=n_total,
        scale=tuple(SCALE),
        max_distance=MATCH_DISTANCE
    )

    return {
        "dataset": dataset_name,
        "runtime": runtime,
        "num_pred_nodes": len(pred_nodes),
        "num_gt_nodes": len(gt_nodes),
        "num_pred_edges": len(pred_edges),
        "num_gt_edges": len(gt_edges),
        "node_precision": node_metrics["node_precision"],
        "node_recall": node_metrics["node_recall"],
        "node_f1": node_metrics["node_f1"],
        "edge_tp": edge_eval.edge_tp,
        "edge_fp": edge_eval.edge_fp,
        "edge_fn": edge_eval.edge_fn,
        "edge_jaccard": edge_eval.edge_jaccard,
        "total_node_ratio": edge_eval.total_node_ratio,
        "adj_edge_jaccard": edge_eval.adj_edge_jaccard,
        "division_jaccard": edge_eval.division_jaccard,
        "combined_score": edge_eval.score,
        "mean_shift_um": float(np.mean(all_shifts)) if all_shifts else 0.0,
        "raw_edge_eval": edge_eval,
    }
