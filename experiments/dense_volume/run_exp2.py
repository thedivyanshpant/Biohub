"""
run_exp2.py — Experiment 2: Density-Adaptive Gap Closing Investigation

Tests:
1. Reference variants: ref = 6.5 (baseline), 6.0, 5.5, 5.0, 4.0
2. Density gain / clamp variants: allowing denser frames to tighten gap closing
3. Proportional density scaling variants: threshold_um scaled by local spacing in dense frames
Evaluates all variants across all 4 validation volumes against the winning CONTROL combo.
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

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
    "GAP_DENSITY_REFERENCE_UM", "GAP_DENSITY_GAIN", "GAP_DENSITY_MAX_STEP_DELTA_UM",
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


def load_raw_and_gt(stems: list[str]):
    val_raw_graphs = {}
    val_gt = {}
    pred_dir = config.REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer" / "split_0"
    for stem in stems:
        gt_path = config.TRAIN_DIR / f"{stem}.geff"
        gt_graph = evaluate.graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = evaluate.graph_to_plain(gt_graph)
        t_true = evaluate.read_estimated_true_node_count(gt_path)
        val_gt[stem] = (gt_nodes_plain, gt_edges_plain, t_true)

        pred_path = pred_dir / f"{stem}.geff"
        pred_graph = evaluate.graph_from_geff(pred_path)
        raw_nodes_by_id = {}
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
            ep = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if ep is None else float(ep),
            })
        val_raw_graphs[stem] = (raw_nodes_by_id, raw_edges)
    return val_raw_graphs, val_gt


def evaluate_custom_pipeline(
    label: str,
    val_stems: list[str],
    val_raw_graphs: dict,
    val_gt: dict,
    shared_heatmap_cache: dict,
    shared_frame_cache: dict,
    custom_gap_fn=None,
    extra_overrides: dict | None = None,
):
    overrides = {**CONTROL_OVERRIDES, **(extra_overrides or {})}
    saved = apply_overrides(overrides)

    orig_gap_fn = postprocess.close_single_frame_gaps
    if custom_gap_fn is not None:
        postprocess.close_single_frame_gaps = custom_gap_fn

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
            rows.append(row)
    finally:
        if custom_gap_fn is not None:
            postprocess.close_single_frame_gaps = orig_gap_fn
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
        }
    summary["per_stem"] = per_stem
    return summary


def make_smooth_density_gap_fn(scale_factor: float = 3.0, dense_volume_only: bool = True):
    """
    Creates a density-adaptive gap closing function where in dense regimes
    the gap threshold is bounded by scale_factor * local_spacing.
    """
    orig_fn = postprocess.close_single_frame_gaps

    def custom_gap(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
        # Check density criterion
        if dense_volume_only and len(nodes_by_id) < 50000:
            return orig_fn(nodes_by_id, edges, stats, dataset, deepcenter_bundle, frame_cache, deepcenter_cache)

        # Standard close_single_frame_gaps with smooth adaptive threshold
        if not postprocess.OUTPUT_GAP_CLOSE or postprocess.GAP_CLOSE_MAX_GAP < 1 or not edges:
            return nodes_by_id, edges

        outgoing = {int(edge["source_id"]) for edge in edges}
        incoming = {int(edge["target_id"]) for edge in edges}
        incident = outgoing | incoming

        ends_by_t = {}
        starts_by_t = {}
        isolated_by_t = {}
        all_ids_by_t = {}
        for node_id, node in nodes_by_id.items():
            t = int(node["t"])
            all_ids_by_t.setdefault(t, []).append(node_id)
            if node_id not in outgoing: ends_by_t.setdefault(t, []).append(node_id)
            if node_id not in incoming: starts_by_t.setdefault(t, []).append(node_id)
            if node_id not in incident: isolated_by_t.setdefault(t, []).append(node_id)

        max_synthetic = min(
            postprocess.GAP_CLOSE_MAX_ADDED_ABS,
            max(1, int(round(len(nodes_by_id) * postprocess.GAP_CLOSE_MAX_ADDED_FRAC))) if postprocess.GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0,
        )
        next_id = postprocess._next_node_id(nodes_by_id)
        frame_cache = frame_cache if frame_cache is not None else {}
        deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
        used_starts = set()
        used_isolated = set()
        synthetic_added = 0
        new_edges = []

        density_cache = {}
        def frame_local_spacing(t: int):
            cached = density_cache.get(t)
            if cached is not None: return cached
            frame_ids = all_ids_by_t.get(t, [])
            if len(frame_ids) <= 1:
                res = {nid: postprocess.GAP_DENSITY_REFERENCE_UM for nid in frame_ids}
                density_cache[t] = res; return res
            positions = np.stack([postprocess._position_um(nodes_by_id[nid]) for nid in frame_ids])
            tree = cKDTree(positions)
            query_k = min(len(frame_ids), max(2, postprocess.GAP_DENSITY_NEIGHBORS + 1))
            distances, _ = tree.query(positions, k=query_k)
            if distances.ndim == 1: distances = distances[:, None]
            res = {}
            for idx, nid in enumerate(frame_ids):
                nb = distances[idx, 1:]
                nb = nb[np.isfinite(nb)]
                res[nid] = float(np.median(nb)) if nb.size else postprocess.GAP_DENSITY_REFERENCE_UM
            density_cache[t] = res
            stats["gap_density_nodes_scored"] += len(res)
            return res

        effective_gap_max = min(postprocess.GAP_CLOSE_MAX_GAP, 1)
        stats["gap_close_effective_max_gap"] = effective_gap_max
        for gap in range(1, effective_gap_max + 1):
            for t, end_ids in sorted(ends_by_t.items()):
                start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
                if not end_ids or not start_ids: continue
                end_points = [postprocess.node_point(nodes_by_id[eid]) for eid in end_ids]
                start_points = [postprocess.node_point(nodes_by_id[sid]) for sid in start_ids]
                threshold_um = postprocess.GAP_CLOSE_UM * (gap + 1)
                d = np.zeros((len(end_ids), len(start_ids)), dtype=np.float64)
                adaptive_threshold = np.full_like(d, threshold_um)

                source_spacing = frame_local_spacing(t)
                target_spacing = frame_local_spacing(t + gap + 1)

                for i, ep in enumerate(end_points):
                    for j, sp in enumerate(start_points):
                        d[i, j] = postprocess.point_distance_um(ep, sp)
                        local_sp = 0.5 * (source_spacing.get(end_ids[i], postprocess.GAP_DENSITY_REFERENCE_UM) + target_spacing.get(start_ids[j], postprocess.GAP_DENSITY_REFERENCE_UM))
                        # Smooth density constraint: threshold cannot exceed scale_factor * local_spacing
                        smooth_cap = scale_factor * local_sp * (gap + 1) / 2.0
                        adaptive_threshold[i, j] = min(threshold_um, smooth_cap)

                base_allowed = d <= threshold_um
                adaptive_allowed = d <= adaptive_threshold
                stats["gap_candidates"] += int(adaptive_allowed.sum())

                if not np.isfinite(d).any(): continue
                max_th = float(np.max(adaptive_threshold))
                big = max_th * 1000.0 + 1.0
                cost = np.where(adaptive_allowed, d, big)
                row_ind, col_ind = linear_sum_assignment(cost)

                for r, c in zip(row_ind, col_ind):
                    if not adaptive_allowed[r, c]: continue
                    source_id = end_ids[int(r)]
                    target_id = start_ids[int(c)]
                    if source_id in outgoing or target_id in used_starts: continue
                    source = nodes_by_id[source_id]
                    target = nodes_by_id[target_id]
                    mid_t = int(source["t"]) + gap
                    mid_point = (
                        (float(source["z"]) + float(target["z"])) / 2.0,
                        (float(source["y"]) + float(target["y"])) / 2.0,
                        (float(source["x"]) + float(target["x"])) / 2.0,
                    )
                    middle_id = None
                    middle_reused = False
                    if postprocess.GAP_CLOSE_REUSE_EXISTING:
                        candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                        if candidates:
                            distances = [postprocess.point_distance_um(postprocess.node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                            best_idx = int(np.argmin(distances))
                            if distances[best_idx] <= postprocess.GAP_CLOSE_REUSE_UM:
                                middle_id = candidates[best_idx]
                                middle_reused = True

                    if middle_id is None:
                        if synthetic_added >= max_synthetic:
                            stats["gap_skipped_node_cap"] += 1
                            continue
                        middle_id = next_id
                        next_id += 1
                        refined_point = postprocess.refine_synthetic_midpoint(dataset, mid_t, mid_point, frame_cache, stats)
                        nodes_by_id[middle_id] = {
                            "node_id": middle_id,
                            "t": mid_t,
                            "z": refined_point[0],
                            "y": refined_point[1],
                            "x": refined_point[2],
                            "gap_synthetic": 1,
                        }
                        synthetic_added += 1
                        stats["gap_inserted_synthetic"] += 1

                    middle = nodes_by_id[middle_id]
                    gap_span_um = float(d[r, c])
                    marginal_gap = gap_span_um >= postprocess.DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM
                    synthetic_middle = int(middle.get("gap_synthetic", 0)) == 1
                    requires_center_confirmation = (
                        postprocess.DEEPCENTER_GAP_VETO and marginal_gap and synthetic_middle
                    )
                    if requires_center_confirmation and not postprocess.deepcenter_accept_repair_point(
                        dataset, mid_t, postprocess.node_point(middle), deepcenter_bundle,
                        frame_cache, deepcenter_cache, stats, "gap", postprocess.DEEPCENTER_GAP_THRESHOLD,
                    ):
                        if int(middle.get("gap_synthetic", 0)) == 1:
                            nodes_by_id.pop(middle_id, None)
                            synthetic_added = max(0, synthetic_added - 1)
                            stats["gap_inserted_synthetic"] = max(0, stats["gap_inserted_synthetic"] - 1)
                        continue

                    if middle_reused:
                        used_isolated.add(middle_id)
                        stats["gap_reused_existing"] += 1

                    outgoing.add(source_id)
                    incoming.add(target_id)
                    used_starts.add(target_id)
                    edge1 = {
                        "source_id": source_id,
                        "target_id": middle_id,
                        "distance_um": postprocess.point_distance_um(postprocess.node_point(source), postprocess.node_point(middle)),
                        "edge_prob": None,
                    }
                    edge2 = {
                        "source_id": middle_id,
                        "target_id": target_id,
                        "distance_um": postprocess.point_distance_um(postprocess.node_point(middle), postprocess.node_point(target)),
                        "edge_prob": None,
                    }
                    new_edges.extend([edge1, edge2])
                    stats["gap_pairs_selected"] += 1
                    stats["gap_added_edges"] += 2
                    if not middle_reused:
                        stats["gap_added_nodes"] += 1

        if new_edges:
            edges = [*edges, *new_edges]
        return nodes_by_id, edges

    return custom_gap


def main():
    val_stems = config.VALIDATION_STEMS
    val_raw_graphs, val_gt = load_raw_and_gt(val_stems)
    shared_heatmap_cache = {}
    shared_frame_cache = {}

    print(f"Loaded {len(val_stems)} validation volumes.")

    # 1. CONTROL
    print("\n--- Evaluating CONTROL (Winning Combo) ---")
    control_res = evaluate_custom_pipeline(
        label="control",
        val_stems=val_stems,
        val_raw_graphs=val_raw_graphs,
        val_gt=val_gt,
        shared_heatmap_cache=shared_heatmap_cache,
        shared_frame_cache=shared_frame_cache,
    )
    control_score = control_res["proxy_score"]
    control_6bba = control_res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
    print(f"CONTROL Overall: {control_score:.5f} | 6bba: {control_6bba:.5f} (TP={control_res['per_stem']['6bba_05db0fb1']['tp']}, FP={control_res['per_stem']['6bba_05db0fb1']['fp']}, FN={control_res['per_stem']['6bba_05db0fb1']['fn']})")

    results = []
    results.append({
        "experiment": "control",
        "variant": "control_winning_combo",
        "overall_score": control_score,
        "overall_delta": 0.0,
        "score_6bba": control_6bba,
        "delta_6bba": 0.0,
        "edge_tp": control_res["edge_tp"],
        "edge_fp": control_res["edge_fp"],
        "edge_fn": control_res["edge_fn"],
        "edge_precision": control_res["edge_precision"],
        "edge_recall": control_res["edge_recall"],
        "44b6_0113": control_res["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
        "44b6_0b24": control_res["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
        "6bba_05b6": control_res["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
        "6bba_05db": control_res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"],
    })

    # Test Reference variants: 6.5, 6.0, 5.5, 5.0, 4.0
    for ref_val in [6.5, 6.0, 5.5, 5.0, 4.0]:
        label = f"gap_ref_{ref_val}"
        res = evaluate_custom_pipeline(
            label=label,
            val_stems=val_stems,
            val_raw_graphs=val_raw_graphs,
            val_gt=val_gt,
            shared_heatmap_cache=shared_heatmap_cache,
            shared_frame_cache=shared_frame_cache,
            extra_overrides={"GAP_DENSITY_REFERENCE_UM": ref_val},
        )
        sc = res["proxy_score"]
        sc_6bba = res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
        print(f"[{label}] Overall: {sc:.5f} (d={sc-control_score:+.5f}) | 6bba: {sc_6bba:.5f} (d={sc_6bba-control_6bba:+.5f})")
        results.append({
            "experiment": "exp2_reference",
            "variant": label,
            "overall_score": sc,
            "overall_delta": sc - control_score,
            "score_6bba": sc_6bba,
            "delta_6bba": sc_6bba - control_6bba,
            "edge_tp": res["edge_tp"],
            "edge_fp": res["edge_fp"],
            "edge_fn": res["edge_fn"],
            "edge_precision": res["edge_precision"],
            "edge_recall": res["edge_recall"],
            "44b6_0113": res["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
            "44b6_0b24": res["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
            "6bba_05b6": res["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
            "6bba_05db": res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"],
        })

    # Test Gain & Clamp variants: allow denser frames to meaningfully restrict threshold
    for clamp_val, gain_val in [(0.25, 0.08), (0.50, 0.10), (1.00, 0.15)]:
        label = f"gap_clamp_{clamp_val}_gain_{gain_val}"
        res = evaluate_custom_pipeline(
            label=label,
            val_stems=val_stems,
            val_raw_graphs=val_raw_graphs,
            val_gt=val_gt,
            shared_heatmap_cache=shared_heatmap_cache,
            shared_frame_cache=shared_frame_cache,
            extra_overrides={
                "GAP_DENSITY_MAX_STEP_DELTA_UM": clamp_val,
                "GAP_DENSITY_GAIN": gain_val,
            },
        )
        sc = res["proxy_score"]
        sc_6bba = res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
        print(f"[{label}] Overall: {sc:.5f} (d={sc-control_score:+.5f}) | 6bba: {sc_6bba:.5f} (d={sc_6bba-control_6bba:+.5f})")
        results.append({
            "experiment": "exp2_clamp_gain",
            "variant": label,
            "overall_score": sc,
            "overall_delta": sc - control_score,
            "score_6bba": sc_6bba,
            "delta_6bba": sc_6bba - control_6bba,
            "edge_tp": res["edge_tp"],
            "edge_fp": res["edge_fp"],
            "edge_fn": res["edge_fn"],
            "edge_precision": res["edge_precision"],
            "edge_recall": res["edge_recall"],
            "44b6_0113": res["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
            "44b6_0b24": res["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
            "6bba_05b6": res["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
            "6bba_05db": res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"],
        })

    # Test Smooth Proportional Scaling: threshold <= k * local_spacing (only on dense volume)
    for k_val in [2.0, 2.5, 3.0, 3.5, 4.0]:
        label = f"smooth_density_cap_{k_val}x"
        fn = make_smooth_density_gap_fn(scale_factor=k_val, dense_volume_only=True)
        res = evaluate_custom_pipeline(
            label=label,
            val_stems=val_stems,
            val_raw_graphs=val_raw_graphs,
            val_gt=val_gt,
            shared_heatmap_cache=shared_heatmap_cache,
            shared_frame_cache=shared_frame_cache,
            custom_gap_fn=fn,
        )
        sc = res["proxy_score"]
        sc_6bba = res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
        print(f"[{label}] Overall: {sc:.5f} (d={sc-control_score:+.5f}) | 6bba: {sc_6bba:.5f} (d={sc_6bba-control_6bba:+.5f})")
        results.append({
            "experiment": "exp2_smooth_scaling",
            "variant": label,
            "overall_score": sc,
            "overall_delta": sc - control_score,
            "score_6bba": sc_6bba,
            "delta_6bba": sc_6bba - control_6bba,
            "edge_tp": res["edge_tp"],
            "edge_fp": res["edge_fp"],
            "edge_fn": res["edge_fn"],
            "edge_precision": res["edge_precision"],
            "edge_recall": res["edge_recall"],
            "44b6_0113": res["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
            "44b6_0b24": res["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
            "6bba_05b6": res["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
            "6bba_05db": res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"],
        })

    # Save results
    out_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "experiments" / "dense_volume" / "exp2_results.csv"
    out_df.to_csv(out_path, index=False)
    print("\nExperiment 2 complete! Results saved to:", out_path)


if __name__ == "__main__":
    main()
