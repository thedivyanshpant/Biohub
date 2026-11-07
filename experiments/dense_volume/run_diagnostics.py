"""
run_diagnostics.py — Experiments 4 & 5: Density Stratification & Error Attribution

Analyzes 6bba_05db0fb1 (and all 4 validation volumes) under the winning CONTROL combo:
1. Experiment 4: Timepoint-by-timepoint analysis (t=0..99)
   - predicted nodes, GT nodes, ratio
   - edge TP, FP, FN, local edge Jaccard
   - nearest-neighbor spacing / candidate density
   - gap-closing edges, motion-relink edges, ILP edges
2. Experiment 5: Component error attribution
   - edge TP, FP, FN per post-processing stage
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

# Set environment
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
PP_BASE_CONFIG = {key: getattr(postprocess, key) for key in PP_SWEEP_KEYS}


def apply_overrides(overrides: dict):
    saved = {k: getattr(postprocess, k) for k in overrides}
    for k, v in overrides.items():
        orig_type = type(PP_BASE_CONFIG[k])
        setattr(postprocess, k, orig_type(v))
    return saved


def restore_overrides(saved: dict):
    for k, v in saved.items():
        setattr(postprocess, k, v)


def run_instrumented_pipeline(stem: str):
    """Run postprocessing on stem with full provenance tracking on every edge."""
    pred_dir = config.REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer" / "split_0"
    gt_path = config.TRAIN_DIR / f"{stem}.geff"
    pred_path = pred_dir / f"{stem}.geff"

    gt_graph = evaluate.graph_from_geff(gt_path)
    gt_nodes_plain, gt_edges_plain = evaluate.graph_to_plain(gt_graph)
    t_true = evaluate.read_estimated_true_node_count(gt_path)

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
            "source_tag": "raw_ilp",
        })

    # Track raw ILP edges by frame
    ilp_edges_by_t = {}
    for e in raw_edges:
        src = raw_nodes_by_id.get(e["source_id"])
        if src:
            t = int(src["t"])
            ilp_edges_by_t[t] = ilp_edges_by_t.get(t, 0) + 1

    saved = apply_overrides(CONTROL_OVERRIDES)
    try:
        nodes_copy = copy.deepcopy(raw_nodes_by_id)
        edges_copy = copy.deepcopy(raw_edges)

        deepcenter_bundle = postprocess.load_deepcenter_veto_detector()
        shared_heatmap_cache = {}
        shared_frame_cache = {}

        # Instrument the pipeline stages directly
        stats = {
            "raw_edges": len(edges_copy),
            "dropped_nonconsecutive_edges": 0,
            "dropped_long_edges": 0,
            "dropped_multi_parent_edges": 0,
            "dropped_multi_child_edges": 0,
            "dropped_division_edges": 0,
            "gap_candidates": 0,
            "gap_pairs_selected": 0,
            "gap_reused_existing": 0,
            "gap_inserted_synthetic": 0,
            "gap_added_nodes": 0,
            "gap_added_edges": 0,
            "gap_skipped_node_cap": 0,
            "gap_density_nodes_scored": 0,
            "gap_density_candidates_expanded": 0,
            "gap_density_candidates_restricted": 0,
            "gap_density_selected_outside_base": 0,
            "gap_density_step_delta_milli_sum": 0,
            "gap_refined_synthetic": 0,
            "gap_refine_failed": 0,
            "gap_refine_rejected_shift": 0,
            "pruned_isolated_nodes": 0,
            "motion_relink_edges": 0,
            "motion_relink_tight_edges": 0,
            "motion_relink_relaxed_edges": 0,
            "motion_relink_frames": 0,
            "motion_relink_replaced_raw_edges": 0,
            "motion_relink_fallback_raw": 0,
            "motion_relink_skipped_large_frame": 0,
            "gap2_candidates": 0,
            "gap2_pairs_selected": 0,
            "gap2_added_nodes": 0,
            "gap2_added_edges": 0,
            "gap2_skipped_cap": 0,
            "safe_division_candidates": 0,
            "safe_division_geometric_candidates": 0,
            "safe_divisions_added": 0,
            "safe_division_skipped_cap": 0,
            "safe_division_mutual_nn_rejected": 0,
            "safe_division_divergence_rejected": 0,
            "safe_division_symmetry_rejected": 0,
            "deepcenter_gap_checked": 0,
            "deepcenter_gap_bypassed_strong_motion": 0,
            "deepcenter_gap_bypassed_observed_node": 0,
            "deepcenter_gap_accepted": 0,
            "deepcenter_gap_rejected": 0,
            "deepcenter_gap_missing": 0,
            "deepcenter_safe_div_checked": 0,
            "deepcenter_safe_div_accepted": 0,
            "deepcenter_safe_div_rejected": 0,
            "deepcenter_safe_div_missing": 0,
            "short_track_components_removed": 0,
            "short_track_nodes_removed": 0,
            "short_track_edges_removed": 0,
            "short_track_filter_skipped_all": 0,
            "short_track_rescue_triggered": 0,
            "short_track_rescue_components": 0,
            "short_track_rescue_nodes": 0,
            "short_track_rescue_budget": 0,
            "linefit_smoothed_nodes": 0,
            "linefit_skipped_nodes": 0,
            "leaf_prune_nodes": 0,
            "leaf_prune_edges": 0,
        }

        # 1. Edge filter & motion relink
        edges = []
        for edge in edges_copy:
            source = nodes_copy.get(int(edge["source_id"]))
            target = nodes_copy.get(int(edge["target_id"]))
            if source is None or target is None:
                continue
            if postprocess.OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
                stats["dropped_nonconsecutive_edges"] += 1
                continue
            distance_um = postprocess.edge_distance_um(source, target)
            edge["distance_um"] = distance_um
            if postprocess.OUTPUT_EDGE_MAX_UM > 0 and distance_um > postprocess.OUTPUT_EDGE_MAX_UM:
                stats["dropped_long_edges"] += 1
                continue
            edges.append(edge)

        # Custom motion relink with edge tagging
        if postprocess.OUTPUT_MOTION_RELINK:
            learned_edge_probs = {}
            for edge in edges:
                prob = edge.get("edge_prob")
                if prob is not None:
                    try:
                        learned_edge_probs[(int(edge["source_id"]), int(edge["target_id"]))] = float(prob)
                    except:
                        pass
            
            # Run motion relink and record tight vs relaxed tags
            ids_by_t = {}
            for nid, node in nodes_copy.items():
                ids_by_t.setdefault(int(node["t"]), []).append(nid)
            for ids in ids_by_t.values():
                ids.sort()

            position_um = {nid: postprocess._position_um(node) for nid, node in nodes_copy.items()}
            predecessor_position_um = {}
            motion_selected = []

            def learned_prob(s, tg):
                v = learned_edge_probs.get((s, tg), 0.0)
                if not np.isfinite(v): return 0.0
                if v < 0 or v > 1: v = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, v))))
                return float(np.clip(v, 0.0, 1.0))

            def assign_pass(s_ids, tg_ids, gate_um):
                if not s_ids or not tg_ids: return []
                big = gate_um * 1000.0 + 1.0
                cost = np.full((len(s_ids), len(tg_ids)), big, dtype=np.float64)
                raw_d = np.full_like(cost, np.inf)
                mot_d = np.full_like(cost, np.inf)
                pr_m = np.zeros_like(cost)
                for i, sid in enumerate(s_ids):
                    spos = position_um[sid]
                    ppos = predecessor_position_um.get(sid)
                    pred_pos = spos if ppos is None else spos + postprocess.MOTION_RELINK_VELOCITY_WEIGHT * (spos - ppos)
                    for j, tid in enumerate(tg_ids):
                        tpos = position_um[tid]
                        raw = float(np.linalg.norm(tpos - spos))
                        if raw > gate_um: continue
                        mot = float(np.linalg.norm(tpos - pred_pos))
                        p = learned_prob(sid, tid)
                        raw_d[i, j] = raw; mot_d[i, j] = mot; pr_m[i, j] = p
                        cost[i, j] = mot + 0.05 * raw - postprocess.MOTION_RELINK_LEARNED_BONUS * p
                r_ind, c_ind = linear_sum_assignment(cost)
                res = []
                for r, c in zip(r_ind, c_ind):
                    if cost[r, c] >= big: continue
                    res.append((s_ids[int(r)], tg_ids[int(c)], float(raw_d[r, c]), float(mot_d[r, c]), float(pr_m[r, c])))
                return res

            for t in sorted(ids_by_t):
                s_ids = ids_by_t.get(t, [])
                tg_ids = ids_by_t.get(t + 1, [])
                if not s_ids or not tg_ids: continue
                un_s = set(s_ids); un_tg = set(tg_ids)
                frame_m = []
                for pass_name, gate in (("tight", postprocess.MOTION_RELINK_TIGHT_UM), ("relaxed", postprocess.MOTION_RELINK_RELAXED_UM)):
                    ps_s = [n for n in s_ids if n in un_s]
                    ps_tg = [n for n in tg_ids if n in un_tg]
                    matches = assign_pass(ps_s, ps_tg, gate)
                    for sid, tid, raw, mot, pr in matches:
                        if sid not in un_s or tid not in un_tg: continue
                        un_s.remove(sid); un_tg.remove(tid)
                        frame_m.append((sid, tid, raw, mot, pass_name, pr))
                        if pass_name == "tight": stats["motion_relink_tight_edges"] += 1
                        else: stats["motion_relink_relaxed_edges"] += 1
                for sid, tid, raw, mot, pass_name, pr in frame_m:
                    motion_selected.append({
                        "source_id": sid,
                        "target_id": tid,
                        "edge_prob": pr,
                        "distance_um": raw,
                        "source_tag": f"motion_{pass_name}",
                    })
                    predecessor_position_um[tid] = position_um[sid]

            if motion_selected:
                stats["motion_relink_replaced_raw_edges"] = len(edges)
                edges = motion_selected
            else:
                stats["motion_relink_fallback_raw"] = 1

        # Single parent repair
        if postprocess.OUTPUT_SINGLE_PARENT_REPAIR and edges:
            best_by_target = {}
            for edge in edges:
                tid = int(edge["target_id"])
                prev = best_by_target.get(tid)
                if prev is None or postprocess.edge_sort_key(edge) > postprocess.edge_sort_key(prev):
                    best_by_target[tid] = edge
            kept_ids = {id(e) for e in best_by_target.values()}
            stats["dropped_multi_parent_edges"] = sum(1 for e in edges if id(e) not in kept_ids)
            edges = [e for e in edges if id(e) in kept_ids]

        # Single child repair
        if postprocess.OUTPUT_SINGLE_CHILD_REPAIR and edges:
            best_by_source = {}
            for edge in edges:
                sid = int(edge["source_id"])
                prev = best_by_source.get(sid)
                if prev is None or postprocess.edge_sort_key(edge) > postprocess.edge_sort_key(prev):
                    best_by_source[sid] = edge
            kept_ids = {id(e) for e in best_by_source.values()}
            stats["dropped_multi_child_edges"] = sum(1 for e in edges if id(e) not in kept_ids)
            edges = [e for e in edges if id(e) in kept_ids]

        # Single frame gap close (tagging synthetic vs reused)
        # We run the standard close_single_frame_gaps and tag untagged edges
        before_gap_edges = set(id(e) for e in edges)
        nodes_copy, edges = postprocess.close_single_frame_gaps(
            nodes_copy, edges, stats, dataset=stem,
            deepcenter_bundle=deepcenter_bundle,
            frame_cache=shared_frame_cache,
            deepcenter_cache=shared_heatmap_cache,
        )
        for e in edges:
            if id(e) not in before_gap_edges:
                src_node = nodes_copy.get(int(e["source_id"]))
                tgt_node = nodes_copy.get(int(e["target_id"]))
                is_synth = (src_node and src_node.get("gap_synthetic")) or (tgt_node and tgt_node.get("gap_synthetic"))
                e["source_tag"] = "gap1_synthetic" if is_synth else "gap1_reused"

        # Gap2 recovery
        before_gap2_edges = set(id(e) for e in edges)
        nodes_copy, edges = postprocess.recover_strict_gap2(nodes_copy, edges, stats, dataset=stem)
        for e in edges:
            if id(e) not in before_gap2_edges:
                e["source_tag"] = "gap2_synthetic"

        # Safe divisions
        before_div_edges = set(id(e) for e in edges)
        edges = postprocess.add_safe_divisions_postlink(
            nodes_copy, edges, stats, dataset=stem,
            deepcenter_bundle=deepcenter_bundle,
            frame_cache=shared_frame_cache,
            deepcenter_cache=shared_heatmap_cache,
        )
        for e in edges:
            if id(e) not in before_div_edges:
                e["source_tag"] = "safe_division"

        # Prune isolated nodes
        if postprocess.OUTPUT_PRUNE_ISOLATED:
            incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
            if incident:
                kept_nodes = {node_id: node for node_id, node in nodes_copy.items() if node_id in incident}
                nodes_copy = kept_nodes
                edges = [e for e in edges if int(e["source_id"]) in nodes_copy and int(e["target_id"]) in nodes_copy]

        # Short track filter
        nodes_copy, edges = postprocess.filter_short_track_components(nodes_copy, edges, stats)

        # Leaf prune
        if postprocess.LEAF_PRUNE_MIN_EDGE_PROB > 0.0:
            nodes_copy, edges = postprocess.prune_weak_leaf_nodes(nodes_copy, edges, stats)

        # Linefit smoothing
        nodes_copy = postprocess.linefit_smooth_output_graph(nodes_copy, edges, stats)

    finally:
        restore_overrides(saved)

    return nodes_copy, edges, gt_nodes_plain, gt_edges_plain, t_true, ilp_edges_by_t, raw_nodes_by_id


def analyze_stem(stem: str):
    print(f"=== Running Detailed Diagnostic Analysis for {stem} ===")
    nodes, edges, gt_nodes, gt_edges, t_true, ilp_edges_by_t, raw_nodes_by_id = run_instrumented_pipeline(stem)

    pred_nodes_plain = {
        nid: (int(n["t"]), float(n["z"]), float(n["y"]), float(n["x"]))
        for nid, n in nodes.items()
    }
    pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in edges]

    # Overall evaluation
    eval_row = evaluate.score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes, gt_edges, t_true)
    print(f"Overall score for {stem}: {eval_row['adjusted_edge_jaccard']:.5f} (TP={eval_row['edge_tp']}, FP={eval_row['edge_fp']}, FN={eval_row['edge_fn']})")

    # Bipartite matching
    pred_to_gt, gt_to_pred = evaluate.match_nodes_bipartite(pred_nodes_plain, gt_nodes, max_dist=7.0)

    gt_outgoing = {}
    gt_incoming_source = {}
    for s, t in gt_edges:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming_source[t] = s

    # Node counts by t
    pred_by_t = {}
    for nid, node in nodes.items():
        pred_by_t.setdefault(int(node["t"]), []).append(nid)

    gt_by_t = {}
    for gid, gval in gt_nodes.items():
        gt_by_t.setdefault(int(gval[0]), []).append(gid)

    # Nearest neighbor distances per frame (in microns)
    voxel_scale = np.array([0.28, 0.1625, 0.1625])  # z, y, x
    nn_dist_by_t = {}
    for t, nids in pred_by_t.items():
        if len(nids) <= 1:
            nn_dist_by_t[t] = 0.0
            continue
        positions = np.array([
            [float(nodes[nid]["z"]), float(nodes[nid]["y"]), float(nodes[nid]["x"])]
            for nid in nids
        ]) * voxel_scale
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        nn_dist_by_t[t] = float(np.median(dists[:, 1]))

    # Analyze edges by provenance tag (Experiment 5)
    provenance_stats = {}
    edge_tag_by_pair = {}
    for edge in edges:
        tag = edge.get("source_tag", "unknown")
        edge_tag_by_pair[(int(edge["source_id"]), int(edge["target_id"]))] = tag
        if tag not in provenance_stats:
            provenance_stats[tag] = {"total": 0, "tp": 0, "fp": 0, "spurious": 0}
        provenance_stats[tag]["total"] += 1

        s = int(edge["source_id"])
        t = int(edge["target_id"])
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)

        is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, ())
        if is_tp:
            provenance_stats[tag]["tp"] += 1
        else:
            is_fp = (mt is not None and mt in gt_incoming_source) or (ms is not None and bool(gt_outgoing.get(ms)))
            if is_fp:
                provenance_stats[tag]["fp"] += 1
            else:
                provenance_stats[tag]["spurious"] += 1

    # Timepoint analysis (Experiment 4)
    # Group edges and GT edges by frame t
    all_t = sorted(set(list(range(100))))
    timepoint_records = []

    matched_gt_edges = set()
    for s, t in pred_edges_plain:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        if ms is not None and mt is not None and mt in gt_outgoing.get(ms, ()):
            matched_gt_edges.add((ms, mt))

    gt_edges_by_t = {}
    for s, t in gt_edges:
        gt_t = int(gt_nodes[s][0])
        gt_edges_by_t.setdefault(gt_t, []).append((s, t))

    edges_by_src_t = {}
    for edge in edges:
        s = int(edge["source_id"])
        src_t = int(nodes[s]["t"])
        edges_by_src_t.setdefault(src_t, []).append(edge)

    for t in all_t:
        p_nodes_count = len(pred_by_t.get(t, []))
        g_nodes_count = len(gt_by_t.get(t, []))
        ratio = p_nodes_count / g_nodes_count if g_nodes_count > 0 else 0.0

        t_edges = edges_by_src_t.get(t, [])
        tp_t = 0
        fp_t = 0
        gap_edges_t = 0
        motion_tight_t = 0
        motion_relaxed_t = 0
        safe_div_t = 0

        for edge in t_edges:
            tag = edge.get("source_tag", "unknown")
            if "gap" in tag:
                gap_edges_t += 1
            elif "tight" in tag:
                motion_tight_t += 1
            elif "relaxed" in tag:
                motion_relaxed_t += 1
            elif "safe_division" in tag:
                safe_div_t += 1

            s = int(edge["source_id"])
            tgt = int(edge["target_id"])
            ms = pred_to_gt.get(s)
            mt = pred_to_gt.get(tgt)
            if ms is not None and mt is not None and mt in gt_outgoing.get(ms, ()):
                tp_t += 1
            else:
                is_fp = (mt is not None and mt in gt_incoming_source) or (ms is not None and bool(gt_outgoing.get(ms)))
                if is_fp:
                    fp_t += 1

        t_gt_edges = gt_edges_by_t.get(t, [])
        fn_t = sum(1 for e in t_gt_edges if e not in matched_gt_edges)
        jaccard_t = tp_t / (tp_t + fp_t + fn_t) if (tp_t + fp_t + fn_t) > 0 else (1.0 if not t_gt_edges and not t_edges else 0.0)

        timepoint_records.append({
            "t": t,
            "pred_nodes": p_nodes_count,
            "gt_nodes": g_nodes_count,
            "ratio": ratio,
            "edge_tp": tp_t,
            "edge_fp": fp_t,
            "edge_fn": fn_t,
            "local_jaccard": jaccard_t,
            "median_nn_spacing_um": nn_dist_by_t.get(t, 0.0),
            "ilp_edges": ilp_edges_by_t.get(t, 0),
            "motion_tight_edges": motion_tight_t,
            "motion_relaxed_edges": motion_relaxed_t,
            "gap_edges": gap_edges_t,
            "safe_div_edges": safe_div_t,
        })

    df_timepoint = pd.DataFrame(timepoint_records)
    return df_timepoint, provenance_stats, eval_row


def main():
    out_dir = PROJECT_ROOT / "experiments" / "dense_volume"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run for 6bba_05db0fb1
    df_tp, prov_stats, eval_row = analyze_stem("6bba_05db0fb1")
    df_tp.to_csv(out_dir / "6bba_timepoint_analysis.csv", index=False)
    with (out_dir / "6bba_error_attribution.json").open("w") as f:
        json.dump(prov_stats, f, indent=2)

    print("\n--- EXPERIMENT 5: ERROR SOURCE ATTRIBUTION (6bba_05db0fb1) ---")
    print(f"{'Component':<20} | {'Total Edges':<12} | {'TP':<6} | {'FP':<6} | {'Precision':<10}")
    print("-" * 65)
    for comp, cstats in sorted(prov_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        prec = cstats['tp'] / (cstats['tp'] + cstats['fp']) if (cstats['tp'] + cstats['fp']) > 0 else 0.0
        print(f"{comp:<20} | {cstats['total']:<12} | {cstats['tp']:<6} | {cstats['fp']:<6} | {prec:.4f}")

    print("\n--- EXPERIMENT 4: TIMEPOINT STRATIFICATION SUMMARY (6bba_05db0fb1) ---")
    # Identify top error frames
    df_tp["total_errors"] = df_tp["edge_fp"] + df_tp["edge_fn"]
    top_error_frames = df_tp.sort_values(by="total_errors", ascending=False).head(15)
    print("Top 15 Error Timepoints:")
    print(top_error_frames[["t", "pred_nodes", "gt_nodes", "edge_tp", "edge_fp", "edge_fn", "local_jaccard", "median_nn_spacing_um", "total_errors"]].to_string(index=False))

    # Density regimes
    early = df_tp[(df_tp["t"] >= 0) & (df_tp["t"] <= 30)]
    mid = df_tp[(df_tp["t"] >= 31) & (df_tp["t"] <= 70)]
    late = df_tp[(df_tp["t"] >= 71) & (df_tp["t"] <= 99)]

    print("\nRegime Breakdown:")
    for name, sub in [("Early (t=0..30)", early), ("Mid (t=31..70)", mid), ("Late (t=71..99)", late)]:
        sub_tp = sub["edge_tp"].sum()
        sub_fp = sub["edge_fp"].sum()
        sub_fn = sub["edge_fn"].sum()
        sub_jacc = sub_tp / (sub_tp + sub_fp + sub_fn) if (sub_tp + sub_fp + sub_fn) > 0 else 0.0
        avg_spacing = sub["median_nn_spacing_um"].mean()
        avg_nodes = sub["pred_nodes"].mean()
        print(f"{name:<18}: TP={sub_tp:<4} FP={sub_fp:<3} FN={sub_fn:<3} LocalJaccard={sub_jacc:.4f} AvgNodes={avg_nodes:.1f} AvgNN={avg_spacing:.2f}um")

    print("\nDone! Results saved to:", out_dir)


if __name__ == "__main__":
    main()
