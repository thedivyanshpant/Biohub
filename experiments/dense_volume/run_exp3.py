"""
run_exp3.py — Experiment 3: Combined Dense Mode

Combines:
1. Best detection threshold from Exp 1: 0.968 (volume-aware, nodes > 50k)
2. Best gap-closing from Exp 2: smooth_density_cap (3.0x local spacing)
Also tests:
3. Combined + relaxed motion suppression on dense volume (since Exp 5 proved motion_relaxed is 46.8% precision, 25 FP vs 22 TP)

Evaluates on ALL FOUR validation volumes against CONTROL.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", Path(__file__).resolve().parents[2]))
CODE_DIR = Path(os.environ.get("BIOHUB_CODE_DIR", PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODE_DIR))

from leader_reproduction import config, evaluate, postprocess
from experiments.dense_volume.run_exp2 import (
    load_raw_and_gt,
    apply_overrides,
    restore_overrides,
    make_smooth_density_gap_fn,
    CONTROL_OVERRIDES,
)
from experiments.dense_volume.run_exp1 import load_graph_from_geff

config.set_all_env_vars()
_cfg = config.parse_config()


def evaluate_custom_combined(
    label: str,
    val_stems: list[str],
    val_raw_graphs: dict,
    val_gt: dict,
    shared_heatmap_cache: dict,
    shared_frame_cache: dict,
    custom_gap_fn=None,
    extra_overrides: dict | None = None,
    dense_motion_tight_only: bool = False,
):
    overrides = {**CONTROL_OVERRIDES, **(extra_overrides or {})}
    saved = apply_overrides(overrides)

    orig_gap_fn = postprocess.close_single_frame_gaps
    if custom_gap_fn is not None:
        postprocess.close_single_frame_gaps = custom_gap_fn

    orig_motion_fn = postprocess.motion_relink_edges
    if dense_motion_tight_only:
        def custom_motion_fn(nodes_by_id, stats, learned_edge_probs=None):
            # If dense volume, tighten relaxed pass to eliminate negative-precision edges
            if len(nodes_by_id) >= 50000:
                saved_relaxed = postprocess.MOTION_RELINK_RELAXED_UM
                try:
                    # Stricter relaxed pass: gate at 7.0 um instead of 9.0 um
                    postprocess.MOTION_RELINK_RELAXED_UM = 7.0
                    return orig_motion_fn(nodes_by_id, stats, learned_edge_probs)
                finally:
                    postprocess.MOTION_RELINK_RELAXED_UM = saved_relaxed
            return orig_motion_fn(nodes_by_id, stats, learned_edge_probs)
        postprocess.motion_relink_edges = custom_motion_fn

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
        if dense_motion_tight_only:
            postprocess.motion_relink_edges = orig_motion_fn
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


def main():
    val_stems = config.VALIDATION_STEMS
    pred_dir = config.REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer" / "split_0"
    pred_th968_dir = config.REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer_th968" / "split_0"

    # Load GT
    val_gt = {}
    for stem in val_stems:
        gt_path = config.TRAIN_DIR / f"{stem}.geff"
        gt_graph = evaluate.graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = evaluate.graph_to_plain(gt_graph)
        t_true = evaluate.read_estimated_true_node_count(gt_path)
        val_gt[stem] = (gt_nodes_plain, gt_edges_plain, t_true)

    # Base raw graphs (th=0.965)
    base_val_raw_graphs = {}
    for stem in val_stems:
        base_val_raw_graphs[stem] = load_graph_from_geff(pred_dir / f"{stem}.geff")

    # Volume-aware th=0.968 graphs (6bba at 0.968, others at 0.965)
    th968_val_raw_graphs = dict(base_val_raw_graphs)
    th968_val_raw_graphs["6bba_05db0fb1"] = load_graph_from_geff(pred_th968_dir / "6bba_05db0fb1.geff")

    shared_heatmap_cache = {}
    shared_frame_cache = {}

    print("Evaluating Experiment 3 Combinations...")

    # 1. CONTROL
    control_res = evaluate_custom_combined(
        label="control",
        val_stems=val_stems,
        val_raw_graphs=base_val_raw_graphs,
        val_gt=val_gt,
        shared_heatmap_cache=shared_heatmap_cache,
        shared_frame_cache=shared_frame_cache,
    )
    control_score = control_res["proxy_score"]
    control_6bba = control_res["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
    print(f"CONTROL Overall: {control_score:.5f} | 6bba: {control_6bba:.5f} (TP={control_res['per_stem']['6bba_05db0fb1']['tp']}, FP={control_res['per_stem']['6bba_05db0fb1']['fp']}, FN={control_res['per_stem']['6bba_05db0fb1']['fn']})")

    results = [{
        "configuration": "control_winning_combo",
        "overall_score": control_score,
        "overall_delta": 0.0,
        "6bba_score": control_6bba,
        "6bba_delta": 0.0,
        "overall_tp": control_res["edge_tp"],
        "overall_fp": control_res["edge_fp"],
        "overall_fn": control_res["edge_fn"],
        "overall_precision": control_res["edge_precision"],
        "overall_recall": control_res["edge_recall"],
        "6bba_tp": control_res["per_stem"]["6bba_05db0fb1"]["tp"],
        "6bba_fp": control_res["per_stem"]["6bba_05db0fb1"]["fp"],
        "6bba_fn": control_res["per_stem"]["6bba_05db0fb1"]["fn"],
        "44b6_0113": control_res["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
        "44b6_0b24": control_res["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
        "6bba_05b6": control_res["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
        "6bba_05db": control_6bba,
    }]

    # 2. COMBINATION 1: Exp 1 Best Alone (th=0.968 on dense volume)
    res_exp1 = evaluate_custom_combined(
        label="exp1_th0968_dense_only",
        val_stems=val_stems,
        val_raw_graphs=th968_val_raw_graphs,
        val_gt=val_gt,
        shared_heatmap_cache=shared_heatmap_cache,
        shared_frame_cache=shared_frame_cache,
    )
    sc = res_exp1["proxy_score"]
    sc_6b = res_exp1["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
    print(f"[Combo 1: Exp 1 (th=0.968)] Overall: {sc:.5f} (d={sc-control_score:+.5f}) | 6bba: {sc_6b:.5f} (d={sc_6b-control_6bba:+.5f})")
    results.append({
        "configuration": "exp1_th0968_dense_only",
        "overall_score": sc,
        "overall_delta": sc - control_score,
        "6bba_score": sc_6b,
        "6bba_delta": sc_6b - control_6bba,
        "overall_tp": res_exp1["edge_tp"],
        "overall_fp": res_exp1["edge_fp"],
        "overall_fn": res_exp1["edge_fn"],
        "overall_precision": res_exp1["edge_precision"],
        "overall_recall": res_exp1["edge_recall"],
        "6bba_tp": res_exp1["per_stem"]["6bba_05db0fb1"]["tp"],
        "6bba_fp": res_exp1["per_stem"]["6bba_05db0fb1"]["fp"],
        "6bba_fn": res_exp1["per_stem"]["6bba_05db0fb1"]["fn"],
        "44b6_0113": res_exp1["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
        "44b6_0b24": res_exp1["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
        "6bba_05b6": res_exp1["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
        "6bba_05db": sc_6b,
    })

    # 3. COMBINATION 2: Exp 1 (th=0.968) + Exp 2 (smooth_density_cap_3.0x)
    smooth_gap_fn = make_smooth_density_gap_fn(scale_factor=3.0, dense_volume_only=True)
    res_exp3a = evaluate_custom_combined(
        label="exp3_th0968_plus_smooth_gap",
        val_stems=val_stems,
        val_raw_graphs=th968_val_raw_graphs,
        val_gt=val_gt,
        shared_heatmap_cache=shared_heatmap_cache,
        shared_frame_cache=shared_frame_cache,
        custom_gap_fn=smooth_gap_fn,
    )
    sc = res_exp3a["proxy_score"]
    sc_6b = res_exp3a["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
    print(f"[Combo 2: Exp1+Exp2] Overall: {sc:.5f} (d={sc-control_score:+.5f}) | 6bba: {sc_6b:.5f} (d={sc_6b-control_6bba:+.5f})")
    results.append({
        "configuration": "exp3_th0968_plus_smooth_gap",
        "overall_score": sc,
        "overall_delta": sc - control_score,
        "6bba_score": sc_6b,
        "6bba_delta": sc_6b - control_6bba,
        "overall_tp": res_exp3a["edge_tp"],
        "overall_fp": res_exp3a["edge_fp"],
        "overall_fn": res_exp3a["edge_fn"],
        "overall_precision": res_exp3a["edge_precision"],
        "overall_recall": res_exp3a["edge_recall"],
        "6bba_tp": res_exp3a["per_stem"]["6bba_05db0fb1"]["tp"],
        "6bba_fp": res_exp3a["per_stem"]["6bba_05db0fb1"]["fp"],
        "6bba_fn": res_exp3a["per_stem"]["6bba_05db0fb1"]["fn"],
        "44b6_0113": res_exp3a["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
        "44b6_0b24": res_exp3a["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
        "6bba_05b6": res_exp3a["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
        "6bba_05db": sc_6b,
    })

    # 4. COMBINATION 3: Exp 1 + Exp 2 + Motion Gating (stricter relaxed pass on dense volume)
    res_exp3b = evaluate_custom_combined(
        label="exp3_th0968_plus_smooth_gap_plus_motion_gate",
        val_stems=val_stems,
        val_raw_graphs=th968_val_raw_graphs,
        val_gt=val_gt,
        shared_heatmap_cache=shared_heatmap_cache,
        shared_frame_cache=shared_frame_cache,
        custom_gap_fn=smooth_gap_fn,
        dense_motion_tight_only=True,
    )
    sc = res_exp3b["proxy_score"]
    sc_6b = res_exp3b["per_stem"]["6bba_05db0fb1"]["adj_jaccard"]
    print(f"[Combo 3: Exp1+Exp2+MotionGate] Overall: {sc:.5f} (d={sc-control_score:+.5f}) | 6bba: {sc_6b:.5f} (d={sc_6b-control_6bba:+.5f})")
    results.append({
        "configuration": "exp3_th0968_plus_smooth_gap_plus_motion_gate",
        "overall_score": sc,
        "overall_delta": sc - control_score,
        "6bba_score": sc_6b,
        "6bba_delta": sc_6b - control_6bba,
        "overall_tp": res_exp3b["edge_tp"],
        "overall_fp": res_exp3b["edge_fp"],
        "overall_fn": res_exp3b["edge_fn"],
        "overall_precision": res_exp3b["edge_precision"],
        "overall_recall": res_exp3b["edge_recall"],
        "6bba_tp": res_exp3b["per_stem"]["6bba_05db0fb1"]["tp"],
        "6bba_fp": res_exp3b["per_stem"]["6bba_05db0fb1"]["fp"],
        "6bba_fn": res_exp3b["per_stem"]["6bba_05db0fb1"]["fn"],
        "44b6_0113": res_exp3b["per_stem"]["44b6_0113de3b"]["adj_jaccard"],
        "44b6_0b24": res_exp3b["per_stem"]["44b6_0b24845f"]["adj_jaccard"],
        "6bba_05b6": res_exp3b["per_stem"]["6bba_05b6850b"]["adj_jaccard"],
        "6bba_05db": sc_6b,
    })

    # Save to RESULTS.csv
    out_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "experiments" / "dense_volume" / "RESULTS.csv"
    out_df.to_csv(out_path, index=False)
    print("\nExperiment 3 complete! Results saved to:", out_path)


if __name__ == "__main__":
    main()
