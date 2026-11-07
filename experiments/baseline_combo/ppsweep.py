"""
ppsweep.py — Complete faithful reproduction of Cell 9-10 PPSWEEP engine.

Executes the post-processing parameter sweep downstream of cached predictions:
- Exact parameter grid (24 candidates + base + combo = 26 configurations)
- Exact parameter names, types, and values
- Exact scoring function (official adjusted Jaccard + division Jaccard)
- Exact notebook selection criterion (margin rule + max adj loss + prefix guard)
- Produces PPSWEEP_RESULTS.csv and PPSWEEP_REPORT.md
"""
from __future__ import annotations

import copy as _copy
import csv
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td

# Ensure code root is in sys.path
PROJECT_CODE = Path(__file__).resolve().parent.parent
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from leader_reproduction import config, postprocess, evaluate

# Ensure config environment variables are active
config.set_all_env_vars()
_cfg = config.parse_config()

PROJECT_ROOT = config.PROJECT_ROOT
DATA_DIR = config.DATA_DIR
TEST_DIR = config.TEST_DIR
TRAIN_DIR = config.TRAIN_DIR
REPO_DIR = config.REPO_DIR
WORKING_DIR = config.WORKING_DIR

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

# Capture base configuration from postprocess module
PP_BASE_CONFIG = {key: getattr(postprocess, key) for key in PP_SWEEP_KEYS}

# Exact candidate grid from Cell 10
PP_CANDIDATES: dict[str, dict] = {
    "gap45": {"GAP_CLOSE_UM": 4.5},
    "tight55": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "relaxed9": {"MOTION_RELINK_RELAXED_UM": 9.0},
    "bonus125": {"MOTION_RELINK_LEARNED_BONUS": 1.25},
    "gap2step40": {"GAP2_MAX_STEP_UM": 4.0},
    "reuse28": {"GAP_CLOSE_REUSE_UM": 2.8},
    "dcgap035": {"DEEPCENTER_GAP_THRESHOLD": 0.35},
}

# Extended candidate set from Cell 10
PP_CANDIDATES.update({
    # -- division recall: loosen DeepCenter safe-div veto --
    "dcsd015": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.15},
    "dcsd010": {"DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10},
    # -- division recall: widen geometry windows --
    "divwide": {
        "SAFE_DIV_MAX_UM": 11.0,
        "SAFE_DIV_SISTER_MAX_UM": 16.0,
        "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0,
    },
    # -- division recall: relax sister-symmetry (higher tau = permissive) --
    "sym075": {"SAFE_DIV_SISTER_SYMMETRY_TAU": 0.75},
    # -- division recall: relax divergence requirement (top rejection source) --
    "diverge150": {"SAFE_DIV_DIVERGE_UM": 1.5},
    # -- the proven tight55 winner composed with each recall knob --
    "t55_dcsd015": {"MOTION_RELINK_TIGHT_UM": 5.5, "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.15},
    "t55_dcsd010": {"MOTION_RELINK_TIGHT_UM": 5.5, "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.10},
    "t55_divwide": {
        "MOTION_RELINK_TIGHT_UM": 5.5,
        "SAFE_DIV_MAX_UM": 11.0,
        "SAFE_DIV_SISTER_MAX_UM": 16.0,
        "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0,
    },
    "t55_diverge150": {"MOTION_RELINK_TIGHT_UM": 5.5, "SAFE_DIV_DIVERGE_UM": 1.5},
    # -- spurious-node suppression via v4 weak-leaf pruning --
    "leaf030": {"LEAF_PRUNE_MIN_EDGE_PROB": 0.30},
    "leaf040": {"LEAF_PRUNE_MIN_EDGE_PROB": 0.40},
    "t55_leaf030": {"MOTION_RELINK_TIGHT_UM": 5.5, "LEAF_PRUNE_MIN_EDGE_PROB": 0.30},
    # -- recall levers for under-predicting videos --
    "minlen5": {"OUTPUT_MIN_TRACK_LEN": 5},
    "rescue085": {"SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB": 0.85},
    "dcgap018": {"DEEPCENTER_GAP_THRESHOLD": 0.18},
    # -- motion-model variants --
    "vel075": {"MOTION_RELINK_VELOCITY_WEIGHT": 0.75},
    "vel025": {"MOTION_RELINK_VELOCITY_WEIGHT": 0.25},
})

PP_SELECT_MARGIN = float(os.environ.get("BIOHUB_PPSWEEP_SELECT_MARGIN", "0.001"))
PP_MAX_ADJ_LOSS = float(os.environ.get("BIOHUB_PPSWEEP_MAX_ADJ_LOSS", "0.0005"))
PPSWEEP_PREFIX_GUARD = os.environ.get("BIOHUB_PPSWEEP_PREFIX_GUARD", "1") != "0"
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


def load_raw_and_gt_graphs(stems: list[str]) -> tuple[dict, dict]:
    val_raw_graphs: dict[str, tuple[dict, list]] = {}
    val_gt: dict[str, tuple[dict, list, float]] = {}

    print(f"Loading {len(stems)} prediction and ground truth graphs into memory...")
    pred_dir = REPO_DIR / "predictions" / "deepanthpcu" / "unet_transformer" / "split_0"
    for stem in stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        if not gt_path.exists():
            raise RuntimeError(f"GT not found: {gt_path}")
        gt_graph = evaluate.graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = evaluate.graph_to_plain(gt_graph)
        t_true = evaluate.read_estimated_true_node_count(gt_path)
        if t_true is None:
            raise RuntimeError(f"estimated_number_of_nodes missing for {stem}")
        val_gt[stem] = (gt_nodes_plain, gt_edges_plain, t_true)

        pred_path = pred_dir / f"{stem}.geff"
        if not pred_path.exists():
            raise RuntimeError(f"Prediction graph not found: {pred_path}")
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

    print(f"✓ Cached {len(val_raw_graphs)} raw prediction graphs + ground truth in RAM.")
    return val_raw_graphs, val_gt


def score_validator_config(
    candidate_config: dict,
    label: str,
    val_stems: list[str],
    val_raw_graphs: dict,
    val_gt: dict,
    verbose: bool = False,
) -> tuple[dict, list]:
    saved = pp_apply(candidate_config)
    rows = []
    t0 = time.time()
    try:
        for stem in val_stems:
            raw_nodes_by_id, raw_edges = val_raw_graphs[stem]
            nodes_copy = _copy.deepcopy(raw_nodes_by_id)
            edges_copy = _copy.deepcopy(raw_edges)

            # Suppress verbose stage prints unless requested
            if not verbose:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    processed_nodes, processed_edges, _stage_stats = postprocess.filter_output_graph(
                        nodes_copy, edges_copy, dataset=stem,
                        deepcenter_bundle=postprocess.DEEPCENTER_VETO_DETECTOR,
                    )
            else:
                processed_nodes, processed_edges, _stage_stats = postprocess.filter_output_graph(
                    nodes_copy, edges_copy, dataset=stem,
                    deepcenter_bundle=postprocess.DEEPCENTER_VETO_DETECTOR,
                )

            gt_nodes_plain, gt_edges_plain, t_true = val_gt[stem]
            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
            pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]
            row = evaluate.score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
            row["stem"] = stem
            row["config"] = label
            row["safe_divisions_added"] = _stage_stats.get("safe_divisions_added", 0)
            rows.append(row)
    finally:
        pp_restore(saved)

    summary = evaluate.aggregate_official(rows)
    summary["n_samples"] = len(rows)
    summary["config"] = label
    summary["seconds"] = time.time() - t0

    # Total confusion numbers
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

    # Node matching numbers
    summary["node_fn"] = sum(r["missed_gt_nodes"] for r in rows)
    summary["node_fp"] = sum(r["spurious_pred_nodes"] for r in rows)
    summary["node_tp"] = sum(len(val_gt[r["stem"]][0]) - r["missed_gt_nodes"] for r in rows)

    # Prefix proxy for stability guard
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

    return summary, rows


def run_sweep(val_stems: list[str]) -> tuple[dict, str, dict]:
    val_raw_graphs, val_gt = load_raw_and_gt_graphs(val_stems)

    pp_results: dict[str, dict] = {}
    all_sample_rows: list[dict] = []

    print("\n" + "=" * 78)
    print("STEP 1: BASE EVALUATION (public 0.939 reference configuration)")
    print("=" * 78)
    base_summary, base_rows = score_validator_config({}, "base", val_stems, val_raw_graphs, val_gt, verbose=True)
    pp_results["base"] = base_summary
    all_sample_rows.extend(base_rows)

    print(
        f"[base] proxy_score={base_summary['proxy_score']:.5f}  "
        f"adj_edge={base_summary['adjusted_edge_jaccard']:.5f}  "
        f"edge_tp={base_summary['edge_tp']}  edge_fp={base_summary['edge_fp']}  "
        f"edge_fn={base_summary['edge_fn']}  recall={base_summary['edge_recall']*100:.2f}%  "
        f"[{base_summary['seconds']:.1f}s]"
    )

    def _prefix_guard_ok(cand_summary: dict) -> bool:
        if not PPSWEEP_PREFIX_GUARD:
            return True
        cand_pp = cand_summary.get("prefix_proxy") or {}
        base_pp = base_summary.get("prefix_proxy") or {}
        for prefix, base_val in base_pp.items():
            if cand_pp.get(prefix, float("-inf")) < base_val - 0.001:
                return False
        return True

    print("\n" + "=" * 78)
    print(f"STEP 2: RUNNING PPSWEEP ({len(PP_CANDIDATES)} candidates x {len(val_stems)} videos)")
    print("=" * 78)

    for idx, (label, candidate_config) in enumerate(PP_CANDIDATES.items(), 1):
        summary, rows = score_validator_config(candidate_config, label, val_stems, val_raw_graphs, val_gt)
        pp_results[label] = summary
        all_sample_rows.extend(rows)
        delta = summary["proxy_score"] - base_summary["proxy_score"]
        status = "PASSED" if _prefix_guard_ok(summary) else "GUARD_FAILED"
        print(
            f"  [{idx:02d}/{len(PP_CANDIDATES):02d}] {label:<20} proxy={summary['proxy_score']:.5f} "
            f"({delta:+.5f})  adj={summary['adjusted_edge_jaccard']:.5f}  "
            f"TP={summary['edge_tp']}  FP={summary['edge_fp']}  FN={summary['edge_fn']}  "
            f"rec={summary['edge_recall']*100:.2f}%  [{status}]  [{summary['seconds']:.1f}s]"
        )

    # Identify positive candidates that pass all guards
    positive = [
        label for label, summary in pp_results.items()
        if label != "base"
        and summary["proxy_score"] >= base_summary["proxy_score"] + 0.0005
        and summary["adjusted_edge_jaccard"] >= base_summary["adjusted_edge_jaccard"] - PP_MAX_ADJ_LOSS
        and _prefix_guard_ok(summary)
    ]
    positive.sort(key=lambda l: pp_results[l]["proxy_score"], reverse=True)
    print(f"\nPositive candidates passing guards ({len(positive)}): {positive}")

    combo_config: dict = {}
    combo_label = ""
    if len(positive) >= 2:
        for label in positive:
            for key, value in PP_CANDIDATES[label].items():
                combo_config.setdefault(key, value)
        combo_label = "combo(" + "+".join(positive) + ")"
        print(f"\nEvaluating composed configuration: {combo_label}")
        summary, rows = score_validator_config(combo_config, combo_label, val_stems, val_raw_graphs, val_gt)
        pp_results[combo_label] = summary
        PP_CANDIDATES[combo_label] = combo_config
        all_sample_rows.extend(rows)
        delta = summary["proxy_score"] - base_summary["proxy_score"]
        print(
            f"  [COMBO] {combo_label} proxy={summary['proxy_score']:.5f} ({delta:+.5f})  "
            f"adj={summary['adjusted_edge_jaccard']:.5f}  TP={summary['edge_tp']}  "
            f"FP={summary['edge_fp']}  FN={summary['edge_fn']}  [{summary['seconds']:.1f}s]"
        )

    # Rank all configurations by PROXY_SCORE
    ranked = sorted(pp_results.items(), key=lambda kv: kv[1]["proxy_score"], reverse=True)

    print("\n" + "=" * 78)
    print("RANKED SWEEP TABLE (sorted by PROXY_SCORE)")
    print("=" * 78)
    for label, summary in ranked:
        delta = summary["proxy_score"] - base_summary["proxy_score"]
        print(
            f"  {label:<40} proxy={summary['proxy_score']:.5f} ({delta:+.5f})  "
            f"adj={summary['adjusted_edge_jaccard']:.5f}  TP={summary['edge_tp']}  "
            f"FP={summary['edge_fp']}  FN={summary['edge_fn']}  rec={summary['edge_recall']*100:.2f}%"
        )

    # Notebook selection rule:
    # First candidate in ranked that satisfies margin, adj loss, and prefix guard
    selected_label = "base"
    selected_config: dict = {}
    for best_label, best_summary in ranked:
        if best_label == "base":
            break
        if (
            best_summary["proxy_score"] >= base_summary["proxy_score"] + PP_SELECT_MARGIN
            and best_summary["adjusted_edge_jaccard"] >= base_summary["adjusted_edge_jaccard"] - PP_MAX_ADJ_LOSS
            and _prefix_guard_ok(best_summary)
        ):
            selected_label = best_label
            selected_config = dict(PP_CANDIDATES[best_label])
            break

    print("\n" + "=" * 78)
    print("SELECTION VERIFICATION")
    print("=" * 78)
    print(f"SELECTED: {selected_label}")
    print(f"Overrides: {selected_config}")
    print(f"Selection criteria: margin >= +{PP_SELECT_MARGIN}, adj loss <= {PP_MAX_ADJ_LOSS}, prefix guard")
    print(f"Base score:     {base_summary['proxy_score']:.5f}")
    sel_summary = pp_results[selected_label]
    print(f"Selected score: {sel_summary['proxy_score']:.5f} (delta: {sel_summary['proxy_score'] - base_summary['proxy_score']:+.5f})")

    # Save results to DataFrame and CSV
    rows_for_csv = []
    for label, summary in ranked:
        delta = summary["proxy_score"] - base_summary["proxy_score"]
        row_dict = {
            "config": label,
            "overrides": json.dumps(PP_CANDIDATES.get(label, {}), sort_keys=True),
            "proxy_score": summary["proxy_score"],
            "delta_proxy": delta,
            "adjusted_edge_jaccard": summary["adjusted_edge_jaccard"],
            "raw_jaccard": summary["raw_jaccard"],
            "edge_tp": summary["edge_tp"],
            "edge_fp": summary["edge_fp"],
            "edge_fn": summary["edge_fn"],
            "edge_precision": summary["edge_precision"],
            "edge_recall": summary["edge_recall"],
            "node_tp": summary["node_tp"],
            "node_fp": summary["node_fp"],
            "node_fn": summary["node_fn"],
            "div_tp": summary["div_tp"],
            "div_fp": summary["div_fp"],
            "div_fn": summary["div_fn"],
            "div_jaccard": summary["division_jaccard"],
            "wrong_association_edges": summary["wrong_association_edges"],
            "edges_fragmented": summary["edges_fragmented"],
            "edges_lost_to_detection": summary["edges_lost_to_detection"],
            "runtime_seconds": summary["seconds"],
        }
        rows_for_csv.append(row_dict)

    df_results = pd.DataFrame(rows_for_csv)
    df_results.to_csv(WORKING_DIR / "ppsweep_results.csv", index=False)
    df_results.to_csv(PROJECT_CODE / "PPSWEEP_RESULTS.csv", index=False)
    print(f"Saved results to {WORKING_DIR / 'ppsweep_results.csv'} and {PROJECT_CODE / 'PPSWEEP_RESULTS.csv'}")

    # Also save selected JSON
    (WORKING_DIR / "ppsweep_selected.json").write_text(
        json.dumps({
            "selected": selected_label,
            "overrides": selected_config,
            "base_proxy": base_summary["proxy_score"],
            "selected_proxy": sel_summary["proxy_score"],
            "held_out_stems": list(val_stems),
        }, indent=2, sort_keys=True) + "\n"
    )

    return pp_results, selected_label, selected_config


if __name__ == "__main__":
    val_stems = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
    run_sweep(val_stems)
