"""
evaluate.py — Faithful reproduction of Cell 8 evaluation metrics.

Evaluates predictions on the 4 overlapping validation volumes:
  - 44b6_0113de3b
  - 44b6_0b24845f
  - 6bba_05b6850b
  - 6bba_05db0fb1

Computes:
  - Node bipartite matching (max_dist=7.0 um)
  - Edge TP, FP, FN, Jaccard
  - Node penalty and Adjusted Edge Jaccard
  - Division TP, FP, FN, Division Jaccard
  - Full error decomposition (detection loss, fragmentation, wrong association)
  - Official weighted aggregate score:
      Score = adjusted_edge_jaccard + 0.1 * division_jaccard
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tracksdata as td
from scipy.optimize import linear_sum_assignment

# Ensure code root is in sys.path
PROJECT_CODE = Path(__file__).resolve().parent.parent
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from leader_reproduction import config

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
VALIDATOR_MATCH_RADIUS_UM = 7.0
VALIDATOR_NODE_COUNT_PENALTY_A = 0.1
VALIDATOR_DIVISION_WEIGHT = 0.1


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def match_nodes_bipartite(pred_nodes: dict, gt_nodes: dict, max_dist: float = 7.0):
    pred_by_t: dict[int, list[int]] = {}
    for pid, (t, *_r) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (t, *_r) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)

    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for t, p_ids in pred_by_t.items():
        g_ids = gt_by_t.get(t, [])
        if not g_ids:
            continue
        voxel_scale = np.array(VOXEL_SCALE_UM, dtype=float)
        p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype=float) * voxel_scale
        g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype=float) * voxel_scale
        diff = p_pos[:, None, :] - g_pos[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=-1))
        BIG = 1e6
        cost_gated = np.where(cost <= max_dist, cost, BIG)
        row_ind, col_ind = linear_sum_assignment(cost_gated)
        for r, c in zip(row_ind, col_ind):
            if cost_gated[r, c] >= BIG:
                continue
            pred_to_gt[p_ids[r]] = g_ids[c]
            gt_to_pred[g_ids[c]] = p_ids[r]
    return pred_to_gt, gt_to_pred


def compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = {}
    gt_incoming_source: dict[int, int] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming_source[t] = s

    tp = 0
    fp = 0
    matched_gt_edges = set()
    for s, t in pred_edges:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, ())
        if is_tp:
            tp += 1
            matched_gt_edges.add((ms, mt))
            continue
        is_fp = (mt is not None and mt in gt_incoming_source) or (
            ms is not None and bool(gt_outgoing.get(ms))
        )
        if is_fp:
            fp += 1
    fn = len(gt_edge_set - matched_gt_edges)
    return tp, fp, fn


def edge_jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def adjusted_jaccard(jaccard: float, t_pred: int, t_true, a: float = 0.1) -> float:
    if not t_true or t_true <= 0:
        return jaccard
    return max(0.0, jaccard * (1.0 - a * (t_pred - t_true) / t_true))


def weakly_connected_components(node_ids, edges):
    parent = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, t in edges:
        if s in parent and t in parent:
            union(s, t)
    return {n: find(n) for n in node_ids}


def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):
    gt_out: dict[int, set[int]] = {}
    gt_in: dict[int, int] = {}
    for s, t in gt_edges:
        gt_out.setdefault(s, set()).add(t)
        gt_in[t] = s

    pred_out: dict[int, set[int]] = {}
    for s, t in pred_edges:
        pred_out.setdefault(s, set()).add(t)

    pred_node_ids = list(pred_nodes.keys())
    pred_edge_list = list(pred_edges)
    components = weakly_connected_components(pred_node_ids, pred_edge_list)
    fork_components = {
        components[n] for n, outs in pred_out.items() if len(outs) >= 2 and n in components
    }
    gt_division_sources = [s for s, outs in gt_out.items() if len(outs) >= 2]

    def lineage_descendants(root_child: int) -> set[int]:
        seen = {root_child}
        stack = [root_child]
        while stack:
            cur = stack.pop()
            for nxt in gt_out.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    tp = 0
    fn = 0
    tp_gt_sources: set[int] = set()

    for gsrc in gt_division_sources:
        children = sorted(gt_out[gsrc])
        if len(children) < 2:
            continue
        anchor_candidates = [gsrc]
        if gsrc in gt_in:
            anchor_candidates.append(gt_in[gsrc])
        anchor_pred_nodes = [gt_to_pred[a] for a in anchor_candidates if a in gt_to_pred]

        lineage_hit_components: list[set[int]] = []
        ok = True
        for child in children[:2]:
            lineage = lineage_descendants(child)
            hit_comp_ids = {
                components[p_id]
                for gt_id in lineage
                if (p_id := gt_to_pred.get(gt_id)) is not None and p_id in components
            }
            if not hit_comp_ids:
                ok = False
                break
            lineage_hit_components.append(hit_comp_ids)

        if not ok or not anchor_pred_nodes:
            fn += 1
            continue

        anchor_comp_ids = {components[p] for p in anchor_pred_nodes if p in components}
        if not anchor_comp_ids:
            fn += 1
            continue

        found = any(
            comp_id in lineage_hit_components[0]
            and comp_id in lineage_hit_components[1]
            and comp_id in fork_components
            for comp_id in anchor_comp_ids
        )
        if found:
            tp += 1
            tp_gt_sources.add(gsrc)
        else:
            fn += 1

    fp = 0
    for n, outs in pred_out.items():
        if len(outs) < 2:
            continue
        g = pred_to_gt.get(n)
        if g is None or g not in gt_out or g in tp_gt_sources:
            continue
        fp += 1

    return tp, fp, fn


def decompose_errors(pred_nodes, gt_nodes, pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    gt_edge_set = set(gt_edges)
    pred_edge_set = set(pred_edges)
    gt_outgoing: dict[int, set[int]] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)

    missed_gt_nodes = sum(1 for g in gt_nodes if g not in gt_to_pred)
    spurious_pred_nodes = sum(1 for p in pred_nodes if p not in pred_to_gt)

    recovered = fragmented = lost_to_detection = 0
    for gs, gtid in gt_edge_set:
        ps, pt = gt_to_pred.get(gs), gt_to_pred.get(gtid)
        if ps is None or pt is None:
            lost_to_detection += 1
        elif (ps, pt) in pred_edge_set:
            recovered += 1
        else:
            fragmented += 1

    wrong_association = 0
    for ps, pt in pred_edge_set:
        ms, mt = pred_to_gt.get(ps), pred_to_gt.get(pt)
        if ms is not None and mt is not None and mt not in gt_outgoing.get(ms, ()):
            wrong_association += 1

    return {
        "missed_gt_nodes": missed_gt_nodes,
        "spurious_pred_nodes": spurious_pred_nodes,
        "edges_recovered": recovered,
        "edges_fragmented": fragmented,
        "edges_lost_to_detection": lost_to_detection,
        "wrong_association_edges": wrong_association,
    }


def _find_key_recursive(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def read_estimated_true_node_count(geff_path: Path):
    for candidate in (geff_path / "zarr.json", geff_path / ".zattrs"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        found = _find_key_recursive(payload, "estimated_number_of_nodes")
        if found is not None:
            try:
                return float(found)
            except (TypeError, ValueError):
                continue
    return None


def graph_to_plain(graph):
    nodes: dict[int, tuple] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = (int(row["t"]), float(row["z"]), float(row["y"]), float(row["x"]))
    edges: list[tuple[int, int]] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edges.append((int(row["source_id"]), int(row["target_id"])))
    return nodes, edges


def score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true):
    p2g, g2p = match_nodes_bipartite(pred_nodes_plain, gt_nodes_plain, max_dist=VALIDATOR_MATCH_RADIUS_UM)
    tp, fp, fn = compute_edge_confusion(pred_edges_plain, gt_edges_plain, p2g, g2p)
    jac = edge_jaccard(tp, fp, fn)
    t_pred = len(pred_nodes_plain)
    adj = adjusted_jaccard(jac, t_pred, t_true, a=VALIDATOR_NODE_COUNT_PENALTY_A)
    div_tp, div_fp, div_fn = compute_division_confusion(
        pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, p2g, g2p
    )
    errors = decompose_errors(pred_nodes_plain, gt_nodes_plain, pred_edges_plain, gt_edges_plain, p2g, g2p)
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    row = {
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn, "edge_jaccard": jac,
        "t_pred": t_pred, "t_true": t_true, "adjusted_edge_jaccard": adj,
        "div_tp": div_tp, "div_fp": div_fp, "div_fn": div_fn, "div_jaccard": div_jac,
        "weight": tp + fp + fn,
    }
    row.update(errors)
    return row


def aggregate_official(sample_rows):
    total_w = sum(r["weight"] for r in sample_rows) or 1
    weighted_adj = sum(r["adjusted_edge_jaccard"] * r["weight"] for r in sample_rows) / total_w
    div_tp = sum(r["div_tp"] for r in sample_rows)
    div_fp = sum(r["div_fp"] for r in sample_rows)
    div_fn = sum(r["div_fn"] for r in sample_rows)
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    return {
        "adjusted_edge_jaccard": weighted_adj,
        "division_jaccard": div_jac,
        "proxy_score": weighted_adj + VALIDATOR_DIVISION_WEIGHT * div_jac,
        "div_tp": div_tp, "div_fp": div_fp, "div_fn": div_fn,
        "missed_gt_nodes": sum(r["missed_gt_nodes"] for r in sample_rows),
        "spurious_pred_nodes": sum(r["spurious_pred_nodes"] for r in sample_rows),
        "edges_recovered": sum(r["edges_recovered"] for r in sample_rows),
        "edges_fragmented": sum(r["edges_fragmented"] for r in sample_rows),
        "edges_lost_to_detection": sum(r["edges_lost_to_detection"] for r in sample_rows),
        "wrong_association_edges": sum(r["wrong_association_edges"] for r in sample_rows),
    }


def parse_submission_graph(submission_path: Path, dataset: str):
    """Extract plain nodes and edges for a specific dataset from submission.csv."""
    nodes = {}
    edges = []
    with submission_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] != dataset:
                continue
            row_type = row["row_type"]
            if row_type == "node":
                nid = int(row["node_id"])
                t = int(row["t"])
                z = float(row["z"])
                y = float(row["y"])
                x = float(row["x"])
                nodes[nid] = (t, z, y, x)
            elif row_type == "edge":
                s = int(row["source_id"])
                target = int(row["target_id"])
                edges.append((s, target))
    return nodes, edges


def evaluate_volumes(
    train_dir: Path = config.TRAIN_DIR,
    submission_path: Path = config.SUBMISSION_PATH,
    working_dir: Path = config.WORKING_DIR,
    stems: list[str] | None = None,
):
    if stems is None:
        stems = config.VALIDATION_STEMS

    print("=" * 70)
    print("EVALUATION ON VALIDATION VOLUMES")
    print("=" * 70)
    print(f"Validation stems ({len(stems)}): {stems}")
    print(f"Submission: {submission_path}")
    print(f"Ground truth directory: {train_dir}")

    sample_rows = []
    details = {}

    for stem in stems:
        gt_path = train_dir / f"{stem}.geff"
        if not gt_path.exists():
            print(f"WARNING: GT path {gt_path} does not exist, skipping")
            continue

        gt_graph = graph_from_geff(gt_path)
        gt_nodes, gt_edges = graph_to_plain(gt_graph)
        t_true_est = read_estimated_true_node_count(gt_path)
        t_true = t_true_est if t_true_est is not None else len(gt_nodes)

        # Parse predictions from submission
        if not submission_path.exists():
            raise FileNotFoundError(f"Submission file not found: {submission_path}")

        pred_nodes, pred_edges = parse_submission_graph(submission_path, stem)
        if not pred_nodes:
            print(f"WARNING: No prediction nodes for {stem} in submission!")
            continue

        row = score_sample(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true)
        row["dataset"] = stem
        sample_rows.append(row)
        details[stem] = row

        score = row["adjusted_edge_jaccard"] + VALIDATOR_DIVISION_WEIGHT * row["div_jaccard"]
        print(f"\n--- [{stem}] ---")
        print(f"  Pred nodes: {len(pred_nodes):,} | GT nodes: {t_true:,}")
        print(f"  Pred edges: {len(pred_edges):,} | GT edges: {len(gt_edges):,}")
        print(f"  Edge TP: {row['edge_tp']:,} | FP: {row['edge_fp']:,} | FN: {row['edge_fn']:,}")
        print(f"  Edge Jaccard: {row['edge_jaccard']:.5f} | Adj Jaccard: {row['adjusted_edge_jaccard']:.5f}")
        print(f"  Div TP: {row['div_tp']} | FP: {row['div_fp']} | FN: {row['div_fn']} | Div Jaccard: {row['div_jaccard']:.5f}")
        print(f"  Volume Score: {score:.5f}")
        print(f"  Errors: Missed GT={row['missed_gt_nodes']} | Spurious={row['spurious_pred_nodes']} | Lost to det={row['edges_lost_to_detection']} | Fragmented={row['edges_fragmented']} | Wrong assoc={row['wrong_association_edges']}")

    if not sample_rows:
        print("ERROR: No volumes were evaluated!")
        return None

    official = aggregate_official(sample_rows)
    print("\n" + "=" * 70)
    print("AGGREGATE OFFICIAL SCORE SUMMARY")
    print("=" * 70)
    print(f"  Weighted Adj Edge Jaccard: {official['adjusted_edge_jaccard']:.5f}")
    print(f"  Division Jaccard:          {official['division_jaccard']:.5f}")
    print(f"  FINAL PROXY SCORE:         {official['proxy_score']:.5f}")
    print(f"  Division TP/FP/FN:         {official['div_tp']} / {official['div_fp']} / {official['div_fn']}")
    print(f"  Total Missed GT Nodes:     {official['missed_gt_nodes']:,}")
    print(f"  Total Spurious Nodes:      {official['spurious_pred_nodes']:,}")
    print(f"  Total Fragmented Edges:    {official['edges_fragmented']:,}")
    print(f"  Total Wrong Assoc Edges:   {official['wrong_association_edges']:,}")
    print("=" * 70)

    # Save results to json
    results_path = working_dir / "evaluation_results.json"
    output_data = {
        "official_aggregate": official,
        "per_volume": details,
    }
    results_path.write_text(json.dumps(output_data, indent=2))
    print(f"Saved results to {results_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Biohub reproduction on validation volumes")
    parser.add_argument("--train-dir", type=Path, default=config.TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=config.TEST_DIR)
    parser.add_argument("--repo-dir", type=Path, default=config.REPO_DIR)
    parser.add_argument("--working-dir", type=Path, default=config.WORKING_DIR)
    parser.add_argument("--submission-path", type=Path, default=config.SUBMISSION_PATH)
    args = parser.parse_args()

    evaluate_volumes(
        train_dir=args.train_dir,
        submission_path=args.submission_path,
        working_dir=args.working_dir,
    )


if __name__ == "__main__":
    main()
