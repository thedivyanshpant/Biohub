"""
Official Competition Metric Evaluator for Biohub Cell Tracking Challenge.

Reference Specification:
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
and tracking_cellmot/metrics.py, division_metrics.py
"""

from typing import NamedTuple, Optional, Tuple, Dict, Set, List
import numpy as np
import scipy.spatial.distance as sp_dist
from scipy.optimize import linear_sum_assignment


class EvaluationResult(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    num_gt_nodes: int
    num_pred_edges: int
    num_gt_edges: int
    edge_jaccard: float
    total_node_ratio: float
    adj_edge_jaccard: float
    division_jaccard: float
    score: float


class DatasetsResult(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    micro_edge_jaccard: float
    adj_edge_jaccard: float
    division_jaccard: float
    score: float
    per_dataset_results: Dict[str, EvaluationResult]


ADJUSTMENT_ALPHA: float = 0.1
SCORE_DIVISION_WEIGHT: float = 0.1
DEFAULT_SCALE: Tuple[float, float, float] = (1.625, 0.40625, 0.40625)
DEFAULT_MATCH_DISTANCE: float = 7.0


def match_nodes_bipartite(
    pred_nodes_t: np.ndarray,  # shape (N, 3) with z, y, x
    pred_ids_t: np.ndarray,    # shape (N,)
    gt_nodes_t: np.ndarray,    # shape (M, 3) with z, y, x
    gt_ids_t: np.ndarray,      # shape (M,)
    scale: Tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = DEFAULT_MATCH_DISTANCE,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Match predicted and ground-truth nodes at a single timepoint using optimal bipartite assignment.
    Weights are w = 1 / (1 + distance), max_distance threshold = 7.0 um.
    """
    if len(pred_nodes_t) == 0 or len(gt_nodes_t) == 0:
        return {}, {}

    scale_arr = np.array(scale, dtype=np.float64)
    pred_scaled = pred_nodes_t * scale_arr
    gt_scaled = gt_nodes_t * scale_arr

    # distance matrix: shape (N, M)
    dist_mat = sp_dist.cdist(pred_scaled, gt_scaled)

    # Cost matrix for linear_sum_assignment (minimize cost -> maximize weight)
    # Valid matches have cost = -1 / (1 + d)
    fill_val = 1e9
    cost_mat = np.full(dist_mat.shape, fill_val, dtype=np.float64)
    valid_mask = dist_mat <= max_distance
    cost_mat[valid_mask] = -1.0 / (1.0 + dist_mat[valid_mask])

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    pred_to_gt = {}
    gt_to_pred = {}
    for r, c in zip(row_ind, col_ind):
        if dist_mat[r, c] <= max_distance:
            p_id = int(pred_ids_t[r])
            g_id = int(gt_ids_t[c])
            pred_to_gt[p_id] = g_id
            gt_to_pred[g_id] = p_id

    return pred_to_gt, gt_to_pred


def evaluate_dataset(
    pred_nodes: Dict[int, Tuple[int, int, int, int]],  # id -> (t, z, y, x)
    pred_edges: List[Tuple[int, int]],                 # list of (source_id, target_id)
    gt_nodes: Dict[int, Tuple[int, int, int, int]],    # id -> (t, z, y, x)
    gt_edges: List[Tuple[int, int]],                   # list of (source_id, target_id)
    n_total: Optional[float] = None,                   # estimated_number_of_nodes
    scale: Tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = DEFAULT_MATCH_DISTANCE,
) -> EvaluationResult:
    """
    Evaluates a predicted tracking graph against ground truth according to the official specification.
    """
    # 1. Frame-by-frame node matching
    pred_by_t = {}
    for pid, (t, z, y, x) in pred_nodes.items():
        pred_by_t.setdefault(t, []).append((pid, z, y, x))

    gt_by_t = {}
    for gid, (t, z, y, x) in gt_nodes.items():
        gt_by_t.setdefault(t, []).append((gid, z, y, x))

    pred_to_gt = {}
    gt_to_pred = {}

    all_t = set(pred_by_t.keys()) | set(gt_by_t.keys())
    for t in all_t:
        p_list = pred_by_t.get(t, [])
        g_list = gt_by_t.get(t, [])
        if not p_list or not g_list:
            continue
        p_ids = np.array([p[0] for p in p_list], dtype=np.int64)
        p_coords = np.array([[p[1], p[2], p[3]] for p in p_list], dtype=np.float64)

        g_ids = np.array([g[0] for g in g_list], dtype=np.int64)
        g_coords = np.array([[g[1], g[2], g[3]] for g in g_list], dtype=np.float64)

        p2g, g2p = match_nodes_bipartite(p_coords, p_ids, g_coords, g_ids, scale=scale, max_distance=max_distance)
        pred_to_gt.update(p2g)
        gt_to_pred.update(g2p)

    # 2. Filter predicted edges
    # - Connect consecutive frames: t_target == t_source + 1
    # - Unique edges (keep lowest edge index)
    # - Cap out-degree: max 2 edges per source
    filtered_edges = []
    seen_edges = set()
    out_counts = {}

    for src, tgt in pred_edges:
        if src not in pred_nodes or tgt not in pred_nodes:
            continue
        t_src = pred_nodes[src][0]
        t_tgt = pred_nodes[tgt][0]
        if t_tgt - t_src != 1:
            continue
        edge_key = (src, tgt)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        out_counts[src] = out_counts.get(src, 0) + 1
        if out_counts[src] <= 2:
            filtered_edges.append((src, tgt))

    # 3. GT edge set & node degrees
    gt_edge_set = set(gt_edges)
    gt_out_degrees = {}
    gt_in_degrees = {}
    for src, tgt in gt_edges:
        gt_out_degrees[src] = gt_out_degrees.get(src, 0) + 1
        gt_in_degrees[tgt] = gt_in_degrees.get(tgt, 0) + 1

    # 4. Collapse merges: when multiple predicted edges map onto the same GT edge
    matched_gt_edges_seen = set()
    dedup_filtered_edges = []
    for src, tgt in filtered_edges:
        src_gt = pred_to_gt.get(src)
        tgt_gt = pred_to_gt.get(tgt)
        if src_gt is not None and tgt_gt is not None:
            gt_pair = (src_gt, tgt_gt)
            if gt_pair in gt_edge_set:
                if gt_pair in matched_gt_edges_seen:
                    continue  # drop merge duplicate
                matched_gt_edges_seen.add(gt_pair)
        dedup_filtered_edges.append((src, tgt))

    # 5. Evaluate Edge TP, FP, FN
    edge_tp = 0
    valid_pred_count = 0

    for src, tgt in dedup_filtered_edges:
        src_gt = pred_to_gt.get(src)
        tgt_gt = pred_to_gt.get(tgt)

        is_tp = False
        if src_gt is not None and tgt_gt is not None:
            if (src_gt, tgt_gt) in gt_edge_set:
                is_tp = True

        if is_tp:
            edge_tp += 1

        # Validity check: source matches a GT node with out_degree > 0 OR target matches GT node with in_degree > 0
        src_valid = (src_gt is not None) and (gt_out_degrees.get(src_gt, 0) > 0)
        tgt_valid = (tgt_gt is not None) and (gt_in_degrees.get(tgt_gt, 0) > 0)
        if src_valid or tgt_valid:
            valid_pred_count += 1

    num_gt_edges = len(gt_edges)
    edge_fp = valid_pred_count - edge_tp
    edge_fn = num_gt_edges - edge_tp

    edge_denom = edge_tp + edge_fp + edge_fn
    edge_jaccard = (edge_tp / edge_denom) if edge_denom > 0 else 0.0

    # 6. Adjusted Edge Jaccard
    num_pred_nodes = len(pred_nodes)
    if n_total is not None and n_total > 0:
        total_node_ratio = (num_pred_nodes - n_total) / n_total
        adj_edge_jaccard = max(0.0, edge_jaccard * (1.0 - ADJUSTMENT_ALPHA * total_node_ratio))
    else:
        total_node_ratio = 0.0
        adj_edge_jaccard = edge_jaccard

    # 7. Division Evaluation
    gt_div_nodes = [node for node, deg in gt_out_degrees.items() if deg >= 2]
    pred_out_degrees = {}
    for src, tgt in dedup_filtered_edges:
        pred_out_degrees[src] = pred_out_degrees.get(src, 0) + 1
    pred_div_nodes = [node for node, deg in pred_out_degrees.items() if deg >= 2]

    # Division TP/FP/FN calculation
    if len(pred_div_nodes) == 0:
        div_tp = 0
        div_fp = 0
        div_fn = len(gt_div_nodes)
    else:
        # Full bipartite division matching
        div_tp, div_fp, div_fn = _evaluate_divisions_full(
            pred_nodes, dedup_filtered_edges, pred_div_nodes,
            gt_nodes, gt_edges, gt_div_nodes,
            pred_to_gt, gt_to_pred
        )

    div_denom = div_tp + div_fp + div_fn
    has_divisions = div_denom > 0
    division_jaccard = (div_tp / div_denom) if has_divisions else float("nan")

    if has_divisions and not np.isnan(division_jaccard):
        score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
    else:
        score = adj_edge_jaccard

    return EvaluationResult(
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=div_tp,
        division_fp=div_fp,
        division_fn=div_fn,
        num_pred_nodes=num_pred_nodes,
        num_gt_nodes=len(gt_nodes),
        num_pred_edges=len(dedup_filtered_edges),
        num_gt_edges=num_gt_edges,
        edge_jaccard=edge_jaccard,
        total_node_ratio=total_node_ratio,
        adj_edge_jaccard=adj_edge_jaccard,
        division_jaccard=division_jaccard,
        score=score,
    )


def _evaluate_divisions_full(
    pred_nodes, pred_edges, pred_div_nodes,
    gt_nodes, gt_edges, gt_div_nodes,
    pred_to_gt, gt_to_pred
) -> Tuple[int, int, int]:
    """Evaluates divisions when predicted forks exist."""
    # Build predecessor and successor lookups
    pred_succ = {}
    pred_pred = {}
    for s, t in pred_edges:
        pred_succ.setdefault(s, []).append(t)
        pred_pred.setdefault(t, []).append(s)

    gt_succ = {}
    gt_pred = {}
    for s, t in gt_edges:
        gt_succ.setdefault(s, []).append(t)
        gt_pred.setdefault(t, []).append(s)

    # For each GT division, check if any candidate predicted fork recovers it
    paired_gt = set()
    paired_pred = set()

    for gt_div in gt_div_nodes:
        gt_children = gt_succ.get(gt_div, [])
        if len(gt_children) < 2:
            continue
        c1, c2 = gt_children[0], gt_children[1]
        c1_tree = {c1} | set(gt_succ.get(c1, []))
        c2_tree = {c2} | set(gt_succ.get(c2, []))
        parent_window = {gt_div} | set(gt_pred.get(gt_div, []))

        # Look for a pred fork whose parent anchor matches parent_window and daughters match c1_tree, c2_tree
        for p_div in pred_div_nodes:
            if p_div in paired_pred:
                continue
            p_anchor = {p_div} | set(pred_pred.get(p_div, []))
            p_gt_matches = {pred_to_gt.get(p) for p in p_anchor if p in pred_to_gt}
            if p_gt_matches.isdisjoint(parent_window):
                continue

            p_children = pred_succ.get(p_div, [])
            if len(p_children) < 2:
                continue

            # Branch 1 and Branch 2 in prediction
            b1 = {p_children[0]} | set(pred_succ.get(p_children[0], []))
            b2 = {p_children[1]} | set(pred_succ.get(p_children[1], []))

            b1_gt = {pred_to_gt.get(n) for n in b1 if n in pred_to_gt}
            b2_gt = {pred_to_gt.get(n) for n in b2 if n in pred_to_gt}

            # Check if (b1 matches c1_tree and b2 matches c2_tree) or vice versa
            match_straight = (not b1_gt.isdisjoint(c1_tree)) and (not b2_gt.isdisjoint(c2_tree))
            match_cross = (not b1_gt.isdisjoint(c2_tree)) and (not b2_gt.isdisjoint(c1_tree))

            if match_straight or match_cross:
                paired_gt.add(gt_div)
                paired_pred.add(p_div)
                break

    tp = len(paired_gt)
    fn = len(gt_div_nodes) - tp
    # FP are predicted forks that were evaluable but not paired
    fp = len(pred_div_nodes) - len(paired_pred)
    return tp, fp, fn


def evaluate_run(
    results_by_dataset: Dict[str, EvaluationResult]
) -> DatasetsResult:
    """
    Aggregates per-dataset evaluation results via micro-averaging as specified by the competition.
    """
    tot_edge_tp = sum(r.edge_tp for r in results_by_dataset.values())
    tot_edge_fp = sum(r.edge_fp for r in results_by_dataset.values())
    tot_edge_fn = sum(r.edge_fn for r in results_by_dataset.values())

    tot_div_tp = sum(r.division_tp for r in results_by_dataset.values())
    tot_div_fp = sum(r.division_fp for r in results_by_dataset.values())
    tot_div_fn = sum(r.division_fn for r in results_by_dataset.values())

    edge_denom = tot_edge_tp + tot_edge_fp + tot_edge_fn
    micro_edge_jaccard = (tot_edge_tp / edge_denom) if edge_denom > 0 else 0.0

    # Adjusted edge jaccard is sample-size weighted average of per-sample adj_edge_jaccard
    weights = [r.edge_tp + r.edge_fp + r.edge_fn for r in results_by_dataset.values()]
    total_w = sum(weights)
    if total_w > 0:
        weighted_adj_edge_jaccard = sum(w * r.adj_edge_jaccard for w, r in zip(weights, results_by_dataset.values())) / total_w
    else:
        weighted_adj_edge_jaccard = micro_edge_jaccard

    div_denom = tot_div_tp + tot_div_fp + tot_div_fn
    has_divisions = div_denom > 0
    division_jaccard = (tot_div_tp / div_denom) if has_divisions else float("nan")

    if has_divisions and not np.isnan(division_jaccard):
        final_score = weighted_adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
    else:
        final_score = weighted_adj_edge_jaccard

    return DatasetsResult(
        edge_tp=tot_edge_tp,
        edge_fp=tot_edge_fp,
        edge_fn=tot_edge_fn,
        division_tp=tot_div_tp,
        division_fp=tot_div_fp,
        division_fn=tot_div_fn,
        micro_edge_jaccard=micro_edge_jaccard,
        adj_edge_jaccard=weighted_adj_edge_jaccard,
        division_jaccard=division_jaccard,
        score=final_score,
        per_dataset_results=results_by_dataset,
    )
