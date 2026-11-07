import copy as _copy





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
PP_BASE_CONFIG = {key: globals()[key] for key in PP_SWEEP_KEYS}
print("Post-process base configuration (public 0.939):")
for key in PP_SWEEP_KEYS:
    print(f"  {key:<40} {PP_BASE_CONFIG[key]}")


def pp_apply(config: dict) -> dict:
    saved = {key: globals()[key] for key in config}
    for key, value in config.items():
        if key not in PP_SWEEP_KEYS:
            raise KeyError(f"{key} is not a sweepable post-process constant")
        globals()[key] = type(PP_BASE_CONFIG[key])(value)
    return saved


def pp_restore(saved: dict) -> None:
    for key, value in saved.items():
        globals()[key] = value


VAL_RAW_GRAPHS: dict[str, tuple[dict, list]] = {}
VAL_BASE_PROCESSED: dict[str, tuple] = {}  # v4: base-processed graphs for the audit
VAL_GT: dict[str, tuple[list, list, object]] = {}

if VALIDATOR_ENABLE and val_stems:
    val_pred_paths = {
        stem: found
        for stem in val_stems
        if (found := next((REPO_DIR / "predictions").rglob(f"{stem}.geff"), None)) is not None
    }
    missing = [s for s in val_stems if s not in val_pred_paths]
    if missing:
        raise RuntimeError(f"VALIDATOR: no prediction .geff for {missing}")
    for stem in val_stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        if not gt_path.exists():
            raise RuntimeError(f"VALIDATOR: missing GT {gt_path}")
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)
        if t_true is None:
            raise RuntimeError(f"VALIDATOR: estimated_number_of_nodes missing for {stem}")
        VAL_GT[stem] = (gt_nodes_plain, gt_edges_plain, t_true)

        pred_graph = graph_from_geff(val_pred_paths[stem])
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            raw_nodes_by_id[node_id] = {
                "node_id": node_id, "t": int(row["t"]),
                "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
            }
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })
        VAL_RAW_GRAPHS[stem] = (raw_nodes_by_id, raw_edges)
    print(f"VALIDATOR: cached {len(VAL_RAW_GRAPHS)} raw prediction graphs + GT")


def score_validator_config(config: dict, label: str, verbose: bool = False) -> tuple[dict, list]:
    """Run the full post-process on every cached validator graph with the
    given overrides and score it with the official metric formula."""
    saved = pp_apply(config)
    _real_test_dir = TEST_DIR
    globals()["TEST_DIR"] = TRAIN_DIR
    rows = []
    t0 = time.time()
    try:
        for stem in val_stems:
            raw_nodes_by_id, raw_edges = VAL_RAW_GRAPHS[stem]
            nodes_copy = _copy.deepcopy(raw_nodes_by_id)
            edges_copy = _copy.deepcopy(raw_edges)
            processed_nodes, processed_edges, _stage_stats = filter_output_graph(
                nodes_copy, edges_copy, dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
            gt_nodes_plain, gt_edges_plain, t_true = VAL_GT[stem]
            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
            pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]
            row = score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
            row["stem"] = stem
            row["config"] = label
            row["safe_divisions_added"] = _stage_stats.get("safe_divisions_added", 0)
            if label == "base":
                VAL_BASE_PROCESSED[stem] = (pred_nodes_plain, pred_edges_plain)
            rows.append(row)
    finally:
        globals()["TEST_DIR"] = _real_test_dir
        pp_restore(saved)
    summary = aggregate_official(rows)
    summary["n_samples"] = len(rows)
    summary["config"] = label
    summary["seconds"] = time.time() - t0

    # v4: per-embryo-prefix proxy for the anti-overfit selection guard
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
            p_adj + VALIDATOR_DIVISION_WEIGHT * edge_jaccard(p_tp, p_fp, p_fn)
        )
    if verbose:
        for row in rows:
            print(f"  {row['stem']:<28} edge_jaccard={row['edge_jaccard']:.4f} "
                  f"adj={row['adjusted_edge_jaccard']:.4f} T_pred={row['t_pred']} T_true={row['t_true']} "
                  f"div(tp/fp/fn)=({row['div_tp']}/{row['div_fp']}/{row['div_fn']}) "
                  f"safe_div_added={row['safe_divisions_added']}")
    print(f"[{label}] n={summary['n_samples']} adjusted_edge_jaccard={summary['adjusted_edge_jaccard']:.4f} "
          f"division_jaccard={summary['division_jaccard']:.4f} (tp/fp/fn={summary['div_tp']}/{summary['div_fp']}/{summary['div_fn']}) "
          f"PROXY_SCORE={summary['proxy_score']:.4f}  [{summary['seconds']/60:.1f} min]")
    return summary, rows


def division_recoverability_audit() -> None:
    """v4 diagnostic (never affects scores or selection).

    Classifies every GT division whose matched prediction parent does not emit
    two child edges:
      * recoverable_by_postprocess -- BOTH GT daughters have a matched pred node,
        so safe-div / relink sweep candidates can potentially convert FN -> TP;
      * missing_second_daughter   -- the second daughter node is absent entirely
        (detection-limited; needs inference-level changes, not post-processing);
      * parent_unmatched          -- the GT parent itself has no pred match.
    """
    if not VAL_BASE_PROCESSED:
        return
    totals = {
        "reproduced_ok": 0,
        "recoverable_by_postprocess": 0,
        "missing_second_daughter": 0,
        "parent_unmatched": 0,
    }
    per_stem_lines: list[str] = []
    for stem in val_stems:
        base = VAL_BASE_PROCESSED.get(stem)
        if base is None:
            continue
        pred_nodes_plain, pred_edges_plain = base
        gt_nodes_plain, gt_edges_plain, _t_true = VAL_GT[stem]
        _p2g, g2p = match_nodes_bipartite(
            pred_nodes_plain, gt_nodes_plain, max_dist=VALIDATOR_MATCH_RADIUS_UM
        )
        gt_children: dict[int, list[int]] = {}
        for s, t in gt_edges_plain:
            gt_children.setdefault(s, []).append(t)
        pred_children_count: dict[int, int] = {}
        for s, t in pred_edges_plain:
            pred_children_count[s] = pred_children_count.get(s, 0) + 1
        stem_counts = {
            "recoverable_by_postprocess": 0,
            "missing_second_daughter": 0,
            "parent_unmatched": 0,
        }
        for parent, kids in gt_children.items():
            if len(kids) < 2:
                continue
            pred_parent = g2p.get(parent)
            if pred_parent is None:
                stem_counts["parent_unmatched"] += 1
            elif pred_children_count.get(pred_parent, 0) >= 2:
                totals["reproduced_ok"] += 1
                continue
            else:
                matched_kids = sum(1 for k in kids if k in g2p)
                if matched_kids >= len(kids):
                    stem_counts["recoverable_by_postprocess"] += 1
                else:
                    stem_counts["missing_second_daughter"] += 1
        for key, value in stem_counts.items():
            totals[key] += value
        if any(stem_counts.values()):
            rendered = "  ".join(f"{k}={v}" for k, v in stem_counts.items())
            per_stem_lines.append(f"  {stem:<28} {rendered}")
    print()
    print("=" * 78)
    print("DIVISION-FN RECOVERABILITY AUDIT (base config, diagnostic only)")
    print("=" * 78)
    for line in per_stem_lines:
        print(line)
    print(
        f"TOTAL: reproduced_ok={totals['reproduced_ok']}  "
        f"recoverable_by_postprocess={totals['recoverable_by_postprocess']}  "
        f"missing_second_daughter={totals['missing_second_daughter']}  "
        f"parent_unmatched={totals['parent_unmatched']}"
    )
    print("recoverable -> try dcsd010/dcsd015/divwide/sym075/diverge150 family")
    print("missing     -> detection-limited; post-processing cannot recover these")


validator_sample_rows: list[dict[str, object]] = []
validator_summary_rows: list[dict[str, object]] = []
PP_RESULTS: dict[str, dict] = {}

if VALIDATOR_ENABLE and val_stems:
    print()
    print("=" * 78)
    print("LOCAL VALIDATOR -- base configuration (public 0.939), official metric formula")
    print("(https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)")
    print("=" * 78)
    base_summary, base_rows = score_validator_config({}, "base", verbose=True)
    PP_RESULTS["base"] = base_summary
    validator_sample_rows.extend(base_rows)
    validator_summary_rows.append(base_summary)
    with VALIDATOR_STATS_PATH.open("w", newline="") as f:
        fieldnames = sorted({k for row in base_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            writer.writerow(row)
    print(f"Per-sample validator rows written to {VALIDATOR_STATS_PATH}")
    division_recoverability_audit()
else:
    print("VALIDATOR: disabled or no held-out samples available -- skipping scoring.")
