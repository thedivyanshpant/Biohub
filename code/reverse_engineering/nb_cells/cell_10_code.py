


PP_CANDIDATES: dict[str, dict] = {
    "gap45": {"GAP_CLOSE_UM": 4.5},
    "tight55": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "relaxed9": {"MOTION_RELINK_RELAXED_UM": 9.0},
    "bonus125": {"MOTION_RELINK_LEARNED_BONUS": 1.25},
    "gap2step40": {"GAP2_MAX_STEP_UM": 4.0},
    "reuse28": {"GAP_CLOSE_REUSE_UM": 2.8},
    "dcgap035": {"DEEPCENTER_GAP_THRESHOLD": 0.35},
}

# ------------------------------------------------------------------ #
# v4: EXTENDED CANDIDATE SET (validated on the same 8 held-out videos;
# only selections passing margin + adj-loss + prefix guard ship).
# Division funnel at v3 reference run: ~2.6k divergence-rejected and
# ~89% of geometric survivors DeepCenter-rejected per video -> the
# dcsd*/divwide/sym075/diverge150 family attacks both kill-gates.
# ------------------------------------------------------------------ #
if PPSWEEP_EXTENDED:
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
PP_SELECT_MARGIN = float(os.environ.get("BIOHUB_PPSWEEP_SELECT_MARGIN", "0.002"))
PP_MAX_ADJ_LOSS = float(os.environ.get("BIOHUB_PPSWEEP_MAX_ADJ_LOSS", "0.0005"))
PP_SWEEP_RESULTS_PATH = WORKING_DIR / "ppsweep_results.csv"
PP_SELECTED_PATH = WORKING_DIR / "ppsweep_selected.json"

selected_label = "base"
selected_config: dict = {}

if VALIDATOR_ENABLE and val_stems and "base" in PP_RESULTS:
    base_summary = PP_RESULTS["base"]

    def _prefix_guard_ok(summary: dict) -> bool:
        """v4 anti-overfit guard: reject candidates whose proxy regresses by
        more than 0.001 on ANY embryo prefix (44b6 / 6bba), even if the pooled
        proxy improved. Keeps the sweep honest on 8 held-out videos."""
        if not PPSWEEP_PREFIX_GUARD:
            return True
        cand_pp = summary.get("prefix_proxy") or {}
        base_pp = base_summary.get("prefix_proxy") or {}
        for prefix, base_val in base_pp.items():
            if cand_pp.get(prefix, float("-inf")) < base_val - 0.001:
                return False
        return True
    print("=" * 78)
    print(f"POST-PROCESS SWEEP -- {len(PP_CANDIDATES)} candidates x {len(val_stems)} held-out videos")
    print("=" * 78)
    for label, config in PP_CANDIDATES.items():
        summary, rows = score_validator_config(config, label)
        PP_RESULTS[label] = summary
        validator_sample_rows.extend(rows)

    
    
    positive = [
        label for label, summary in PP_RESULTS.items()
        if label != "base"
        and summary["proxy_score"] >= base_summary["proxy_score"] + 0.0005
        and summary["adjusted_edge_jaccard"] >= base_summary["adjusted_edge_jaccard"] - PP_MAX_ADJ_LOSS
        and _prefix_guard_ok(summary)
    ]
    positive.sort(key=lambda l: PP_RESULTS[l]["proxy_score"], reverse=True)
    combo_config: dict = {}
    for label in positive:
        for key, value in PP_CANDIDATES[label].items():
            combo_config.setdefault(key, value)
    if len(positive) >= 2:
        combo_label = "combo(" + "+".join(positive) + ")"
        summary, rows = score_validator_config(combo_config, combo_label)
        PP_RESULTS[combo_label] = summary
        PP_CANDIDATES[combo_label] = combo_config
        validator_sample_rows.extend(rows)

    print()
    print("=" * 78)
    print("SWEEP TABLE (sorted by PROXY_SCORE)")
    print("=" * 78)
    ranked = sorted(PP_RESULTS.items(), key=lambda kv: kv[1]["proxy_score"], reverse=True)
    for label, summary in ranked:
        delta = summary["proxy_score"] - base_summary["proxy_score"]
        print(f"  {label:<40} proxy={summary['proxy_score']:.4f} ({delta:+.4f})  "
              f"adj={summary['adjusted_edge_jaccard']:.4f}  divJ={summary['division_jaccard']:.4f} "
              f"(tp/fp/fn={summary['div_tp']}/{summary['div_fp']}/{summary['div_fn']})")
    pd.DataFrame([
        {"config": label,
         **{k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
            for k, v in summary.items() if k != "config"},
         "overrides": json.dumps(PP_CANDIDATES.get(label, {}), sort_keys=True)}
        for label, summary in ranked
    ]).to_csv(PP_SWEEP_RESULTS_PATH, index=False)

    # v4: scan the ranking and take the FIRST candidate that passes all
    # three guards (proxy margin, adjusted-edge loss, prefix stability).
    # A higher-proxy candidate that fails a guard must never block a
    # valid runner-up from shipping.
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
    print()
    print(f"SELECTED: {selected_label}  overrides={selected_config}  "
          f"(margin rule: >= +{PP_SELECT_MARGIN} proxy and adj loss <= {PP_MAX_ADJ_LOSS})")
    print(f"  base     proxy={base_summary['proxy_score']:.4f} adj={base_summary['adjusted_edge_jaccard']:.4f} divJ={base_summary['division_jaccard']:.4f}")
    sel = PP_RESULTS[selected_label]
    print(f"  selected proxy={sel['proxy_score']:.4f} adj={sel['adjusted_edge_jaccard']:.4f} divJ={sel['division_jaccard']:.4f}")

    with VALIDATOR_STATS_PATH.open("w", newline="") as f:
        fieldnames = sorted({k for row in validator_sample_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in validator_sample_rows:
            writer.writerow(row)
else:
    print("SWEEP: validator unavailable -- keeping the base configuration.")

PP_SELECTED_PATH.write_text(json.dumps({
    "selected": selected_label,
    "overrides": selected_config,
    "base_proxy": PP_RESULTS.get("base", {}).get("proxy_score"),
    "selected_proxy": PP_RESULTS.get(selected_label, {}).get("proxy_score"),
    "held_out_stems": list(val_stems),
}, indent=2, sort_keys=True) + "\n")

if selected_config:
    print()
    print(f"Re-writing submission.csv with the selected post-process configuration: {selected_label}")
    _saved = pp_apply(selected_config)
    try:
        write_test_submission(selected_label)
    finally:
        pp_restore(_saved)
    for key, value in selected_config.items():
        os.environ["BIOHUB_" + key] = str(value)
else:
    print("Keeping the base submission.csv (public 0.939 configuration).")


_final = pd.read_csv(SUBMISSION_PATH)
assert _final.columns.tolist() == CSV_COLUMNS, _final.columns.tolist()
assert _final["id"].tolist() == list(range(len(_final))), "row ids not contiguous"
_expected_sets = sorted(p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith(".zarr"))
assert sorted(_final["dataset"].astype(str).unique()) == _expected_sets, "dataset mismatch"
for _ds, _grp in _final.groupby("dataset"):
    _n = _grp[_grp.row_type.eq("node")]
    _e = _grp[_grp.row_type.eq("edge")]
    _t = dict(zip(_n.node_id.astype(int), _n.t.astype(int)))
    assert all(_t[int(s)] + 1 == _t[int(d)] for s, d in zip(_e.source_id, _e.target_id)), f"{_ds}: bad edge time"
    assert _e.target_id.value_counts().max() <= 1, f"{_ds}: multi-parent"
    assert _e.source_id.value_counts().max() <= 2, f"{_ds}: out-degree > 2"
    print(f"  {_ds}: nodes={len(_n)} edges={len(_e)} divisions={(_e.source_id.value_counts() == 2).sum()}")
print(f"Final submission.csv rows={len(_final)}  config={selected_label}")
