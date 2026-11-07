









print("=" * 78)
print("PIPELINE MANIFEST -- resolved state, not just config")
print("=" * 78)

_secondary_weights_env = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "")
_secondary_ready = bool(_secondary_weights_env and Path(_secondary_weights_env).exists())
print(f"Dual-seed ensemble:      requested=True  weights_found={_secondary_ready}"
      f"{'  <-- FALLING BACK TO SINGLE-SEED, check BIOHUB_SECONDARY_WEIGHTS' if not _secondary_ready else ''}")

_bidir_weight = float(os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0"))
print(f"Bidirectional fusion:    weight={_bidir_weight}  "
      f"active={_bidir_weight > 0.0}  mode={os.environ.get('BIOHUB_BIDIRECTIONAL_FUSION_MODE', '(unset)')}")

_dc_requested = os.environ.get("BIOHUB_USE_DEEPCENTER_VETO", "0") != "0"
_dc_bundle = globals().get("DEEPCENTER_VETO_DETECTOR")
_dc_loaded = "DEEPCENTER_VETO_DETECTOR" in globals() and _dc_bundle is not None
_dc_path = _dc_bundle.get("path") if _dc_loaded else None
print(f"DeepCenter veto:         requested={_dc_requested}  loaded={_dc_loaded}"
      f"{'  <-- REQUESTED BUT NOT LOADED, gap/division vetoes are no-ops' if _dc_requested and not _dc_loaded else ''}")
if _dc_loaded:
    print(f"  - checkpoint file:     {_dc_path}")
    print(f"  - expected epoch:      {os.environ.get('BIOHUB_DEEPCENTER_EXPECTED_EPOCH')}")
print(f"  - gap veto:            {os.environ.get('BIOHUB_DEEPCENTER_GAP_VETO', '0') != '0'}")
print(f"  - safe-div veto:       {os.environ.get('BIOHUB_DEEPCENTER_SAFE_DIV_VETO', '0') != '0'}")

print(f"Safe-div thresholds:     parent<={os.environ.get('BIOHUB_SAFE_DIV_MAX_UM')}um  "
      f"sister<={os.environ.get('BIOHUB_SAFE_DIV_SISTER_MAX_UM')}um  "
      f"global_cap={os.environ.get('BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP')}  "
      f"frame_cap={os.environ.get('BIOHUB_SAFE_DIV_FRAME_FRAC_CAP')}")

print(f"Validator:               enabled={VALIDATOR_ENABLE}  "
      f"held_out_samples={len(val_stems)}  match_radius={VALIDATOR_MATCH_RADIUS_UM}um")

print("-" * 78)
print("v4 additions:")
print(f"  Leaf-prune:              threshold={LEAF_PRUNE_MIN_EDGE_PROB}  active={LEAF_PRUNE_MIN_EDGE_PROB > 0}")
print(f"  PPSWEEP extended:        enabled={PPSWEEP_EXTENDED}  candidates={len(PP_CANDIDATES)}")
print(f"  Prefix guard:            enabled={PPSWEEP_PREFIX_GUARD}")
if PP_SELECTED_PATH.exists():
    _sel = json.loads(PP_SELECTED_PATH.read_text())
    print(f"  Selected sweep config:   {_sel.get('selected')}  overrides={_sel.get('overrides')}")
print("=" * 78)
