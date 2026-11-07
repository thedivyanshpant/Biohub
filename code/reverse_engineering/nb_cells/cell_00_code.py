import os
BIOHUB_PRESET = 'harmonic_v3_division_wide'
BIOHUB_SCORE_AXIS = 'v4: 0.939 base + extended validator-gated sweep (div-recall + leaf-prune)'

os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = '1.0'
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "2"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.0"
os.environ["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"
os.environ["BIOHUB_GAP_DENSITY_REFERENCE_UM"] = "6.5"
os.environ["BIOHUB_GAP_DENSITY_GAIN"] = "0.040"
os.environ["BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM"] = "0.125"
os.environ["BIOHUB_GAP_DENSITY_NEIGHBORS"] = "3"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"  


os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "14.0"  



os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.6"  
os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "2.25"  


os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"

os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.2"     
os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "4"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.88"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.0"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.012"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "120"
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "2"
os.environ["BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM"] = "8.5"
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_GAP_THRESHOLD"] = "0.25"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"
os.environ["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = "harmonic_probability"
os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"
os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"
os.environ["BIOHUB_PPSWEEP_SELECT_MARGIN"] = "0.001"
os.environ["BIOHUB_PPSWEEP_MAX_ADJ_LOSS"] = "0.0005"

os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"
os.environ["BIOHUB_DEEPCENTER_TTA"] = "1"

# --- v4 additions: PPSWEEP extension + weak-leaf pruning (validator-gated) ---
# Extended sweep candidates + per-embryo-prefix stability guard.
os.environ["BIOHUB_PPSWEEP_EXTENDED"] = "1"
os.environ["BIOHUB_PPSWEEP_PREFIX_GUARD"] = "1"
# Weak-leaf pruning threshold: 0.0 = inert (PPSWEEP sweeps 0.30/0.40).
os.environ["BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB"] = "0.0"
print("BIOHUB_PRESET:", BIOHUB_PRESET)
print("BIOHUB_SCORE_AXIS:", BIOHUB_SCORE_AXIS)