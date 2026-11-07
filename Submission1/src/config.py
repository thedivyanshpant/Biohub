import os
from pathlib import Path

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/input")
    DATA_DIR = PROJECT_ROOT / "biohub-cell-tracking-during-development"
    TEST_DIR = DATA_DIR / "test"
    WORKING_DIR = Path("/kaggle/working")
    PRIMARY_ARTIFACT_DIR = PROJECT_ROOT / "biohub-tracking-support-pack-50ep-v1"
    SECONDARY_ARTIFACT_DIR = PROJECT_ROOT / "biohub-temporal-unet3d-seed314159-v1"
    DEEPCENTER_ARTIFACT_DIR = PROJECT_ROOT / "biohub-deepcenter-unet3d-center-prior-v1"
else:
    PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", Path(__file__).resolve().parents[2]))
    DATA_DIR = Path(os.environ.get("BIOHUB_DATA_DIR", PROJECT_ROOT / "database"))
    TEST_DIR = DATA_DIR / "test"
    WORKING_DIR = Path(os.environ.get("BIOHUB_WORKING_DIR", PROJECT_ROOT / "Submission1" / "working"))
    KAGGLE_ARTIFACTS = Path(os.environ.get("BIOHUB_ARTIFACTS_DIR", PROJECT_ROOT / "kaggle_artifacts"))
    PRIMARY_ARTIFACT_DIR = KAGGLE_ARTIFACTS / "biohub-tracking-support-pack-50ep-v1"
    SECONDARY_ARTIFACT_DIR = KAGGLE_ARTIFACTS / "biohub-temporal-unet3d-seed314159-v1"
    DEEPCENTER_ARTIFACT_DIR = KAGGLE_ARTIFACTS / "biohub-deepcenter-unet3d-center-prior-v1"

WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = WORKING_DIR / "submission.csv"

def set_all_env_vars():
    """Set all environment variables (incorporating base + ppsweep overrides)."""
    env = os.environ
    
    # Base configuration from biohub-geometric-fusion.ipynb
    env["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
    env["BIOHUB_DET_THRESHOLD"] = "0.965"
    env["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
    env["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "2"
    env["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
    env["BIOHUB_GAP_CLOSE_UM"] = "5.0"
    env["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"
    env["BIOHUB_GAP_DENSITY_REFERENCE_UM"] = "6.5"
    env["BIOHUB_GAP_DENSITY_GAIN"] = "0.040"
    env["BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM"] = "0.125"
    env["BIOHUB_GAP_DENSITY_NEIGHBORS"] = "3"
    env["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
    env["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"
    env["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
    env["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.6"
    env["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
    env["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"
    env["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.2"
    env["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"
    env["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "4"
    env["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.88"
    env["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.0"
    env["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.012"
    env["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "120"
    env["BIOHUB_USE_DEEPCENTER_VETO"] = "1"
    env["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "1"
    env["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "2"
    env["BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM"] = "8.5"
    env["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(DEEPCENTER_ARTIFACT_DIR / "weights" / "full_frame_center" / "best.pt")
    env["BIOHUB_DEEPCENTER_GAP_VETO"] = "1"
    env["BIOHUB_DEEPCENTER_GAP_THRESHOLD"] = "0.25"
    env["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"
    env["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
    env["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"
    env["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = "harmonic_probability"
    env["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
    env["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"
    env["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"
    env["BIOHUB_DEEPCENTER_TTA"] = "1"
    
    # Secondary model configuration
    env["BIOHUB_SECONDARY_WEIGHTS"] = str(SECONDARY_ARTIFACT_DIR / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth")
    env["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.15"
    env["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.80"
    env["BIOHUB_SECONDARY_LINK_MODE"] = "low_margin_consensus"
    env["BIOHUB_SECONDARY_MIX_TEMPERATURE"] = "1"
    env["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = "0.35"
    env["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "0.48"
    
    # TTA configuration
    env["BIOHUB_EDGE_FEATURE_TTA"] = "1"
    env["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"
    env["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"
    
    # Path setup
    env["BIOHUB_MODEL_ARTIFACTS"] = str(PRIMARY_ARTIFACT_DIR)
    
    # --- PPSWEEP Selected Overrides (Score: 0.90017) ---
    env["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "0.25"
    env["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.25"
    env["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.10"
    env["BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB"] = "0.30"
    env["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"
    env["BIOHUB_SAFE_DIV_MAX_UM"] = "11.0"
    env["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "16.0"
    env["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "12.0"
    env["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "1.5"
    env["BIOHUB_MOTION_RELINK_RELAXED_UM"] = "9.0"

def parse_config():
    """Parse all configuration from environment."""
    env = os.environ
    return {
        "det_threshold": float(env.get("BIOHUB_DET_THRESHOLD", "0.99")),
        "unet_batch_size": int(env.get("BIOHUB_UNET_BATCH_SIZE", "4")),
        "use_ilp": env.get("BIOHUB_USE_ILP", "1") != "0",
        "ilp_edge_weight": float(env.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0")),
        "ilp_appearance_weight": float(env.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.1")),
        "ilp_disappearance_weight": float(env.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "0.1")),
        "ilp_division_weight": float(env.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.0")),
        "output_edge_max_um": float(env.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0")),
        "output_enforce_next_frame": env.get("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", "1") != "0",
        "output_single_parent_repair": env.get("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", "1") != "0",
        "output_single_child_repair": env.get("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", "0") != "0",
        "output_prune_isolated": env.get("BIOHUB_OUTPUT_PRUNE_ISOLATED", "1") != "0",
        "output_motion_relink": env.get("BIOHUB_OUTPUT_MOTION_RELINK", "1") != "0",
        "motion_relink_tight_um": float(env.get("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0")),
        "motion_relink_relaxed_um": float(env.get("BIOHUB_MOTION_RELINK_RELAXED_UM", "10.0")),
        "motion_relink_velocity_weight": float(env.get("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "0.5")),
        "motion_relink_learned_bonus": float(env.get("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "0.75")),
        "motion_relink_max_frame_nodes": int(env.get("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", "2600")),
        "output_division_geometry_filter": env.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0",
        "div_parent_max_um": float(env.get("BIOHUB_DIV_PARENT_MAX_UM", "10.5")),
        "div_sister_max_um": float(env.get("BIOHUB_DIV_SISTER_MAX_UM", "8.0")),
        "div_drop_to_single_if_bad": env.get("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "1") != "0",
        "output_gap_close": env.get("BIOHUB_OUTPUT_GAP_CLOSE", "1") != "0",
        "gap_close_max_gap": int(env.get("BIOHUB_GAP_CLOSE_MAX_GAP", "1")),
        "gap_close_um": float(env.get("BIOHUB_GAP_CLOSE_UM", "6.0")),
        "gap_density_adaptive": env.get("BIOHUB_GAP_DENSITY_ADAPTIVE", "0") != "0",
        "gap_density_reference_um": float(env.get("BIOHUB_GAP_DENSITY_REFERENCE_UM", "6.5")),
        "gap_density_gain": float(env.get("BIOHUB_GAP_DENSITY_GAIN", "0.040")),
        "gap_density_max_step_delta_um": float(env.get("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", "0.125")),
        "gap_density_neighbors": int(env.get("BIOHUB_GAP_DENSITY_NEIGHBORS", "3")),
        "gap_close_reuse_existing": env.get("BIOHUB_GAP_CLOSE_REUSE_EXISTING", "1") != "0",
        "gap_close_reuse_um": float(env.get("BIOHUB_GAP_CLOSE_REUSE_UM", "3.2")),
        "gap_close_max_added_frac": float(env.get("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", "0.05")),
        "gap_close_max_added_abs": int(env.get("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", "2000")),
        "gap_refine_synthetic": env.get("BIOHUB_GAP_REFINE_SYNTHETIC", "1") != "0",
        "gap_refine_win_z": int(env.get("BIOHUB_GAP_REFINE_WIN_Z", "1")),
        "gap_refine_win_yx": int(env.get("BIOHUB_GAP_REFINE_WIN_YX", "3")),
        "gap_refine_max_shift_um": float(env.get("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", "3.2")),
        "output_filter_short_tracks": env.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1") != "0",
        "output_min_track_len": int(env.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "6")),
        "output_keep_division_components": env.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0",
        "adaptive_short_track_rescue": env.get("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", "0") != "0",
        "short_track_rescue_trigger_removed_frac": float(env.get("BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC", "0.10")),
        "short_track_rescue_min_len": int(env.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN", "4")),
        "short_track_rescue_min_mean_edge_prob": float(env.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "0.82")),
        "short_track_rescue_max_mean_edge_dist_um": float(env.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM", "3.25")),
        "short_track_rescue_max_nodes_frac": float(env.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC", "0.018")),
        "short_track_rescue_max_nodes_abs": int(env.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS", "180")),
        "output_linefit_smooth": env.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1") != "0",
        "output_linefit_weight": float(env.get("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "0.8")),
        "output_linefit_window": int(env.get("BIOHUB_OUTPUT_LINEFIT_WINDOW", "2")),
        "leaf_prune_min_edge_prob": float(env.get("BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB", "0.0")),
        "output_gap2_recovery": env.get("BIOHUB_OUTPUT_GAP2_RECOVERY", "0") != "0",
        "gap2_max_total_um": float(env.get("BIOHUB_GAP2_MAX_TOTAL_UM", "10.2")),
        "gap2_max_step_um": float(env.get("BIOHUB_GAP2_MAX_STEP_UM", "4.4")),
        "gap2_max_links_frac": float(env.get("BIOHUB_GAP2_MAX_LINKS_FRAC", "0.0045")),
        "gap2_max_links_abs": int(env.get("BIOHUB_GAP2_MAX_LINKS_ABS", "180")),
        "gap2_require_context": env.get("BIOHUB_GAP2_REQUIRE_CONTEXT", "1") != "0",
        "gap2_frame_frac_cap": float(env.get("BIOHUB_GAP2_FRAME_FRAC_CAP", "0.006")),
        "output_safe_divisions": env.get("BIOHUB_OUTPUT_SAFE_DIVISIONS", "1") != "0",
        "safe_div_max_um": float(env.get("BIOHUB_SAFE_DIV_MAX_UM", "4.7")),
        "safe_div_sister_max_um": float(env.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "7.2")),
        "safe_div_sister_symmetry_tau": float(env.get("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "0.0")),
        "safe_div_existing_child_max_um": float(env.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8")),
        "safe_div_frame_frac_cap": float(env.get("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "0.008")),
        "safe_div_global_frac_cap": float(env.get("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "0.004")),
        "safe_div_diverge_um": float(env.get("BIOHUB_SAFE_DIV_DIVERGE_UM", "2.25")),
        "safe_div_require_divergence": env.get("BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE", "1") != "0",
        "safe_div_require_mutual_nn": env.get("BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN", "1") != "0",
        "use_deepcenter_veto": env.get("BIOHUB_USE_DEEPCENTER_VETO", "1") != "0",
        "require_deepcenter_veto": env.get("BIOHUB_REQUIRE_DEEPCENTER_VETO", "1") != "0",
        "deepcenter_gap_veto": env.get("BIOHUB_DEEPCENTER_GAP_VETO", "1") != "0",
        "deepcenter_safe_div_veto": env.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0",
        "deepcenter_gap_threshold": float(env.get("BIOHUB_DEEPCENTER_GAP_THRESHOLD", "0.10")),
        "deepcenter_expected_epoch": int(env.get("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", "0")),
        "deepcenter_gap_confirm_min_span_um": float(env.get("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", "0")),
        "deepcenter_safe_div_threshold": float(env.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.12")),
    }
