"""
config.py — Faithful reproduction of Cell 0 + Cell 2

All environment variables and configuration parameters exactly as set
in biohub-geometric-fusion.ipynb.

Adapted for HPC paths instead of /kaggle/... paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ============================================================
# HPC path adaptation
# ============================================================
PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("BIOHUB_DATA_DIR", PROJECT_ROOT / "database"))
TEST_DIR = DATA_DIR / "test"
TRAIN_DIR = DATA_DIR / "train"
WORKING_DIR = Path(os.environ.get("BIOHUB_WORKING_DIR", PROJECT_ROOT / "code" / "leader_reproduction" / "working"))
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle artifact paths (downloaded via kaggle CLI)
KAGGLE_ARTIFACTS = Path(os.environ.get("BIOHUB_ARTIFACTS_DIR", PROJECT_ROOT / "kaggle_artifacts"))
PRIMARY_ARTIFACT_DIR = KAGGLE_ARTIFACTS / "biohub-tracking-support-pack-50ep-v1"
SECONDARY_ARTIFACT_DIR = KAGGLE_ARTIFACTS / "biohub-temporal-unet3d-seed314159-v1"
DEEPCENTER_ARTIFACT_DIR = KAGGLE_ARTIFACTS / "biohub-deepcenter-unet3d-center-prior-v1"

# Reproduction output
REPO_DIR = WORKING_DIR / "tracking_repo"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
RUN_STATS_PATH = WORKING_DIR / "run_stats.csv"

# ============================================================
# Cell 0: All environment variable settings (EXACT REPRODUCTION)
# ============================================================
BIOHUB_PRESET = "harmonic_v3_division_wide"
BIOHUB_SCORE_AXIS = "v4: 0.939 base + extended validator-gated sweep (div-recall + leaf-prune)"


def set_all_env_vars():
    """Set all environment variables exactly as Cell 0 does."""
    env = os.environ

    # --- Cell 0 env vars (exact) ---
    env["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
    env["BIOHUB_DET_THRESHOLD"] = "0.965"
    env["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.0"
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
    env["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"
    env["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "14.0"
    env["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.6"
    env["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "2.25"
    env["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"
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
    # DeepCenter checkpoint — adapted to HPC path
    env["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(
        DEEPCENTER_ARTIFACT_DIR / "weights" / "full_frame_center" / "best.pt"
    )
    env["BIOHUB_DEEPCENTER_GAP_VETO"] = "1"
    env["BIOHUB_DEEPCENTER_GAP_THRESHOLD"] = "0.25"
    env["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"
    env["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
    env["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"
    env["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = "harmonic_probability"
    env["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
    env["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"
    env["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"
    env["BIOHUB_PPSWEEP_SELECT_MARGIN"] = "0.001"
    env["BIOHUB_PPSWEEP_MAX_ADJ_LOSS"] = "0.0005"
    env["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"
    env["BIOHUB_DEEPCENTER_TTA"] = "1"
    env["BIOHUB_PPSWEEP_EXTENDED"] = "1"
    env["BIOHUB_PPSWEEP_PREFIX_GUARD"] = "1"
    env["BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB"] = "0.0"

    # --- Cell 3 secondary seed env vars ---
    env["BIOHUB_SECONDARY_WEIGHTS"] = str(
        SECONDARY_ARTIFACT_DIR / "weights" / "unet_transformer"
        / "split_0" / "edge_predictor_best.pth"
    )
    env["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.15"
    env["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.80"
    env["BIOHUB_SECONDARY_LINK_MODE"] = "low_margin_consensus"
    env["BIOHUB_SECONDARY_MIX_TEMPERATURE"] = "1"
    env["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = "0.35"
    env["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "0.48"

    # --- Cell 4 edge-TTA env vars ---
    env["BIOHUB_EDGE_FEATURE_TTA"] = "1"
    env["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"
    env["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "0.75"

    # --- HPC-specific artifact pointers ---
    env["BIOHUB_MODEL_ARTIFACTS"] = str(PRIMARY_ARTIFACT_DIR)
    env["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
    env["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
    env.setdefault("POLARS_PREFER_PKG", "32")

    # Override Kaggle paths that don't exist on HPC
    env["BIOHUB_PRIMARY_ARTIFACT_MANIFEST"] = str(
        PRIMARY_ARTIFACT_DIR / "ARTIFACT_MANIFEST.json"
    )
    env["BIOHUB_SECONDARY_ARTIFACT_MANIFEST"] = str(
        SECONDARY_ARTIFACT_DIR / "ARTIFACT_MANIFEST.json"
    )
    env["BIOHUB_DEEPCENTER_MANIFEST_DEFAULT"] = str(
        DEEPCENTER_ARTIFACT_DIR / "ARTIFACT_MANIFEST.json"
    )
    env["BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT"] = str(
        DEEPCENTER_ARTIFACT_DIR / "weights" / "full_frame_center" / "best.pt"
    )
    env["BIOHUB_TARGET_ARTIFACT_SLUG"] = "biohub-tracking-support-pack-50ep-v1"


# ============================================================
# Cell 2: Parameter parsing (exact reproduction)
# ============================================================
METHOD = "unet_transformer"
WEIGHTS_RELATIVE = f"weights/{METHOD}/split_0/edge_predictor_best.pth"
EXPERIMENT_TAG = "selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap"
COMPETITION = "biohub-cell-tracking-during-development"


def parse_config():
    """Parse all configuration from environment (Cell 2 logic)."""
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
        "output_division_geometry_filter": env.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0",
    }


# ============================================================
# Validation volumes: test volumes with known ground-truth
# ============================================================
VALIDATION_STEMS = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]

# SHA256 checksums for integrity verification (from notebook)
EXPECTED_SHA256 = {
    "primary": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    "secondary": "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
    "deepcenter": "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0",
    "manifest": "978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029",
}
