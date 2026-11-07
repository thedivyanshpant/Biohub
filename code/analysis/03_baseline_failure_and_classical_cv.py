#!/usr/bin/env python3
"""
Biohub Comprehensive Discovery Pass: Part 3
Baseline Failure Analysis (EXP00 Diagnosis) & Classical Computer Vision Feasibility Benchmarking.
"""

import os
import sys
import time
import json
import math
import zarr
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.spatial.distance as sp_dist
from scipy.optimize import linear_sum_assignment
from skimage.feature import blob_log, blob_dog

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # (Z, Y, X) in microns
MATCH_DIST = 7.0  # microns

DATABASE_DIR = os.environ.get("BIOHUB_DATABASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database")))
TRAIN_DIR = os.path.join(DATABASE_DIR, "train")
TEST_DIR = os.path.join(DATABASE_DIR, "test")
OUT_DIR = os.environ.get("BIOHUB_CODE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
EXP00_SUB = os.environ.get("BIOHUB_EXP00_SUB", os.path.join(OUT_DIR, "experiments/exp00_baseline/submission_exp00.csv"))

# 4 Validation Datasets
VAL_DATASETS = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1"
]


def load_gt_for_dataset(name):
    geff_path = os.path.join(TRAIN_DIR, f"{name}.geff")
    root = zarr.open_group(geff_path, mode="r")
    with open(os.path.join(geff_path, "zarr.json")) as f:
        meta = json.load(f)
    n_total = meta.get("attributes", {}).get("geff", {}).get("extra", {}).get("estimated_number_of_nodes")

    node_ids = root["nodes/ids"][:]
    t_vals = root["nodes/props/t/values"][:]
    z_vals = root["nodes/props/z/values"][:]
    y_vals = root["nodes/props/y/values"][:]
    x_vals = root["nodes/props/x/values"][:]

    gt_nodes = {}
    for nid, t, z, y, x in zip(node_ids, t_vals, z_vals, y_vals, x_vals):
        gt_nodes[int(nid)] = (int(t), int(z), int(y), int(x))
    return gt_nodes, n_total


def evaluate_detection_predictions(pred_nodes_dict, gt_nodes_dict, scale=SCALE, max_dist=MATCH_DIST):
    """
    Evaluates centroid precision, recall, F1 at given physical match distance.
    Returns tp, fp, fn, precision, recall, f1, localization_errors, merged_count, split_count.
    """
    # Group by timepoint
    pred_by_t = {}
    for pid, (t, z, y, x) in pred_nodes_dict.items():
        pred_by_t.setdefault(t, []).append((pid, z, y, x))

    gt_by_t = {}
    for gid, (t, z, y, x) in gt_nodes_dict.items():
        gt_by_t.setdefault(t, []).append((gid, z, y, x))

    total_tp = 0
    total_loc_errors = []
    matched_gt = set()
    matched_pred = set()

    all_t = set(pred_by_t.keys()) | set(gt_by_t.keys())

    for t in all_t:
        plist = pred_by_t.get(t, [])
        glist = gt_by_t.get(t, [])
        if not plist or not glist:
            continue

        p_ids = [p[0] for p in plist]
        p_phys = np.array([[p[1], p[2], p[3]] for p in plist]) * scale

        g_ids = [g[0] for g in glist]
        g_phys = np.array([[g[1], g[2], g[3]] for g in glist]) * scale

        dmat = sp_dist.cdist(p_phys, g_phys)
        cost_mat = np.full(dmat.shape, 1e9)
        valid = dmat <= max_dist
        cost_mat[valid] = -1.0 / (1.0 + dmat[valid])

        r_ind, c_ind = linear_sum_assignment(cost_mat)
        for r, c in zip(r_ind, c_ind):
            if dmat[r, c] <= max_dist:
                total_tp += 1
                matched_pred.add(p_ids[r])
                matched_gt.add(g_ids[c])
                total_loc_errors.append(dmat[r, c])

    total_pred = len(pred_nodes_dict)
    total_gt = len(gt_nodes_dict)
    total_fp = total_pred - total_tp
    total_fn = total_gt - total_tp

    precision = (total_tp / total_pred) if total_pred > 0 else 0.0
    recall = (total_tp / total_gt) if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_pred": total_pred,
        "num_gt": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loc_error_median": float(np.median(total_loc_errors)) if total_loc_errors else 0.0,
        "loc_error_mean": float(np.mean(total_loc_errors)) if total_loc_errors else 0.0,
        "loc_error_p90": float(np.percentile(total_loc_errors, 90)) if total_loc_errors else 0.0,
    }


# ==============================================================================
# CLASSICAL DETECTION CANDIDATES (Implemented on 3D frames)
# ==============================================================================

def detect_baseline_ds4(vol, percentile=90):
    """EXP00: 4x downsample + uniform filter(3) + 90th percentile + CC."""
    ds = vol[::4, ::4, ::4]
    filt = ndi.uniform_filter(ds, size=3)
    thresh = np.percentile(filt, percentile)
    mask = filt > thresh
    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return []
    centroids = ndi.center_of_mass(mask, labeled, range(1, num_features + 1))
    return [(c[0] * 4, c[1] * 4, c[2] * 4) for c in centroids]


def detect_downsample_ds2(vol, percentile=95):
    """2x downsample + uniform filter(3) + 95th percentile + CC."""
    ds = vol[::2, ::2, ::2]
    filt = ndi.uniform_filter(ds, size=3)
    thresh = np.percentile(filt, percentile)
    mask = filt > thresh
    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return []
    centroids = ndi.center_of_mass(mask, labeled, range(1, num_features + 1))
    return [(c[0] * 2, c[1] * 2, c[2] * 2) for c in centroids]


def detect_fullres_threshold(vol, percentile=97):
    """Full-resolution uniform filter(3) + 97th percentile threshold + CC."""
    filt = ndi.uniform_filter(vol.astype(np.float32), size=3)
    thresh = np.percentile(filt, percentile)
    mask = filt > thresh
    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return []
    centroids = ndi.center_of_mass(mask, labeled, range(1, num_features + 1))
    return [(c[0], c[1], c[2]) for c in centroids]


def detect_local_maxima_3d(vol, min_distance_vox=(2, 5, 5), threshold_pct=90):
    """3D Local Maxima Peak Finding (non-maximum suppression)."""
    filt = ndi.gaussian_filter(vol.astype(np.float32), sigma=(1.0, 1.5, 1.5))
    thresh = np.percentile(filt, threshold_pct)
    # 3D maximum filter
    footprint = np.ones((2 * min_distance_vox[0] + 1, 2 * min_distance_vox[1] + 1, 2 * min_distance_vox[2] + 1), dtype=bool)
    max_filt = ndi.maximum_filter(filt, footprint=footprint)
    peaks = (filt == max_filt) & (filt > thresh)
    coords = np.argwhere(peaks)
    return [(c[0], c[1], c[2]) for c in coords]


def detect_dog_blobs(vol, threshold_pct=92):
    """3D Difference of Gaussians (DoG) blob detector (subsampled 2x for speed)."""
    ds = vol[::2, ::2, ::2].astype(np.float32)
    thresh = np.percentile(ds, threshold_pct)
    g1 = ndi.gaussian_filter(ds, sigma=(1.0, 1.5, 1.5))
    g2 = ndi.gaussian_filter(ds, sigma=(2.0, 3.0, 3.0))
    dog = g1 - g2
    dog_thresh = np.percentile(dog, 85)
    # find local maxima of dog
    peaks = (dog == ndi.maximum_filter(dog, size=(3, 5, 5))) & (dog > dog_thresh) & (ds > thresh)
    coords = np.argwhere(peaks)
    return [(c[0] * 2, c[1] * 2, c[2] * 2) for c in coords]


def detect_watershed_splitting(vol, percentile=93):
    """Distance Transform + Watershed splitting on thresholded objects."""
    ds = vol[::2, ::2, ::2]
    filt = ndi.uniform_filter(ds, size=3)
    thresh = np.percentile(filt, percentile)
    mask = filt > thresh
    if not np.any(mask):
        return []
    dist = ndi.distance_transform_edt(mask)
    local_max = (dist == ndi.maximum_filter(dist, size=(3, 5, 5))) & (dist > 1.5)
    markers, num_markers = ndi.label(local_max)
    if num_markers == 0:
        markers, num_markers = ndi.label(mask)
    dist_inv = dist.max() - dist
    dist_inv_u16 = ((dist_inv / (dist_inv.max() + 1e-6)) * 65535).astype(np.uint16)
    labels = ndi.watershed_ift(dist_inv_u16, markers.astype(np.int32))
    centroids = ndi.center_of_mass(mask, labels, range(1, num_markers + 1))
    valid_c = [c for c in centroids if not (math.isnan(c[0]) or math.isnan(c[1]) or math.isnan(c[2]))]
    return [(c[0] * 2, c[1] * 2, c[2] * 2) for c in valid_c]


def main():
    t0 = time.time()
    print("================================================================================")
    print("BIOHUB DISCOVERY PASS — PART 3: BASELINE FAILURE & CLASSICAL CV FEASIBILITY")
    print("================================================================================")

    # 1. Baseline Failure Analysis on EXP00
    print("\n--- 1. Evaluating EXP00 Predictions on Public Validation Sets ---")
    sub_df = pd.read_csv(EXP00_SUB)
    node_sub = sub_df[sub_df["row_type"] == "node"]

    exp00_results = {}
    for name in VAL_DATASETS:
        gt_nodes, n_total = load_gt_for_dataset(name)
        pred_nodes = {}
        ds_rows = node_sub[node_sub["dataset"] == name]
        for _, r in ds_rows.iterrows():
            pred_nodes[int(r["node_id"])] = (int(r["t"]), int(r["z"]), int(r["y"]), int(r["x"]))

        res = evaluate_detection_predictions(pred_nodes, gt_nodes)
        res["n_total"] = n_total
        exp00_results[name] = res
        print(f"Dataset `{name}` (GT Nodes={res['num_gt']}, Pred={res['num_pred']}, n_total={n_total}):")
        print(f"  TP: {res['tp']} | FP: {res['fp']} | FN: {res['fn']}")
        print(f"  Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f} | F1: {res['f1']:.4f}")
        print(f"  Localization Error: median={res['loc_error_median']:.2f} um, p90={res['loc_error_p90']:.2f} um\n")

    # 2. Classical CV Feasibility Benchmarking
    print("\n--- 2. Benchmarking Classical Computer Vision Detectors on Validation Volumes ---")
    methods = {
        "Heuristic_Baseline_DS4 (EXP00)": detect_baseline_ds4,
        "Downsample_DS2": detect_downsample_ds2,
        "FullRes_Percentile_Threshold": detect_fullres_threshold,
        "Local_Maxima_3D_PeakFinding": detect_local_maxima_3d,
        "Difference_of_Gaussians_DoG": detect_dog_blobs,
        "DistanceTransform_Watershed": detect_watershed_splitting,
    }

    # Evaluate on frames t=0, 20, 40, 60, 80 across all 4 validation datasets
    eval_frames = [0, 20, 40, 60, 80]

    cv_benchmark_rows = []

    for name in VAL_DATASETS:
        zp = os.path.join(TEST_DIR, f"{name}.zarr")
        zg = zarr.open_group(zp, mode="r")
        arr = zg["0"]
        gt_nodes_all, _ = load_gt_for_dataset(name)

        # Filter GT nodes to eval_frames
        gt_nodes_sub = {nid: coords for nid, coords in gt_nodes_all.items() if coords[0] in eval_frames}

        print(f"\nBenchmarking on `{name}` ({len(gt_nodes_sub)} GT nodes across {len(eval_frames)} frames)...")

        for m_name, m_func in methods.items():
            t_start_m = time.time()
            pred_nodes = {}
            nid_counter = 1
            for f_idx in eval_frames:
                vol = arr[f_idx][:]
                pts = m_func(vol)
                for p in pts:
                    pred_nodes[nid_counter] = (f_idx, int(round(p[0])), int(round(p[1])), int(round(p[2])))
                    nid_counter += 1
            t_dur = time.time() - t_start_m

            eval_res = evaluate_detection_predictions(pred_nodes, gt_nodes_sub)
            cv_benchmark_rows.append({
                "dataset": name,
                "method": m_name,
                "num_pred": eval_res["num_pred"],
                "num_gt": eval_res["num_gt"],
                "tp": eval_res["tp"],
                "fp": eval_res["fp"],
                "fn": eval_res["fn"],
                "precision": eval_res["precision"],
                "recall": eval_res["recall"],
                "f1": eval_res["f1"],
                "loc_error_median_um": eval_res["loc_error_median"],
                "loc_error_p90_um": eval_res["loc_error_p90"],
                "runtime_sec": t_dur,
            })
            print(f"  [{m_name:30s}] P={eval_res['precision']:.3f} | R={eval_res['recall']:.3f} | F1={eval_res['f1']:.3f} | Pred={eval_res['num_pred']} | Runtime={t_dur:.2f}s")

    df_cv = pd.DataFrame(cv_benchmark_rows)
    cv_csv_path = os.path.join(OUT_DIR, "detection_analysis.csv")
    df_cv.to_csv(cv_csv_path, index=False)
    print(f"\nSaved detection analysis benchmark to: {cv_csv_path}")

    # Aggregated method performance across all validation sets
    summary_cv = {}
    for m_name in methods:
        sub = df_cv[df_cv["method"] == m_name]
        tot_tp = sub["tp"].sum()
        tot_fp = sub["fp"].sum()
        tot_fn = sub["fn"].sum()
        tot_pred = sub["num_pred"].sum()
        tot_gt = sub["num_gt"].sum()
        prec = (tot_tp / tot_pred) if tot_pred > 0 else 0.0
        rec = (tot_tp / tot_gt) if tot_gt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        summary_cv[m_name] = {
            "total_pred": int(tot_pred),
            "total_tp": int(tot_tp),
            "total_fp": int(tot_fp),
            "total_fn": int(tot_fn),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
            "avg_runtime_sec": float(sub["runtime_sec"].mean()),
        }

    summary_part3 = {
        "exp00_baseline_diagnosis": exp00_results,
        "classical_cv_comparison": summary_cv,
    }

    json_path = os.path.join(OUT_DIR, "analysis_part3_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_part3, f, indent=2)
    print(f"Saved summary metrics to: {json_path}")


if __name__ == "__main__":
    main()
