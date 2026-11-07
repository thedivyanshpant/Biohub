#!/usr/bin/env python3
"""
Biohub Comprehensive Discovery Pass: Part 2
Microscopy Image Intensity, Temporal Drift, Axial/Lateral Variation,
Cell/Nucleus Morphology, and Appearance Dynamics across Edges.
"""

import os
import sys
import glob
import json
import time
import math
import zarr
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import scipy.ndimage as ndi

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # (Z, Y, X) in microns

DATABASE_DIR = os.environ.get("BIOHUB_DATABASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database")))
TRAIN_DIR = os.path.join(DATABASE_DIR, "train")
OUT_DIR = os.environ.get("BIOHUB_CODE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def analyze_dataset_intensity(args):
    dataset_name, zarr_path = args
    try:
        zg = zarr.open_group(zarr_path, mode="r")
        arr = zg["0"]
        num_t = arr.shape[0]

        # Sample 5 timepoints: 0, 24, 49, 74, 99
        sample_t = [0, num_t // 4, num_t // 2, (3 * num_t) // 4, num_t - 1]
        
        frame_means = []
        frame_stds = []
        frame_p10s = []
        frame_p50s = []
        frame_p90s = []
        frame_p99s = []
        
        all_sampled_voxels = []
        
        # Spatial profiles: axial Z profile (mean across t, y, x)
        z_profiles = []

        for t in sample_t:
            vol = arr[t]  # shape (64, 256, 256), uint16
            
            # Subsample by stride 2 for fast distribution calculation
            sub = vol[::2, ::2, ::2].astype(np.float32)
            all_sampled_voxels.append(sub.flatten())
            
            frame_means.append(float(np.mean(sub)))
            frame_stds.append(float(np.std(sub)))
            p_vals = np.percentile(sub, [10, 50, 90, 99])
            frame_p10s.append(float(p_vals[0]))
            frame_p50s.append(float(p_vals[1]))
            frame_p90s.append(float(p_vals[2]))
            frame_p99s.append(float(p_vals[3]))

            # Axial profile for this frame: mean over Y, X
            z_profiles.append(np.mean(vol, axis=(1, 2)).astype(np.float32))

        combined_voxels = np.concatenate(all_sampled_voxels)
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
        percentile_values = np.percentile(combined_voxels, pcts)
        
        global_min = float(np.min(combined_voxels))
        global_max = float(np.max(combined_voxels))
        global_mean = float(np.mean(combined_voxels))
        global_std = float(np.std(combined_voxels))

        # Temporal drift: slope / difference between t=0 and t=last
        mean_drift = float(frame_means[-1] - frame_means[0])
        relative_drift = float(mean_drift / frame_means[0]) if frame_means[0] > 0 else 0.0

        # Mean axial profile across sampled frames (length 64)
        mean_z_prof = np.mean(np.array(z_profiles), axis=0)
        # Z-attenuation: ratio of top slice (z=0) to bottom slice (z=63)
        z_attenuation = float(mean_z_prof[0] / mean_z_prof[-1]) if mean_z_prof[-1] > 0 else 1.0

        return {
            "dataset": dataset_name,
            "global_min": global_min,
            "global_max": global_max,
            "global_mean": global_mean,
            "global_std": global_std,
            "p1": float(percentile_values[0]),
            "p5": float(percentile_values[1]),
            "p10": float(percentile_values[2]),
            "p25": float(percentile_values[3]),
            "p50": float(percentile_values[4]),
            "p75": float(percentile_values[5]),
            "p90": float(percentile_values[6]),
            "p95": float(percentile_values[7]),
            "p99": float(percentile_values[8]),
            "p99_5": float(percentile_values[9]),
            "p99_9": float(percentile_values[10]),
            "mean_t0": frame_means[0],
            "mean_t99": frame_means[-1],
            "mean_drift": mean_drift,
            "relative_drift": relative_drift,
            "z_attenuation_ratio": z_attenuation,
            "mean_z_profile": mean_z_prof.tolist(),
        }
    except Exception as e:
        print(f"Error analyzing intensity for {dataset_name}: {e}")
        return None


def analyze_cell_morphology_and_appearance():
    """Analyzes thousands of cell visual neighborhoods and appearance changes along edges."""
    print("Sampling cell neighborhoods for visual morphology and appearance continuity...")
    # Select 25 diverse datasets
    train_zarrs = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.zarr")))
    sampled_zarrs = train_zarrs[::8][:25]

    cell_morphology_records = []
    edge_appearance_records = []

    for zp in sampled_zarrs:
        name = os.path.basename(zp).replace(".zarr", "")
        gp = os.path.join(TRAIN_DIR, f"{name}.geff")
        if not os.path.exists(gp):
            continue

        try:
            zg = zarr.open_group(zp, mode="r")
            arr = zg["0"]

            rg = zarr.open_group(gp, mode="r")
            n_ids = rg["nodes/ids"][:]
            t_vals = rg["nodes/props/t/values"][:]
            z_vals = rg["nodes/props/z/values"][:]
            y_vals = rg["nodes/props/y/values"][:]
            x_vals = rg["nodes/props/x/values"][:]
            edge_ids = rg["edges/ids"][:]

            nodes_by_id = {}
            for nid, t, z, y, x in zip(n_ids, t_vals, z_vals, y_vals, x_vals):
                nodes_by_id[int(nid)] = (int(t), int(z), int(y), int(x))

            # Sample up to 100 cells per dataset for morphology
            cell_sample_ids = list(nodes_by_id.keys())[:100]

            # Cache loaded 3D volumes by t
            vol_cache = {}

            for nid in cell_sample_ids:
                t, z, y, x = nodes_by_id[nid]
                if t not in vol_cache:
                    vol_cache[t] = arr[t][:]
                vol = vol_cache[t]

                # Bounding box limits: radius Z=5, Y=12, X=12
                z_min, z_max = max(0, z - 5), min(64, z + 6)
                y_min, y_max = max(0, y - 12), min(256, y + 13)
                x_min, x_max = max(0, x - 12), min(256, x + 13)

                patch = vol[z_min:z_max, y_min:y_max, x_min:x_max].astype(np.float32)
                if patch.size == 0:
                    continue

                center_val = float(vol[z, y, x])

                # Measure 1D radial profiles along Z, Y, and X axes from center
                # Z profile
                z_slice = vol[:, y, x].astype(np.float32)
                # XY slice at center z
                xy_slice = vol[z, :, :].astype(np.float32)

                # Local background: ring around cell in XY plane (radius 8..12 voxels)
                y_grid, x_grid = np.ogrid[:256, :256]
                dist_xy = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
                bg_mask = (dist_xy >= 8) & (dist_xy <= 12)
                local_bg = float(np.median(xy_slice[bg_mask])) if np.any(bg_mask) else float(np.percentile(vol, 10))

                sbr = (center_val / local_bg) if local_bg > 0 else 1.0
                contrast = ((center_val - local_bg) / local_bg) if local_bg > 0 else 0.0

                # Measure FWHM in XY and Z
                # In XY: radial profile
                radial_bins = np.arange(0, 15)
                rad_prof = []
                for r in radial_bins:
                    mask_r = (dist_xy >= r - 0.5) & (dist_xy < r + 0.5)
                    if np.any(mask_r):
                        rad_prof.append(float(np.mean(xy_slice[mask_r])))
                    else:
                        rad_prof.append(local_bg)

                # Half maximum intensity:
                half_max = (center_val + local_bg) / 2.0
                # Find radius where profile crosses half_max
                xy_radius_vox = 0.0
                for r_idx, val in enumerate(rad_prof):
                    if val < half_max:
                        xy_radius_vox = float(r_idx)
                        break
                if xy_radius_vox == 0.0:
                    xy_radius_vox = 3.5  # default estimate if profile doesn't fall below

                # In Z: axial profile
                z_dist = np.abs(np.arange(64) - z)
                z_half_max = (center_val + local_bg) / 2.0
                z_radius_vox = 0.0
                for r in range(1, 10):
                    z_lo = max(0, z - r)
                    z_hi = min(63, z + r)
                    val = (float(vol[z_lo, y, x]) + float(vol[z_hi, y, x])) / 2.0
                    if val < z_half_max:
                        z_radius_vox = float(r)
                        break
                if z_radius_vox == 0.0:
                    z_radius_vox = 2.0

                xy_diameter_um = xy_radius_vox * 2 * SCALE[1]
                z_diameter_um = z_radius_vox * 2 * SCALE[0]

                # Max gradient at boundary
                grad_y = np.gradient(xy_slice, axis=0)
                grad_x = np.gradient(xy_slice, axis=1)
                grad_mag = np.sqrt(grad_y**2 + grad_x**2)
                boundary_mask = (dist_xy >= xy_radius_vox - 1) & (dist_xy <= xy_radius_vox + 1)
                max_boundary_grad = float(np.max(grad_mag[boundary_mask])) if np.any(boundary_mask) else 0.0

                cell_morphology_records.append({
                    "dataset": name,
                    "node_id": nid,
                    "center_intensity": center_val,
                    "local_bg_intensity": local_bg,
                    "sbr": sbr,
                    "contrast": contrast,
                    "xy_radius_vox": xy_radius_vox,
                    "xy_diameter_um": xy_diameter_um,
                    "z_radius_vox": z_radius_vox,
                    "z_diameter_um": z_diameter_um,
                    "max_boundary_grad": max_boundary_grad,
                })

            # Sample up to 80 edges per dataset for appearance continuity
            sample_edges = edge_ids[:80]
            for src, tgt in sample_edges:
                s, tg = int(src), int(tgt)
                if s not in nodes_by_id or tg not in nodes_by_id:
                    continue
                ts, zs, ys, xs = nodes_by_id[s]
                tt, zt, yt, xt = nodes_by_id[tg]
                if tt - ts != 1:
                    continue

                if ts not in vol_cache:
                    vol_cache[ts] = arr[ts][:]
                if tt not in vol_cache:
                    vol_cache[tt] = arr[tt][:]

                v_s = vol_cache[ts]
                v_t = vol_cache[tt]

                c_s = float(v_s[zs, ys, xs])
                c_t = float(v_t[zt, yt, xt])
                delta_c = abs(c_t - c_s)
                rel_delta_c = (delta_c / c_s) if c_s > 0 else 0.0

                # 3D patch: shape (5, 9, 9) around each center
                zs_min, zs_max = max(0, zs - 2), min(64, zs + 3)
                ys_min, ys_max = max(0, ys - 4), min(256, ys + 5)
                xs_min, xs_max = max(0, xs - 4), min(256, xs + 5)

                zt_min, zt_max = max(0, zt - 2), min(64, zt + 3)
                yt_min, yt_max = max(0, yt - 4), min(256, yt + 5)
                xt_min, xt_max = max(0, xt - 4), min(256, xt + 5)

                p_s = v_s[zs_min:zs_max, ys_min:ys_max, xs_min:xs_max].astype(np.float32)
                p_t = v_t[zt_min:zt_max, yt_min:yt_max, xt_min:xt_max].astype(np.float32)

                if p_s.shape == p_t.shape and p_s.size > 0:
                    mean_s, std_s = np.mean(p_s), np.std(p_s)
                    mean_t, std_t = np.mean(p_t), np.std(p_t)
                    delta_mean = abs(mean_t - mean_s)
                    delta_std = abs(std_t - std_s)
                    
                    # Normalized Cross Correlation (NCC)
                    if std_s > 1e-4 and std_t > 1e-4:
                        ncc = float(np.mean((p_s - mean_s) * (p_t - mean_t)) / (std_s * std_t))
                    else:
                        ncc = 1.0

                    edge_appearance_records.append({
                        "dataset": name,
                        "src_id": s,
                        "tgt_id": tg,
                        "center_intensity_s": c_s,
                        "center_intensity_t": c_t,
                        "abs_center_delta": delta_c,
                        "rel_center_delta": rel_delta_c,
                        "delta_mean": delta_mean,
                        "delta_std": delta_std,
                        "patch_ncc": ncc,
                    })

        except Exception as e:
            print(f"Error processing morphology for {name}: {e}")

    return pd.DataFrame(cell_morphology_records), pd.DataFrame(edge_appearance_records)


def main():
    t0 = time.time()
    print("================================================================================")
    print("BIOHUB DISCOVERY PASS — PART 2: INTENSITY, MORPHOLOGY & APPEARANCE ANALYSIS")
    print("================================================================================")

    train_zarrs = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.zarr")))
    tasks = [(os.path.basename(zp).replace(".zarr", ""), zp) for zp in train_zarrs]

    print(f"Analyzing intensity across all {len(tasks)} training volumes...")
    with ProcessPoolExecutor(max_workers=32) as executor:
        intensity_results = list(executor.map(analyze_dataset_intensity, tasks))

    intensity_results = [r for r in intensity_results if r is not None]
    df_intensity = pd.DataFrame(intensity_results)
    print(f"Completed intensity pass across all 199 volumes in {time.time() - t0:.2f} s")

    # Analyze morphology and appearance
    t_morph = time.time()
    df_morph, df_appear = analyze_cell_morphology_and_appearance()
    print(f"Completed morphology ({len(df_morph)} cells) and appearance ({len(df_appear)} edges) in {time.time() - t_morph:.2f} s")

    # Compute Summary Statistics
    summary_part2 = {
        "intensity_summary": {
            "global_min_median": float(df_intensity["global_min"].median()),
            "global_min_min": float(df_intensity["global_min"].min()),
            "global_max_median": float(df_intensity["global_max"].median()),
            "global_max_max": float(df_intensity["global_max"].max()),
            "global_mean_median": float(df_intensity["global_mean"].median()),
            "global_mean_min": float(df_intensity["global_mean"].min()),
            "global_mean_max": float(df_intensity["global_mean"].max()),
            "global_std_median": float(df_intensity["global_std"].median()),
            "percentiles_aggregate": {
                "p1_median": float(df_intensity["p1"].median()),
                "p5_median": float(df_intensity["p5"].median()),
                "p10_median": float(df_intensity["p10"].median()),
                "p25_median": float(df_intensity["p25"].median()),
                "p50_median": float(df_intensity["p50"].median()),
                "p75_median": float(df_intensity["p75"].median()),
                "p90_median": float(df_intensity["p90"].median()),
                "p95_median": float(df_intensity["p95"].median()),
                "p99_median": float(df_intensity["p99"].median()),
                "p99_5_median": float(df_intensity["p99_5"].median()),
                "p99_9_median": float(df_intensity["p99_9"].median()),
            },
            "temporal_drift": {
                "mean_drift_median": float(df_intensity["mean_drift"].median()),
                "relative_drift_median": float(df_intensity["relative_drift"].median()),
                "relative_drift_p10": float(df_intensity["relative_drift"].quantile(0.1)),
                "relative_drift_p90": float(df_intensity["relative_drift"].quantile(0.9)),
                "max_bleaching_loss_pct": float(df_intensity["relative_drift"].min() * 100),
            },
            "axial_variation": {
                "z_attenuation_ratio_median": float(df_intensity["z_attenuation_ratio"].median()),
                "z_attenuation_ratio_p10": float(df_intensity["z_attenuation_ratio"].quantile(0.1)),
                "z_attenuation_ratio_p90": float(df_intensity["z_attenuation_ratio"].quantile(0.9)),
            },
            "intensity_regimes": {
                "num_datasets_under_100_mean": int((df_intensity["global_mean"] < 100).sum()),
                "num_datasets_100_to_500_mean": int(((df_intensity["global_mean"] >= 100) & (df_intensity["global_mean"] < 500)).sum()),
                "num_datasets_over_500_mean": int((df_intensity["global_mean"] >= 500).sum()),
            }
        },
        "morphology_summary": {
            "total_cells_analyzed": len(df_morph),
            "xy_radius_vox_median": float(df_morph["xy_radius_vox"].median()),
            "xy_radius_vox_mean": float(df_morph["xy_radius_vox"].mean()),
            "xy_diameter_um_median": float(df_morph["xy_diameter_um"].median()),
            "xy_diameter_um_mean": float(df_morph["xy_diameter_um"].mean()),
            "xy_diameter_um_p10": float(df_morph["xy_diameter_um"].quantile(0.10)),
            "xy_diameter_um_p90": float(df_morph["xy_diameter_um"].quantile(0.90)),
            "z_radius_vox_median": float(df_morph["z_radius_vox"].median()),
            "z_diameter_um_median": float(df_morph["z_diameter_um"].median()),
            "z_diameter_um_mean": float(df_morph["z_diameter_um"].mean()),
            "sbr_median": float(df_morph["sbr"].median()),
            "sbr_mean": float(df_morph["sbr"].mean()),
            "contrast_median": float(df_morph["contrast"].median()),
            "contrast_mean": float(df_morph["contrast"].mean()),
            "contrast_p10": float(df_morph["contrast"].quantile(0.10)),
            "contrast_p90": float(df_morph["contrast"].quantile(0.90)),
        },
        "appearance_dynamics": {
            "total_edges_analyzed": len(df_appear),
            "patch_ncc_median": float(df_appear["patch_ncc"].median()),
            "patch_ncc_mean": float(df_appear["patch_ncc"].mean()),
            "patch_ncc_p10": float(df_appear["patch_ncc"].quantile(0.10)),
            "patch_ncc_p90": float(df_appear["patch_ncc"].quantile(0.90)),
            "rel_center_intensity_delta_median": float(df_appear["rel_center_delta"].median()),
            "rel_center_intensity_delta_p90": float(df_appear["rel_center_delta"].quantile(0.90)),
        }
    }

    # Save summary JSON
    json_path = os.path.join(OUT_DIR, "analysis_part2_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_part2, f, indent=2)
    print(f"Saved intensity & morphology summary to: {json_path}")
    print("\n--- Summary Preview ---")
    print(json.dumps(summary_part2, indent=2))


if __name__ == "__main__":
    main()
