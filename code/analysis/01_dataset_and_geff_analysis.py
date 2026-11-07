#!/usr/bin/env python3
"""
Biohub Comprehensive Discovery Pass: Part 1
Dataset Structure, GEFF Topology, Motion Dynamics, Divisions, Lineages, Sparsity, and Tracking Ambiguity.
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
import scipy.spatial.distance as sp_dist

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # (Z, Y, X) in microns

DATABASE_DIR = os.environ.get("BIOHUB_DATABASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database")))
TRAIN_DIR = os.path.join(DATABASE_DIR, "train")
TEST_DIR = os.path.join(DATABASE_DIR, "test")
OUT_DIR = os.environ.get("BIOHUB_CODE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_dir_size(path):
    """Calculates total size in bytes of a directory recursively."""
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def process_single_dataset(args):
    dataset_name, zarr_path, geff_path, is_test = args

    # 1. Zarr metadata
    zarr_size = get_dir_size(zarr_path)
    geff_size = get_dir_size(geff_path) if geff_path and os.path.exists(geff_path) else 0

    shape = None
    dtype = None
    chunks = None
    timepoints, Z, Y, X = None, None, None, None
    voxel_spacing = [1.0, 1.625, 0.40625, 0.40625]

    try:
        zg = zarr.open_group(zarr_path, mode="r")
        arr = zg["0"]
        shape = list(arr.shape)
        dtype = str(arr.dtype)
        chunks = list(arr.chunks)
        timepoints, Z, Y, X = shape[0], shape[1], shape[2], shape[3]
    except Exception as e:
        print(f"Error opening Zarr {zarr_path}: {e}")

    # 2. GEFF metadata & arrays
    node_count = 0
    edge_count = 0
    div_count = 0
    lineage_count = 0
    n_total = None

    nodes_dict = {}  # nid -> (t, z, y, x)
    edges_list = []  # (src, tgt)
    out_degrees = {}
    in_degrees = {}

    if geff_path and os.path.exists(geff_path):
        try:
            with open(os.path.join(geff_path, "zarr.json")) as f:
                meta = json.load(f)
            n_total = meta.get("attributes", {}).get("geff", {}).get("extra", {}).get("estimated_number_of_nodes")
        except Exception:
            pass

        try:
            root = zarr.open_group(geff_path, mode="r")
            node_ids = root["nodes/ids"][:]
            t_vals = root["nodes/props/t/values"][:]
            z_vals = root["nodes/props/z/values"][:]
            y_vals = root["nodes/props/y/values"][:]
            x_vals = root["nodes/props/x/values"][:]
            edge_ids = root["edges/ids"][:]

            node_count = len(node_ids)
            edge_count = len(edge_ids)

            for nid, t, z, y, x in zip(node_ids, t_vals, z_vals, y_vals, x_vals):
                nodes_dict[int(nid)] = (int(t), int(z), int(y), int(x))

            for src, tgt in edge_ids:
                s, t = int(src), int(tgt)
                edges_list.append((s, t))
                out_degrees[s] = out_degrees.get(s, 0) + 1
                in_degrees[t] = in_degrees.get(t, 0) + 1

            for s, deg in out_degrees.items():
                if deg >= 2:
                    div_count += 1

            # Lineage count: roots in directed forest (in_degree == 0 among nodes that appear in edges or all nodes)
            # Standard lineage definition: connected component in undirected projection
            adj = {}
            for nid in nodes_dict:
                adj[nid] = []
            for s, t in edges_list:
                adj[s].append(t)
                adj[t].append(s)

            visited = set()
            for nid in nodes_dict:
                if nid not in visited:
                    lineage_count += 1
                    q = [nid]
                    visited.add(nid)
                    while q:
                        curr = q.pop()
                        for neighbor in adj.get(curr, []):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
        except Exception as e:
            print(f"Error reading GEFF {geff_path}: {e}")

    # Return dataset level summary row
    ds_row = {
        "dataset": dataset_name,
        "is_test": is_test,
        "timepoints": timepoints,
        "Z": Z,
        "Y": Y,
        "X": X,
        "dtype": dtype,
        "voxel_spacing": "1.0, 1.625, 0.40625, 0.40625",
        "chunk_shape": str(chunks),
        "zarr_size_bytes": zarr_size,
        "zarr_size_mb": round(zarr_size / (1024 * 1024), 2),
        "geff_size_bytes": geff_size,
        "geff_size_mb": round(geff_size / (1024 * 1024), 4),
        "node_count": node_count,
        "edge_count": edge_count,
        "division_count": div_count,
        "lineage_count": lineage_count,
        "estimated_number_of_nodes": n_total,
        "sparsity_ratio": (node_count / n_total) if (n_total and n_total > 0) else None,
        "avg_nodes_per_frame": round(node_count / timepoints, 2) if timepoints else None,
    }

    # 3. Detailed Edge Motion Analysis
    motion_records = []
    edges_by_src = {}
    for s, tgt in edges_list:
        edges_by_src.setdefault(s, []).append(tgt)

    for s, tgt in edges_list:
        if s not in nodes_dict or tgt not in nodes_dict:
            continue
        ts, zs, ys, xs = nodes_dict[s]
        tt, zt, yt, xt = nodes_dict[tgt]
        dt = tt - ts
        dz = zt - zs
        dy = yt - ys
        dx = xt - xs
        dz_um = dz * SCALE[0]
        dy_um = dy * SCALE[1]
        dx_um = dx * SCALE[2]
        disp_um = math.sqrt(dz_um**2 + dy_um**2 + dx_um**2)
        vel_um_s = disp_um / dt if dt > 0 else disp_um

        motion_records.append({
            "dataset": dataset_name,
            "src_id": s,
            "tgt_id": tgt,
            "t_src": ts,
            "t_tgt": tt,
            "dt": dt,
            "dz_vox": dz,
            "dy_vox": dy,
            "dx_vox": dx,
            "dz_um": dz_um,
            "dy_um": dy_um,
            "dx_um": dx_um,
            "disp_um": disp_um,
            "vel_um_s": vel_um_s,
            "sq_disp_um2": disp_um**2,
        })

    # 4. Detailed Division Analysis
    division_records = []
    for s, targets in edges_by_src.items():
        if len(targets) >= 2:
            # Division event!
            ts, zs, ys, xs = nodes_dict[s]
            p_pos = np.array([zs * SCALE[0], ys * SCALE[1], xs * SCALE[2]])

            # Sort daughters by id
            d1, d2 = targets[0], targets[1]
            td1, zd1, yd1, xd1 = nodes_dict[d1]
            td2, zd2, yd2, xd2 = nodes_dict[d2]

            d1_pos = np.array([zd1 * SCALE[0], yd1 * SCALE[1], xd1 * SCALE[2]])
            d2_pos = np.array([zd2 * SCALE[0], yd2 * SCALE[1], xd2 * SCALE[2]])

            p_to_d1 = float(np.linalg.norm(d1_pos - p_pos))
            p_to_d2 = float(np.linalg.norm(d2_pos - p_pos))
            d1_to_d2 = float(np.linalg.norm(d1_pos - d2_pos))

            # Direction angle between daughters relative to parent
            v1 = d1_pos - p_pos
            v2 = d2_pos - p_pos
            dot = np.dot(v1, v2)
            norm_prod = (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_deg = math.degrees(math.acos(np.clip(dot / norm_prod, -1.0, 1.0))) if norm_prod > 0 else 0.0

            division_records.append({
                "dataset": dataset_name,
                "parent_id": s,
                "t_div": ts,
                "z_p": zs, "y_p": ys, "x_p": xs,
                "daughter1_id": d1,
                "daughter2_id": d2,
                "p_to_d1_dist_um": p_to_d1,
                "p_to_d2_dist_um": p_to_d2,
                "d1_to_d2_dist_um": d1_to_d2,
                "daughter_divergence_angle_deg": angle_deg,
            })

    # 5. Spatial Nearest-Neighbor & Tracking Ambiguity Analysis
    # Group nodes by timepoint
    nodes_by_t = {}
    for nid, (t, z, y, x) in nodes_dict.items():
        nodes_by_t.setdefault(t, []).append((nid, z * SCALE[0], y * SCALE[1], x * SCALE[2]))

    nn1_list = []
    nn2_list = []
    competing_counts = {3: 0, 5: 0, 7: 0, 10: 0, 15: 0, 20: 0}
    total_edges_checked = 0

    for t, nlist in nodes_by_t.items():
        if len(nlist) > 1:
            coords = np.array([[n[1], n[2], n[3]] for n in nlist])
            dmat = sp_dist.cdist(coords, coords)
            np.fill_diagonal(dmat, np.inf)
            dmat_sorted = np.sort(dmat, axis=1)
            nn1_list.extend(dmat_sorted[:, 0].tolist())
            if dmat_sorted.shape[1] > 1:
                nn2_list.extend(dmat_sorted[:, 1].tolist())

    # Tracking ambiguity: for each edge u -> v, check other nodes in frame t
    for s, tgt in edges_list:
        if s not in nodes_dict or tgt not in nodes_dict:
            continue
        ts, zs, ys, xs = nodes_dict[s]
        tt, zt, yt, xt = nodes_dict[tgt]
        if tt in nodes_by_t:
            u_phys = np.array([zs * SCALE[0], ys * SCALE[1], xs * SCALE[2]])
            cand_list = nodes_by_t[tt]
            c_coords = np.array([[c[1], c[2], c[3]] for c in cand_list])
            dists = np.linalg.norm(c_coords - u_phys, axis=1)
            total_edges_checked += 1
            for rad in [3, 5, 7, 10, 15, 20]:
                # count candidates within rad excluding the true target
                cnt = int(np.sum(dists <= rad)) - 1
                if cnt > 0:
                    competing_counts[rad] += 1

    # 6. Smoothness & Acceleration Predictability
    # For triples u -> v -> w
    pred_errors = []
    for s, targets in edges_by_src.items():
        for mid in targets:
            for w in edges_by_src.get(mid, []):
                # Triple s -> mid -> w
                tu, zu, yu, xu = nodes_dict[s]
                tv, zv, yv, xv = nodes_dict[mid]
                tw, zw, yw, xw = nodes_dict[w]
                if tv - tu == 1 and tw - tv == 1:
                    pu = np.array([zu * SCALE[0], yu * SCALE[1], xu * SCALE[2]])
                    pv = np.array([zv * SCALE[0], yv * SCALE[1], xv * SCALE[2]])
                    pw = np.array([zw * SCALE[0], yw * SCALE[1], xw * SCALE[2]])
                    # Extrapolation: pw_hat = pv + (pv - pu)
                    pw_hat = 2 * pv - pu
                    err = float(np.linalg.norm(pw - pw_hat))
                    pred_errors.append(err)

    # 7. Lineage length & depth analysis
    lineage_stats = []
    # Build forward graph for lineages
    in_degree_gt = {}
    for s, tgt in edges_list:
        in_degree_gt[tgt] = in_degree_gt.get(tgt, 0) + 1

    roots = [nid for nid in nodes_dict if in_degree_gt.get(nid, 0) == 0]
    for r in roots:
        # Explore tree
        q = [(r, 1)]
        max_depth = 1
        num_cells = 0
        num_divs = 0
        while q:
            curr, depth = q.pop()
            num_cells += 1
            max_depth = max(max_depth, depth)
            succs = edges_by_src.get(curr, [])
            if len(succs) >= 2:
                num_divs += 1
            for child in succs:
                q.append((child, depth + 1))
        lineage_stats.append({
            "root_id": r,
            "max_depth": max_depth,
            "cell_count": num_cells,
            "division_count": num_divs,
            "is_branching": num_divs > 0,
        })

    return {
        "dataset_row": ds_row,
        "motion_records": motion_records,
        "division_records": division_records,
        "nn1": nn1_list,
        "nn2": nn2_list,
        "competing_counts": competing_counts,
        "total_edges_checked": total_edges_checked,
        "pred_errors": pred_errors,
        "lineage_stats": lineage_stats,
    }


def main():
    t0 = time.time()
    print("================================================================================")
    print("BIOHUB DISCOVERY PASS — PART 1: METADATA, GEFF TOPOLOGY, MOTION & LINEAGE ANALYSIS")
    print("================================================================================")

    # Collect all datasets
    train_zarrs = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.zarr")))
    test_zarrs = sorted(glob.glob(os.path.join(TEST_DIR, "*.zarr")))

    tasks = []
    for zp in train_zarrs:
        name = os.path.basename(zp).replace(".zarr", "")
        gp = os.path.join(TRAIN_DIR, f"{name}.geff")
        tasks.append((name, zp, gp, False))

    for zp in test_zarrs:
        name = os.path.basename(zp).replace(".zarr", "")
        # For public test, we know the matching GEFF in train
        gp = os.path.join(TRAIN_DIR, f"{name}.geff")
        tasks.append((name, zp, gp, True))

    print(f"Total tasks prepared: {len(tasks)} (199 train + {len(test_zarrs)} test)")

    # Execute with multiprocessing across 32 workers
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(process_single_dataset, tasks))

    print(f"Processed all {len(results)} datasets in {time.time() - t0:.2f} s")

    # Aggregate results
    dataset_rows = []
    all_motion = []
    all_divisions = []
    all_nn1 = []
    all_nn2 = []
    tot_competing = {3: 0, 5: 0, 7: 0, 10: 0, 15: 0, 20: 0}
    tot_edges_checked = 0
    all_pred_errors = []
    all_lineages = []

    for r in results:
        dataset_rows.append(r["dataset_row"])
        all_motion.extend(r["motion_records"])
        all_divisions.extend(r["division_records"])
        all_nn1.extend(r["nn1"])
        all_nn2.extend(r["nn2"])
        tot_edges_checked += r["total_edges_checked"]
        for rad in tot_competing:
            tot_competing[rad] += r["competing_counts"][rad]
        all_pred_errors.extend(r["pred_errors"])
        all_lineages.extend(r["lineage_stats"])

    # 1. Dataset statistics CSV
    df_datasets = pd.DataFrame(dataset_rows)
    df_train_only = df_datasets[~df_datasets["is_test"]]
    csv_ds_path = os.path.join(OUT_DIR, "train_dataset_statistics.csv")
    df_datasets.to_csv(csv_ds_path, index=False)
    print(f"Saved dataset metadata to: {csv_ds_path} ({len(df_datasets)} rows)")

    # 2. Motion statistics CSV
    df_motion = pd.DataFrame(all_motion)
    csv_motion_path = os.path.join(OUT_DIR, "cell_motion_statistics.csv")
    # To keep CSV compact as requested, save summary percentiles per dataset or sample/downsample if large
    # Let's see: 128,883 rows is ~12 MB, let's save full or aggregated
    df_motion.to_csv(csv_motion_path, index=False)
    print(f"Saved cell motion statistics to: {csv_motion_path} ({len(df_motion):,} rows)")

    # 3. Division statistics CSV
    df_div = pd.DataFrame(all_divisions)
    csv_div_path = os.path.join(OUT_DIR, "division_statistics.csv")
    df_div.to_csv(csv_div_path, index=False)
    print(f"Saved division statistics to: {csv_div_path} ({len(df_div):,} rows)")

    # 4. Compute High-Level Metrics for Report
    motion_disp = df_motion["disp_um"].values
    motion_dz = df_motion["dz_um"].values
    motion_dy = df_motion["dy_um"].values
    motion_dx = df_motion["dx_um"].values

    summary_metrics = {
        "dataset_counts": {
            "total_train_datasets": len(df_train_only),
            "total_nodes": int(df_train_only["node_count"].sum()),
            "total_edges": int(df_train_only["edge_count"].sum()),
            "total_divisions": int(df_train_only["division_count"].sum()),
            "total_lineages": int(df_train_only["lineage_count"].sum()),
            "nodes_per_ds": {
                "min": int(df_train_only["node_count"].min()),
                "median": float(df_train_only["node_count"].median()),
                "mean": float(df_train_only["node_count"].mean()),
                "max": int(df_train_only["node_count"].max()),
                "std": float(df_train_only["node_count"].std()),
                "p10": float(np.percentile(df_train_only["node_count"], 10)),
                "p25": float(np.percentile(df_train_only["node_count"], 25)),
                "p75": float(np.percentile(df_train_only["node_count"], 75)),
                "p90": float(np.percentile(df_train_only["node_count"], 90)),
            },
            "edges_per_ds": {
                "min": int(df_train_only["edge_count"].min()),
                "median": float(df_train_only["edge_count"].median()),
                "mean": float(df_train_only["edge_count"].mean()),
                "max": int(df_train_only["edge_count"].max()),
                "std": float(df_train_only["edge_count"].std()),
            },
            "divisions_per_ds": {
                "min": int(df_train_only["division_count"].min()),
                "median": float(df_train_only["division_count"].median()),
                "mean": float(df_train_only["division_count"].mean()),
                "max": int(df_train_only["division_count"].max()),
                "nonzero_datasets": int((df_train_only["division_count"] > 0).sum()),
            },
            "sparsity_ratio": {
                "min": float(df_train_only["sparsity_ratio"].dropna().min()),
                "median": float(df_train_only["sparsity_ratio"].dropna().median()),
                "mean": float(df_train_only["sparsity_ratio"].dropna().mean()),
                "max": float(df_train_only["sparsity_ratio"].dropna().max()),
                "p10": float(np.percentile(df_train_only["sparsity_ratio"].dropna(), 10)),
                "p90": float(np.percentile(df_train_only["sparsity_ratio"].dropna(), 90)),
            }
        },
        "motion_metrics": {
            "disp_um": {
                "min": float(np.min(motion_disp)),
                "mean": float(np.mean(motion_disp)),
                "median": float(np.median(motion_disp)),
                "std": float(np.std(motion_disp)),
                "p50": float(np.percentile(motion_disp, 50)),
                "p75": float(np.percentile(motion_disp, 75)),
                "p90": float(np.percentile(motion_disp, 90)),
                "p95": float(np.percentile(motion_disp, 95)),
                "p99": float(np.percentile(motion_disp, 99)),
                "p99_9": float(np.percentile(motion_disp, 99.9)),
                "max": float(np.max(motion_disp)),
            },
            "component_disp_um": {
                "dz_median": float(np.median(np.abs(motion_dz))),
                "dz_p95": float(np.percentile(np.abs(motion_dz), 95)),
                "dy_median": float(np.median(np.abs(motion_dy))),
                "dy_p95": float(np.percentile(np.abs(motion_dy), 95)),
                "dx_median": float(np.median(np.abs(motion_dx))),
                "dx_p95": float(np.percentile(np.abs(motion_dx), 95)),
            },
            "linear_extrapolation_error_um": {
                "count": len(all_pred_errors),
                "median": float(np.median(all_pred_errors)),
                "mean": float(np.mean(all_pred_errors)),
                "p90": float(np.percentile(all_pred_errors, 90)),
                "p95": float(np.percentile(all_pred_errors, 95)),
            }
        },
        "spatial_density_nn": {
            "nn1_um": {
                "min": float(np.min(all_nn1)),
                "p1": float(np.percentile(all_nn1, 1)),
                "p5": float(np.percentile(all_nn1, 5)),
                "p10": float(np.percentile(all_nn1, 10)),
                "median": float(np.median(all_nn1)),
                "mean": float(np.mean(all_nn1)),
                "p90": float(np.percentile(all_nn1, 90)),
                "max": float(np.max(all_nn1)),
                "fraction_closer_than_3um": float(np.mean(np.array(all_nn1) < 3.0)),
                "fraction_closer_than_5um": float(np.mean(np.array(all_nn1) < 5.0)),
                "fraction_closer_than_7um": float(np.mean(np.array(all_nn1) < 7.0)),
            },
            "nn2_um": {
                "median": float(np.median(all_nn2)),
                "mean": float(np.mean(all_nn2)),
                "p90": float(np.percentile(all_nn2, 90)),
            }
        },
        "tracking_ambiguity": {
            "total_edges_evaluated": tot_edges_checked,
            "fraction_with_competitors": {
                "3um": float(tot_competing[3] / tot_edges_checked),
                "5um": float(tot_competing[5] / tot_edges_checked),
                "7um": float(tot_competing[7] / tot_edges_checked),
                "10um": float(tot_competing[10] / tot_edges_checked),
                "15um": float(tot_competing[15] / tot_edges_checked),
                "20um": float(tot_competing[20] / tot_edges_checked),
            }
        },
        "division_metrics": {
            "total_divisions": len(df_div),
            "p_to_d1_dist_median": float(df_div["p_to_d1_dist_um"].median()),
            "p_to_d1_dist_p95": float(df_div["p_to_d1_dist_um"].quantile(0.95)),
            "p_to_d2_dist_median": float(df_div["p_to_d2_dist_um"].median()),
            "p_to_d2_dist_p95": float(df_div["p_to_d2_dist_um"].quantile(0.95)),
            "d1_to_d2_dist_median": float(df_div["d1_to_d2_dist_um"].median()),
            "d1_to_d2_dist_p95": float(df_div["d1_to_d2_dist_um"].quantile(0.95)),
            "divergence_angle_deg_median": float(df_div["daughter_divergence_angle_deg"].median()),
            "divergence_angle_deg_mean": float(df_div["daughter_divergence_angle_deg"].mean()),
        },
        "lineage_metrics": {
            "total_trees": len(all_lineages),
            "branching_trees_count": sum(1 for l in all_lineages if l["is_branching"]),
            "branching_trees_fraction": float(sum(1 for l in all_lineages if l["is_branching"]) / len(all_lineages)),
            "depth_distribution": {
                "median": float(np.median([l["max_depth"] for l in all_lineages])),
                "mean": float(np.mean([l["max_depth"] for l in all_lineages])),
                "p90": float(np.percentile([l["max_depth"] for l in all_lineages], 90)),
                "max": int(max(l["max_depth"] for l in all_lineages)),
            },
            "cells_per_lineage": {
                "median": float(np.median([l["cell_count"] for l in all_lineages])),
                "mean": float(np.mean([l["cell_count"] for l in all_lineages])),
                "p90": float(np.percentile([l["cell_count"] for l in all_lineages], 90)),
                "max": int(max(l["cell_count"] for l in all_lineages)),
            }
        }
    }

    json_path = os.path.join(OUT_DIR, "analysis_part1_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Saved summary metrics to: {json_path}")
    print("\n--- Summary Preview ---")
    print(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()
