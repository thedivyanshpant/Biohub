#!/usr/bin/env python3
"""
Biohub Baseline: Faithful Python reproduction of the Kaggle top notebook
(cell-tracking-getting-started-w-nearest-neighbor.ipynb).
"""

import os
import json
import zarr
import numpy as np
import pandas as pd
from scipy.ndimage import label, uniform_filter, center_of_mass
from scipy.optimize import linear_sum_assignment


def load_zarr_volume_accessor(zarr_path: str):
    """
    Opens Zarr v3 volume and verifies array metadata.
    Returns the Zarr array accessor for reading 3D frames.
    """
    zg = zarr.open_group(zarr_path, mode="r")
    arr = zg["0"]
    shape = tuple(arr.shape)
    dtype = np.dtype(arr.dtype)
    assert len(shape) == 4 and shape == (100, 64, 256, 256), f"Unexpected shape: {shape}"
    assert dtype == np.uint16, f"Unexpected dtype: {dtype}"
    return arr, shape, dtype


def detect_cells_in_frame(vol: np.ndarray, downsample: int = 4, filter_size: int = 3, percentile: float = 90.0):
    """
    Faithful implementation of cell detection from the Kaggle notebook:
      1. Spatial downsampling [::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
      2. 3D uniform filter (size=3)
      3. Global percentile threshold (90%)
      4. 3D connected component labeling
      5. Center of mass centroids scaled back to native coordinates and rounded to integer.
    """
    ds = vol[::downsample, ::downsample, ::downsample]
    smoothed = uniform_filter(ds.astype(np.float32), size=filter_size)
    threshold = np.percentile(smoothed, percentile)
    binary = smoothed > threshold

    labeled, n_features = label(binary)

    centroids = []
    if n_features > 0:
        coms = center_of_mass(binary, labeled, range(1, n_features + 1))
        for c in coms:
            c_native = np.array(c, dtype=np.float64) * downsample
            centroids.append((
                int(round(c_native[0])),
                int(round(c_native[1])),
                int(round(c_native[2]))
            ))
    return centroids


def track_consecutive_frames(
    prev_centroids: dict,  # nid -> (z, y, x)
    curr_centroids: dict,  # nid -> (z, y, x)
    scale: np.ndarray,
    max_link_distance: float = 15.0
):
    """
    Faithful implementation of Hungarian bipartite association from Kaggle notebook:
      1. Convert voxel coordinates to physical micrometers via SCALE.
      2. Compute pairwise Euclidean distance matrix.
      3. Solve optimal linear sum assignment.
      4. Accept edges with physical distance <= MAX_LINK_DISTANCE.
    """
    edges = []
    if not prev_centroids or not curr_centroids:
        return edges

    prev_ids = list(prev_centroids.keys())
    curr_ids = list(curr_centroids.keys())

    prev_coords = np.array([prev_centroids[pid] for pid in prev_ids], dtype=np.float64)
    curr_coords = np.array([curr_centroids[cid] for cid in curr_ids], dtype=np.float64)

    prev_phys = prev_coords * scale
    curr_phys = curr_coords * scale

    # Physical Euclidean distance matrix: shape (N_prev, N_curr)
    dist = np.sqrt(((prev_phys[:, None] - curr_phys[None, :]) ** 2).sum(axis=2))
    row_ind, col_ind = linear_sum_assignment(dist)

    for ri, ci in zip(row_ind, col_ind):
        if dist[ri, ci] <= max_link_distance:
            edges.append((prev_ids[ri], curr_ids[ci]))

    return edges


def process_volume(folder_name: str, zarr_path: str, config: dict):
    """
    Processes one complete 4D Zarr volume through the baseline pipeline:
      - Reads frames t=0..99
      - Detects cells
      - Tracks between consecutive frames
      - Formats rows in exact Kaggle submission schema
    """
    alg = config["algorithm"]
    downsample = alg["downsample"]
    filter_size = alg["filter_size"]
    percentile = alg["percentile"]
    max_link_distance = alg["max_link_distance"]
    scale = np.array(alg["scale"], dtype=np.float64)

    arr, shape, dtype = load_zarr_volume_accessor(zarr_path)
    n_t = shape[0]

    all_rows = []
    pred_nodes = {}  # nid -> (t, z, y, x)
    pred_edges = []  # (src_id, tgt_id)

    prev_centroids = {}
    node_id_counter = 1

    for t in range(n_t):
        vol = arr[t][:]  # Shape (64, 256, 256), uint16
        centroids_t = detect_cells_in_frame(
            vol=vol,
            downsample=downsample,
            filter_size=filter_size,
            percentile=percentile
        )

        curr_centroids = {}
        for c in centroids_t:
            nid = node_id_counter
            node_id_counter += 1
            curr_centroids[nid] = c
            pred_nodes[nid] = (t, c[0], c[1], c[2])

            all_rows.append({
                "dataset": folder_name,
                "row_type": "node",
                "node_id": nid,
                "t": t,
                "z": c[0],
                "y": c[1],
                "x": c[2],
                "source_id": -1,
                "target_id": -1,
            })

        # Track between frame t-1 and frame t
        if prev_centroids and curr_centroids:
            edges_t = track_consecutive_frames(
                prev_centroids=prev_centroids,
                curr_centroids=curr_centroids,
                scale=scale,
                max_link_distance=max_link_distance
            )
            for src_id, tgt_id in edges_t:
                pred_edges.append((src_id, tgt_id))
                all_rows.append({
                    "dataset": folder_name,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": src_id,
                    "target_id": tgt_id,
                })

        prev_centroids = curr_centroids

    return all_rows, pred_nodes, pred_edges
