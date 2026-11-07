#!/usr/bin/env python3
"""
Biohub Classical Detector V2: Full-Resolution Anisotropic Detectors.
Contains:
  1. Normalization modules (percentile, depth-aware, local background subtraction)
  2. 3D Anisotropic Local Maxima Peak Finding
  3. 3D Anisotropic Difference of Gaussians (DoG) Blob Detection
  4. 3D Distance Transform + Seeded Watershed Splitting
  5. Multi-Detector Ensembling & Deduplication
"""

import math
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # (Z, Y, X) in µm


# ==============================================================================
# 1. NORMALIZATION ROUTINES
# ==============================================================================

def normalize_percentile(vol: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Robust percentile clipping and scaling to [0, 1]."""
    v_float = vol.astype(np.float32)
    lo, hi = np.percentile(v_float, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(v_float)
    clipped = np.clip(v_float, lo, hi)
    return (clipped - lo) / (hi - lo)


def normalize_depth_aware(vol: np.ndarray, p_low: float = 10.0, p_high: float = 95.0) -> np.ndarray:
    """
    Per-Z-slice background subtraction and dynamic range scaling to counteract
    depth-dependent axial light attenuation.
    """
    v_float = vol.astype(np.float32)
    num_z = v_float.shape[0]
    out = np.zeros_like(v_float)

    z_los = []
    z_his = []
    for z in range(num_z):
        sl = v_float[z]
        lo, hi = np.percentile(sl, [p_low, p_high])
        z_los.append(lo)
        z_his.append(hi)

    # Smooth the curves along Z to avoid inter-slice boundary artifacts
    z_los_smooth = ndi.gaussian_filter1d(np.array(z_los, dtype=np.float32), sigma=2.0)
    z_his_smooth = ndi.gaussian_filter1d(np.array(z_his, dtype=np.float32), sigma=2.0)

    for z in range(num_z):
        lo = z_los_smooth[z]
        hi = max(z_his_smooth[z], lo + 1e-4)
        out[z] = np.clip((v_float[z] - lo) / (hi - lo), 0.0, 1.0)

    return out


def normalize_local_background_subtraction(
    vol: np.ndarray,
    bg_size=(3, 21, 21),
    p_low=1.0,
    p_high=99.0
) -> np.ndarray:
    """
    Estimates non-uniform tissue autofluorescence via morphological/uniform low-pass filter
    and subtracts it from the signal.
    """
    v_norm = normalize_percentile(vol, p_low=p_low, p_high=p_high)
    bg = ndi.uniform_filter(v_norm, size=bg_size)
    fg = np.maximum(0.0, v_norm - bg)
    fg_max = np.percentile(fg, 99.5)
    if fg_max > 0:
        fg = np.clip(fg / fg_max, 0.0, 1.0)
    return fg


# ==============================================================================
# 2. ELLIPSOIDAL FOOTPRINT HELPER
# ==============================================================================

def make_anisotropic_footprint(radius_z: int = 2, radius_xy: int = 5) -> np.ndarray:
    """
    Creates an anisotropic 3D ellipsoidal footprint for maximum filtering.
    Enforces minimum separation: dz voxels in Z and dxy voxels in lateral XY.
    """
    rz, rxy = int(radius_z), int(radius_xy)
    z, y, x = np.ogrid[-rz:rz + 1, -rxy:rxy + 1, -rxy:rxy + 1]
    mask = (z / max(1, rz))**2 + (y / max(1, rxy))**2 + (x / max(1, rxy))**2 <= 1.0
    return mask.astype(bool)


# ==============================================================================
# 3. DETECTOR IMPLEMENTATIONS
# ==============================================================================

def detect_baseline_native(vol: np.ndarray, percentile: float = 97.0, filter_size: int = 3):
    """
    Control detector: Runs the baseline uniform filter + threshold + connected components
    at FULL native resolution (no 4x downsampling).
    """
    v_float = vol.astype(np.float32)
    smoothed = ndi.uniform_filter(v_float, size=filter_size)
    thresh = np.percentile(smoothed, percentile)
    binary = smoothed > thresh

    labeled, n_features = ndi.label(binary)
    if n_features == 0:
        return []

    centroids = ndi.center_of_mass(binary, labeled, range(1, n_features + 1))
    return [(int(round(c[0])), int(round(c[1])), int(round(c[2])), 1.0) for c in centroids]


def detect_local_maxima(
    vol: np.ndarray,
    norm_method: str = "percentile",
    sigma: tuple = (1.2, 2.5, 2.5),
    radius_z: int = 2,
    radius_xy: int = 5,
    threshold_pct: float = 90.0,
):
    """
    3D Anisotropic Local Maxima Peak Finding:
      1. Preprocesses volume with selected normalization.
      2. 3D Gaussian smoothing with anisotropic kernel (accounting for PSF elongation).
      3. Non-maximum suppression via anisotropic ellipsoidal footprint.
      4. Thresholding by intensity percentile.
    """
    if norm_method == "depth_aware":
        v_norm = normalize_depth_aware(vol)
    elif norm_method == "local_bg":
        v_norm = normalize_local_background_subtraction(vol)
    else:
        v_norm = normalize_percentile(vol)

    # Anisotropic Gaussian smoothing
    smoothed = ndi.gaussian_filter(v_norm, sigma=sigma)

    thresh = np.percentile(smoothed, threshold_pct)

    # 3D anisotropic peak finding using fast separable rectangular footprint
    max_filt = ndi.maximum_filter(smoothed, size=(2 * radius_z + 1, 2 * radius_xy + 1, 2 * radius_xy + 1))

    peaks = (smoothed == max_filt) & (smoothed > thresh)
    peak_coords = np.argwhere(peaks)

    results = []
    for c in peak_coords:
        conf = float(smoothed[c[0], c[1], c[2]])
        results.append((int(c[0]), int(c[1]), int(c[2]), conf))
    return results


def detect_anisotropic_dog(
    vol: np.ndarray,
    norm_method: str = "percentile",
    sigma1: tuple = (1.0, 2.0, 2.0),
    sigma2: tuple = (2.0, 4.0, 4.0),
    radius_z: int = 2,
    radius_xy: int = 5,
    dog_threshold_pct: float = 85.0,
    intensity_threshold_pct: float = 80.0,
):
    """
    3D Anisotropic Difference-of-Gaussians (DoG) Multiscale Blob Detector.
    Matches the empirical nuclear profile (Z radius ~3 voxels, XY radius ~7 voxels).
    """
    if norm_method == "depth_aware":
        v_norm = normalize_depth_aware(vol)
    elif norm_method == "local_bg":
        v_norm = normalize_local_background_subtraction(vol)
    else:
        v_norm = normalize_percentile(vol)

    g1 = ndi.gaussian_filter(v_norm, sigma=sigma1)
    g2 = ndi.gaussian_filter(v_norm, sigma=sigma2)
    dog = g1 - g2

    dog_thresh = np.percentile(dog, dog_threshold_pct)
    int_thresh = np.percentile(v_norm, intensity_threshold_pct)

    max_filt = ndi.maximum_filter(dog, size=(2 * radius_z + 1, 2 * radius_xy + 1, 2 * radius_xy + 1))

    peaks = (dog == max_filt) & (dog > dog_thresh) & (v_norm > int_thresh)
    coords = np.argwhere(peaks)

    results = []
    for c in coords:
        conf = float(dog[c[0], c[1], c[2]])
        results.append((int(c[0]), int(c[1]), int(c[2]), conf))
    return results


def detect_watershed_distance(
    vol: np.ndarray,
    norm_method: str = "percentile",
    threshold_pct: float = 90.0,
    min_seed_dist_um: float = 4.0,
    min_seed_height: float = 1.0,
):
    """
    3D Anisotropic Distance Transform + Seeded Watershed Splitting:
      1. Binarizes foreground using adaptive/percentile threshold.
      2. Computes anisotropic physical Euclidean distance transform (using true voxel scale).
      3. Identifies local peaks of the distance transform as cell seed centers.
      4. Runs watershed to separate touching nuclei.
      5. Returns region centroids.
    """
    if norm_method == "depth_aware":
        v_norm = normalize_depth_aware(vol)
    else:
        v_norm = normalize_percentile(vol)

    smoothed = ndi.gaussian_filter(v_norm, sigma=(1.0, 1.5, 1.5))
    thresh = np.percentile(smoothed, threshold_pct)
    binary = smoothed > thresh

    if not np.any(binary):
        return []

    # Exact physical Euclidean distance transform
    dist_um = ndi.distance_transform_edt(binary, sampling=tuple(SCALE))

    # Seed finding: peaks of distance transform separated by min_seed_dist_um
    rz = max(1, int(round(min_seed_dist_um / SCALE[0])))
    rxy = max(2, int(round(min_seed_dist_um / SCALE[1])))

    dist_max = ndi.maximum_filter(dist_um, size=(2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1))
    seeds = (dist_um == dist_max) & (dist_um >= min_seed_height)

    markers, num_markers = ndi.label(seeds)
    if num_markers == 0:
        markers, num_markers = ndi.label(binary)
        if num_markers == 0:
            return []

    # Run watershed on negative distance map
    labels = watershed(-dist_um, markers=markers, mask=binary)

    centroids = ndi.center_of_mass(binary, labels, range(1, num_markers + 1))

    results = []
    for c in centroids:
        if not (math.isnan(c[0]) or math.isnan(c[1]) or math.isnan(c[2])):
            zc = int(round(c[0]))
            yc = int(round(c[1]))
            xc = int(round(c[2]))
            conf = float(v_norm[zc, yc, xc])
            results.append((zc, yc, xc, conf))
    return results


def deduplicate_detections(detections, min_dist_um=4.0):
    """
    Deduplicates a candidate list of detections (z, y, x, confidence) by physical distance.
    Prioritizes higher confidence detections.
    """
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda d: d[3], reverse=True)
    kept = []
    kept_coords = []

    for d in sorted_dets:
        phys_pos = np.array([d[0] * SCALE[0], d[1] * SCALE[1], d[2] * SCALE[2]])
        if not kept_coords:
            kept.append(d)
            kept_coords.append(phys_pos)
            continue

        dists = np.linalg.norm(np.array(kept_coords) - phys_pos, axis=1)
        if np.all(dists >= min_dist_um):
            kept.append(d)
            kept_coords.append(phys_pos)

    return kept


def detect_hybrid_dog_localmax(vol: np.ndarray, dog_threshold_pct: float = 88.0, threshold_pct: float = 92.0, **kwargs):
    """
    Ensemble combination: Unions candidates from Anisotropic DoG and Depth-Aware Local Maxima,
    then deduplicates with physical proximity filtering.
    """
    dog_dets = detect_anisotropic_dog(vol, norm_method="depth_aware", dog_threshold_pct=dog_threshold_pct)
    lmax_dets = detect_local_maxima(vol, norm_method="depth_aware", threshold_pct=threshold_pct)

    combined = dog_dets + lmax_dets
    return deduplicate_detections(combined, min_dist_um=4.0)
