#!/usr/bin/env python3
"""
Biohub Classical Detector V2: Sub-Voxel Centroid Refinement.
Provides sub-voxel centroid refinement algorithms for detected local maxima:
  1. Integer (identity / baseline reference)
  2. 1D Separable Parabolic / Quadratic Peak Fit
  3. 3D Full Hessian Taylor Quadratic Peak Fit
  4. Anisotropic Intensity-Weighted Center of Mass (CoM)
  5. Anisotropic Gaussian-Weighted Centroid
"""

import math
import numpy as np


def refine_integer(vol: np.ndarray, peaks: list, **kwargs) -> list:
    """Returns detections with unmodified integer coordinates."""
    return [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in peaks]


def refine_parabolic_1d(vol: np.ndarray, peaks: list, max_shift: float = 0.5, **kwargs) -> list:
    """
    Separable 1D quadratic/parabolic sub-voxel peak fitting along Z, Y, X.
    For each axis:
      delta = (f_{-1} - f_{+1}) / (2 * (f_{-1} - 2*f_0 + f_{+1}))
    Clipped to [-max_shift, max_shift].
    """
    D, H, W = vol.shape
    refined = []

    for z, y, x, conf in peaks:
        z0, y0, x0 = int(z), int(y), int(x)

        # Z shift
        dz = 0.0
        if 0 < z0 < D - 1:
            fz_m = float(vol[z0 - 1, y0, x0])
            fz_0 = float(vol[z0, y0, x0])
            fz_p = float(vol[z0 + 1, y0, x0])
            denom_z = 2.0 * (fz_m - 2.0 * fz_0 + fz_p)
            if denom_z < -1e-7:
                dz = (fz_m - fz_p) / denom_z
                dz = max(-max_shift, min(max_shift, dz))

        # Y shift
        dy = 0.0
        if 0 < y0 < H - 1:
            fy_m = float(vol[z0, y0 - 1, x0])
            fy_0 = float(vol[z0, y0, x0])
            fy_p = float(vol[z0, y0 + 1, x0])
            denom_y = 2.0 * (fy_m - 2.0 * fy_0 + fy_p)
            if denom_y < -1e-7:
                dy = (fy_m - fy_p) / denom_y
                dy = max(-max_shift, min(max_shift, dy))

        # X shift
        dx = 0.0
        if 0 < x0 < W - 1:
            fx_m = float(vol[z0, y0, x0 - 1])
            fx_0 = float(vol[z0, y0, x0])
            fx_p = float(vol[z0, y0, x0 + 1])
            denom_x = 2.0 * (fx_m - 2.0 * fx_0 + fx_p)
            if denom_x < -1e-7:
                dx = (fx_m - fx_p) / denom_x
                dx = max(-max_shift, min(max_shift, dx))

        refined.append((z0 + dz, y0 + dy, x0 + dx, conf))

    return refined


def refine_quadratic_3d(vol: np.ndarray, peaks: list, max_shift: float = 0.5, **kwargs) -> list:
    """
    Full 3D Taylor quadratic expansion:
      f(x) = f0 + g^T dx + 0.5 dx^T H dx
      dx = - H^{-1} g
    """
    dim_z, dim_y, dim_x = vol.shape
    refined = []

    for z, y, x, conf in peaks:
        z0, y0, x0 = int(z), int(y), int(x)

        if not (0 < z0 < dim_z - 1 and 0 < y0 < dim_y - 1 and 0 < x0 < dim_x - 1):
            refined.append((float(z0), float(y0), float(x0), conf))
            continue

        f0 = float(vol[z0, y0, x0])
        f_zp = float(vol[z0 + 1, y0, x0])
        f_zm = float(vol[z0 - 1, y0, x0])
        f_yp = float(vol[z0, y0 + 1, x0])
        f_ym = float(vol[z0, y0 - 1, x0])
        f_xp = float(vol[z0, y0, x0 + 1])
        f_xm = float(vol[z0, y0, x0 - 1])

        # Gradient
        g = np.array([
            0.5 * (f_zp - f_zm),
            0.5 * (f_yp - f_ym),
            0.5 * (f_xp - f_xm)
        ], dtype=np.float64)

        # Hessian diagonal
        Hzz = f_zp - 2.0 * f0 + f_zm
        Hyy = f_yp - 2.0 * f0 + f_ym
        Hxx = f_xp - 2.0 * f0 + f_xm

        # Cross terms
        Hzy = 0.25 * (float(vol[z0+1, y0+1, x0]) - float(vol[z0+1, y0-1, x0]) -
                      float(vol[z0-1, y0+1, x0]) + float(vol[z0-1, y0-1, x0]))
        Hzx = 0.25 * (float(vol[z0+1, y0, x0+1]) - float(vol[z0+1, y0, x0-1]) -
                      float(vol[z0-1, y0, x0+1]) + float(vol[z0-1, y0, x0-1]))
        Hyx = 0.25 * (float(vol[z0, y0+1, x0+1]) - float(vol[z0, y0+1, x0-1]) -
                      float(vol[z0, y0-1, x0+1]) + float(vol[z0, y0-1, x0-1]))

        H_mat = np.array([
            [Hzz, Hzy, Hzx],
            [Hzy, Hyy, Hyx],
            [Hzx, Hyx, Hxx]
        ], dtype=np.float64)

        try:
            # Check negative definiteness (all eigenvalues < 0 for local maximum)
            eigvals = np.linalg.eigvalsh(H_mat)
            if np.all(eigvals < -1e-6):
                shift = -np.linalg.solve(H_mat, g)
                if np.all(np.abs(shift) <= 1.0):
                    dz = float(np.clip(shift[0], -max_shift, max_shift))
                    dy = float(np.clip(shift[1], -max_shift, max_shift))
                    dx = float(np.clip(shift[2], -max_shift, max_shift))
                    refined.append((z0 + dz, y0 + dy, x0 + dx, conf))
                    continue
        except Exception:
            pass

        # Fallback to 1D separable if full 3D solve is ill-conditioned
        dz = 0.0
        if Hzz < -1e-7:
            dz = float(np.clip((f_zm - f_zp) / (2.0 * Hzz), -max_shift, max_shift))
        dy = 0.0
        if Hyy < -1e-7:
            dy = float(np.clip((f_ym - f_yp) / (2.0 * Hyy), -max_shift, max_shift))
        dx = 0.0
        if Hxx < -1e-7:
            dx = float(np.clip((f_xm - f_xp) / (2.0 * Hxx), -max_shift, max_shift))
        refined.append((z0 + dz, y0 + dy, x0 + dx, conf))

    return refined


def refine_com(
    vol: np.ndarray,
    peaks: list,
    radius_z: int = 1,
    radius_xy: int = 3,
    bg_subtract: bool = True,
    **kwargs
) -> list:
    """
    Anisotropic 3D intensity-weighted Center of Mass (CoM).
    Neighborhood: ellipsoidal window with radii (radius_z, radius_xy, radius_xy).
    Weights: w_i = max(0, vol[i] - bg_local).
    """
    D, H, W = vol.shape
    refined = []

    # Precompute local grid offsets within ellipsoid
    z_grid, y_grid, x_grid = np.ogrid[-radius_z:radius_z + 1, -radius_xy:radius_xy + 1, -radius_xy:radius_xy + 1]
    ellipsoid_mask = ((z_grid / max(1, radius_z))**2 +
                      (y_grid / max(1, radius_xy))**2 +
                      (x_grid / max(1, radius_xy))**2) <= 1.0
    coords_z, coords_y, coords_x = np.nonzero(ellipsoid_mask)
    rel_z = coords_z - radius_z
    rel_y = coords_y - radius_xy
    rel_x = coords_x - radius_xy

    for z, y, x, conf in peaks:
        z0, y0, x0 = int(z), int(y), int(x)

        # Check boundary
        if (z0 - radius_z < 0 or z0 + radius_z >= D or
            y0 - radius_xy < 0 or y0 + radius_xy >= H or
            x0 - radius_xy < 0 or x0 + radius_xy >= W):
            refined.append((float(z0), float(y0), float(x0), conf))
            continue

        patch = vol[z0 - radius_z:z0 + radius_z + 1,
                    y0 - radius_xy:y0 + radius_xy + 1,
                    x0 - radius_xy:x0 + radius_xy + 1]

        vals = patch[coords_z, coords_y, coords_x].astype(np.float64)

        if bg_subtract:
            bg = float(np.min(vals))
            weights = np.maximum(0.0, vals - bg)
        else:
            weights = np.maximum(0.0, vals)

        w_sum = float(np.sum(weights))
        if w_sum > 1e-8:
            dz = float(np.sum(weights * rel_z) / w_sum)
            dy = float(np.sum(weights * rel_y) / w_sum)
            dx = float(np.sum(weights * rel_x) / w_sum)
            # Clip shift to within neighborhood
            dz = float(np.clip(dz, -float(radius_z), float(radius_z)))
            dy = float(np.clip(dy, -float(radius_xy), float(radius_xy)))
            dx = float(np.clip(dx, -float(radius_xy), float(radius_xy)))
            refined.append((z0 + dz, y0 + dy, x0 + dx, conf))
        else:
            refined.append((float(z0), float(y0), float(x0), conf))

    return refined


def refine_gaussian_weighted(
    vol: np.ndarray,
    peaks: list,
    sigma_z: float = 1.2,
    sigma_xy: float = 2.5,
    radius_z: int = 2,
    radius_xy: int = 4,
    **kwargs
) -> list:
    """
    Anisotropic Gaussian-weighted Centroid.
    Combines local intensity with a Gaussian spatial prior centered at the integer peak:
      w_i = (vol[i] - min(vol)) * exp(-0.5 * [(dz/sz)^2 + (dy/sxy)^2 + (dx/sxy)^2])
    """
    D, H, W = vol.shape
    refined = []

    # Precompute local grid offsets and spatial Gaussian weights
    z_grid, y_grid, x_grid = np.ogrid[-radius_z:radius_z + 1, -radius_xy:radius_xy + 1, -radius_xy:radius_xy + 1]
    dist_sq = (z_grid / sigma_z)**2 + (y_grid / sigma_xy)**2 + (x_grid / sigma_xy)**2
    g_weights = np.exp(-0.5 * dist_sq)

    rel_z = z_grid.reshape(-1, 1, 1) * np.ones_like(g_weights)
    rel_y = y_grid.reshape(1, -1, 1) * np.ones_like(g_weights)
    rel_x = x_grid.reshape(1, 1, -1) * np.ones_like(g_weights)

    for z, y, x, conf in peaks:
        z0, y0, x0 = int(z), int(y), int(x)

        if (z0 - radius_z < 0 or z0 + radius_z >= D or
            y0 - radius_xy < 0 or y0 + radius_xy >= H or
            x0 - radius_xy < 0 or x0 + radius_xy >= W):
            refined.append((float(z0), float(y0), float(x0), conf))
            continue

        patch = vol[z0 - radius_z:z0 + radius_z + 1,
                    y0 - radius_xy:y0 + radius_xy + 1,
                    x0 - radius_xy:x0 + radius_xy + 1].astype(np.float64)

        bg = float(np.min(patch))
        int_weights = np.maximum(0.0, patch - bg)
        comb_weights = int_weights * g_weights

        w_sum = float(np.sum(comb_weights))
        if w_sum > 1e-8:
            dz = float(np.sum(comb_weights * rel_z) / w_sum)
            dy = float(np.sum(comb_weights * rel_y) / w_sum)
            dx = float(np.sum(comb_weights * rel_x) / w_sum)
            dz = float(np.clip(dz, -float(radius_z), float(radius_z)))
            dy = float(np.clip(dy, -float(radius_xy), float(radius_xy)))
            dx = float(np.clip(dx, -float(radius_xy), float(radius_xy)))
            refined.append((z0 + dz, y0 + dy, x0 + dx, conf))
        else:
            refined.append((float(z0), float(y0), float(x0), conf))

    return refined


def compute_refinement_shifts(raw_peaks: list, refined_peaks: list, scale=(1.625, 0.40625, 0.40625)) -> dict:
    """Computes mean and median coordinate shifts between integer and refined centroids."""
    if not raw_peaks or not refined_peaks:
        return {"mean_dz": 0.0, "mean_dy": 0.0, "mean_dx": 0.0, "mean_dist_um": 0.0}

    raw_arr = np.array([[p[0], p[1], p[2]] for p in raw_peaks], dtype=np.float64)
    ref_arr = np.array([[p[0], p[1], p[2]] for p in refined_peaks], dtype=np.float64)

    diff = ref_arr - raw_arr  # (N, 3) in voxels
    diff_phys = diff * np.array(scale, dtype=np.float64)  # (N, 3) in µm
    dists_phys = np.linalg.norm(diff_phys, axis=1)

    return {
        "mean_dz_vox": float(np.mean(np.abs(diff[:, 0]))),
        "mean_dy_vox": float(np.mean(np.abs(diff[:, 1]))),
        "mean_dx_vox": float(np.mean(np.abs(diff[:, 2]))),
        "mean_dist_um": float(np.mean(dists_phys)),
        "median_dist_um": float(np.median(dists_phys)),
        "max_dist_um": float(np.max(dists_phys)),
    }
