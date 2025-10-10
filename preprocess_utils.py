#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kh Tohidul Islam; Monash Biomedical Imaging, Monash University, Clayton, Australia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Utilities for MRI Volumes
---------------------------------------

Helper functions for:
  - Isotropic resampling (scikit-image)
  - Affine updates on resample
  - Voxel-axis alignment to near-identity voxel->RAS
  - Robust percentile normalization
  - Padding to a chosen multiple

These utilities are intentionally lightweight, dependency-friendly, and
documented for public reuse in neuroimaging pipelines.
"""

from typing import Tuple
import numpy as np
from skimage.transform import resize
from scipy.stats import scoreatpercentile
from munkres import Munkres


def resample_volume_iso(
    data: np.ndarray,
    affine: np.ndarray,
    voxel_size: Tuple[float, float, float],
    iso: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a 3D volume to isotropic voxels with linear interpolation.

    Parameters
    ----------
    data : np.ndarray
        3D image array (Z, Y, X) or (X, Y, Z). We assume nibabel default: (X, Y, Z).
    affine : np.ndarray
        4x4 affine mapping voxel indices to world (RAS).
    voxel_size : tuple of float
        Original voxel sizes (dx, dy, dz), typically from header.get_zooms()[:3].
    iso : tuple of float
        Target voxel size (isotropic), e.g., (1.0, 1.0, 1.0).
    order : int
        Interpolation order for skimage.transform.resize (1=linear).

    Returns
    -------
    resampled : np.ndarray
    new_affine : np.ndarray
    """
    if data.ndim != 3:
        raise ValueError("Only 3D volumes are supported.")

    scale = np.array(voxel_size, dtype=float) / np.array(iso, dtype=float)
    new_shape = np.ceil(np.array(data.shape, dtype=float) * scale).astype(int)

    # skimage expects data in (Z, Y, X) or (Y, X) convention-agnostic; we pass as-is
    resampled = resize(
        data,
        output_shape=tuple(new_shape.tolist()),
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)

    # Update affine: scale voxel-to-world directions to reflect new voxel size
    new_aff = affine.copy()
    # Scale the direction cosines so that voxel sizes become 'iso'
    # Equivalent to: current_dir * (old_voxel / new_voxel)
    # But since we changed the number of voxels (shape), we want the physical spacing encoded in the affine to be iso.
    # A simple approach: normalize columns to unit length then multiply by iso.
    dirs = new_aff[:3, :3]
    # Compute current voxel sizes from affine columns (norms)
    col_norms = np.linalg.norm(dirs, axis=0)
    # Avoid divide-by-zero
    col_norms[col_norms == 0] = 1.0
    unit_dirs = dirs / col_norms
    new_aff[:3, :3] = unit_dirs * np.array(iso)[None, :]

    # Keep translation; center shift is not strictly necessary here because we keep origin consistent
    return resampled, new_aff


def align_with_identity_vox2ras(
    data: np.ndarray,
    affine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder/flip voxel axes so that the voxel->RAS basis is close to identity.

    This does not guarantee a canonical (e.g., RAS+) orientation across toolchains,
    but moves towards axis-aligned columns with positive handedness when possible.

    Returns the reoriented data and updated affine.
    """
    if data.ndim != 3:
        raise ValueError("Only 3D volumes are supported.")

    # Cost matrix for matching voxel axes to canonical basis (I, J, K)
    cost = np.zeros((3, 3), dtype=float)
    A = affine[:3, :3]
    # Column vectors that map voxel axes to world
    cols = [A[:, i] for i in range(3)]
    canonical = np.eye(3)

    for i in range(3):          # choose voxel axis i
        for j in range(3):      # match to canonical axis j
            num = abs(np.dot(cols[i], canonical[:, j]))
            denom = (np.linalg.norm(cols[i]) * np.linalg.norm(canonical[:, j]))
            cost[i, j] = - (num / (denom if denom != 0 else 1.0))

    # Hungarian algorithm to pick the best permutation
    perm = [j for _, j in sorted(Munkres().compute(cost.tolist()))]

    # Apply permutation to data and affine
    data_perm = np.transpose(data, axes=perm)
    new_aff = affine.copy()
    new_aff[:3, :3] = affine[:3, perm]

    # Ensure positive direction along diagonals if possible, flipping when needed
    for d in range(3):
        if new_aff[d, d] < 0:
            new_aff[:3, d] *= -1.0
            data_perm = np.flip(data_perm, axis=d)

    return data_perm, new_aff


def robust_percentile_normalize(
    data: np.ndarray, p_low: float = 0.5, p_high: float = 99.5
) -> np.ndarray:
    """
    Normalize intensities to [0, 1] based on robust percentiles.

    Parameters
    ----------
    data : np.ndarray
        3D image array.
    p_low : float
        Lower percentile (inclusive).
    p_high : float
        Upper percentile (inclusive).

    Returns
    -------
    data01 : np.ndarray
        Data scaled to [0, 1].
    """
    arr = data.astype(np.float32, copy=False)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        # Fallback to min/max if pathological
        lo, hi = float(arr.min()), float(arr.max()) if arr.size else (0.0, 1.0)
        if hi == lo:
            return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)


def pad_to_multiple(data: np.ndarray, multiple: int = 32) -> np.ndarray:
    """
    Zero-pad (at the end) each dimension so that shape is a multiple of `multiple`.

    Parameters
    ----------
    data : np.ndarray
        3D volume.
    multiple : int
        Desired multiple (e.g., 32 for conv-nets).

    Returns
    -------
    padded : np.ndarray
    """
    shape = np.array(data.shape, dtype=int)
    target = np.ceil(shape / multiple).astype(int) * multiple
    padded = np.zeros(target, dtype=data.dtype)
    padded[: shape[0], : shape[1], : shape[2]] = data
    return padded
