#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kh Tohidul Islam; Monash Biomedical Imaging, Monash University, Clayton, Australia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resample 3D NIfTI Volumes to Isotropic Voxels + Robust Normalization
--------------------------------------------------------------------

This script batch-processes 3D NIfTI volumes:
  1) Resamples to isotropic voxel size (default: 1 mm)
  2) Optionally aligns voxel axes to a near-identity orientation (voxel->RAS)
  3) Optionally pads the volume to a user-defined multiple (e.g., 32)
  4) Robustly normalizes intensities via [p_low, p_high] percentiles
  5) Saves to output directory with preserved/updated affine

Designed for ultra-low-field (64 mT) and high-field MRI alike.
Safe for public release: clear CLI, logging, and validation.

Requirements:
    - Python 3.8+
    - nibabel
    - numpy
    - scikit-image
    - scipy
    - munkres (for Hungarian assignment)
    - (optional) tqdm

Example:
    python 03_resample_to_isotropic.py \
        --input_dir /path/to/nifti_in \
        --output_dir /path/to/nifti_out \
        --iso 1.0 \
        --pad_multiple 32 \
        --align_identity \
        --dtype uint8 \
        --p_low 0.5 --p_high 99.5
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

import nibabel as nib
import numpy as np

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

from preprocess_utils import (
    resample_volume_iso,
    align_with_identity_vox2ras,
    robust_percentile_normalize,
    pad_to_multiple,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def process_one(
    in_path: str,
    out_path: str,
    iso: float = 1.0,
    align_identity: bool = False,
    pad_multiple_val: Optional[int] = None,
    p_low: float = 0.5,
    p_high: float = 99.5,
    out_dtype: str = "uint8",
):
    """Process a single NIfTI file with resampling, (optional) alignment/padding, and normalization."""
    img = nib.load(in_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    zooms = img.header.get_zooms()[:3]

    # 1) Resample to isotropic
    resampled, new_affine = resample_volume_iso(data, affine, zooms, iso=(iso, iso, iso))

    # 2) Optional alignment (voxel axes -> near-identity orientation)
    if align_identity:
        resampled, new_affine = align_with_identity_vox2ras(resampled, new_affine)

    # 3) Optional padding
    if pad_multiple_val is not None and pad_multiple_val > 1:
        resampled = pad_to_multiple(resampled, multiple=pad_multiple_val)

    # 4) Robust intensity normalization
    resampled = robust_percentile_normalize(resampled, p_low=p_low, p_high=p_high)

    # 5) Cast to output dtype
    if out_dtype.lower() in ("uint8", "u1"):
        resampled = (resampled * 255.0).round().astype(np.uint8)
    elif out_dtype.lower() in ("uint16", "u2"):
        resampled = (resampled * 65535.0).round().astype(np.uint16)
    elif out_dtype.lower() in ("float32", "f4"):
        resampled = resampled.astype(np.float32)
    else:
        raise ValueError(f"Unsupported --dtype: {out_dtype}. Choose uint8 | uint16 | float32.")

    # Save NIfTI
    out_img = nib.Nifti1Image(resampled, new_affine)
    # Preserve reasonable header meta where possible
    out_header = out_img.header
    out_header.set_xyzt_units("mm")
    nib.save(out_img, out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Resample NIfTI volumes to isotropic voxels with optional alignment, padding, and robust normalization."
    )
    ap.add_argument("--input_dir", required=True, type=str, help="Folder containing input .nii or .nii.gz files.")
    ap.add_argument("--output_dir", required=True, type=str, help="Folder to write processed NIfTI files.")
    ap.add_argument("--iso", default=1.0, type=float, help="Target isotropic voxel size in mm (default: 1.0).")
    ap.add_argument("--align_identity", action="store_true", help="Align voxel axes to near-identity voxel->RAS.")
    ap.add_argument("--pad_multiple", type=int, default=None, help="Pad to a multiple (e.g., 32). If omitted, no padding.")
    ap.add_argument("--p_low", type=float, default=0.5, help="Lower percentile for robust normalization (default: 0.5).")
    ap.add_argument("--p_high", type=float, default=99.5, help="Upper percentile for robust normalization (default: 99.5).")
    ap.add_argument("--dtype", type=str, default="uint8", choices=["uint8", "uint16", "float32"], help="Output data type.")
    ap.add_argument("--suffix", type=str, default="_iso", help="Filename suffix before extension (default: _iso).")

    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        log(f"ERROR: --input_dir does not exist: {args.input_dir}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect NIfTI files
    candidates = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".nii", ".nii.gz")) and not f.startswith(".")
    ]
    if not candidates:
        log("No NIfTI files found in input directory.")
        sys.exit(0)

    log(f"Found {len(candidates)} file(s). Writing to: {args.output_dir}")
    iterator = tqdm(candidates, desc="Resampling") if TQDM else candidates

    for fname in iterator:
        in_path = os.path.join(args.input_dir, fname)
        stem = fname[:-7] if fname.lower().endswith(".nii.gz") else fname[:-4]
        out_fname = f"{stem}{args.suffix}.nii.gz"
        out_path = os.path.join(args.output_dir, out_fname)

        try:
            process_one(
                in_path=in_path,
                out_path=out_path,
                iso=args.iso,
                align_identity=args.align_identity,
                pad_multiple_val=args.pad_multiple,
                p_low=args.p_low,
                p_high=args.p_high,
                out_dtype=args.dtype,
            )
            log(f"✅ Saved: {out_path}")
        except Exception as e:
            log(f"❌ Failed: {in_path} -> {e}")

    log("Done.")


if __name__ == "__main__":
    main()
