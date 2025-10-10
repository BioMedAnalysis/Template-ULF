#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kh Tohidul Islam; Monash Biomedical Imaging, Monash University, Clayton, Australia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Nonlinear Template Building with ANTs (SyN Quick)
-----------------------------------------------------------
This script performs one *nonlinear* refinement iteration of a population template:
  1) Registers all subject images from a previous iteration to the previous iteration's template
  2) Warps all subjects to that template
  3) Averages the warped subjects to produce the next iteration's template

It is agnostic to modality (T1, T2, FLAIR, etc.) and iteration count.
You can loop this script externally for multiple iterations.

Requirements:
  - antspyx   https://pypi.org/project/antspyx/
  - Python 3.8+
  - (Optional) tqdm for nicer progress (not required here to reduce deps)

Typical pipeline order:
  01_convert_dicom_to_nifti.py
  02_bias_field_correction_fast.py
  03_skull_stripping_bet.py
  04_ants_linear_registration_template.py
  05_ants_nonlinear_template_iteration.py   ← this script (run per iteration)

Example usage:
  python 05_ants_nonlinear_template_iteration.py \
      --base_dir /path/to/linear_registered \
      --prev_iter_label T2_NL4 \
      --prev_iter_index 4 \
      --next_iter_label T2_NL5 \
      --next_iter_index 5 \
      --template_name template_nonlinear_iter4.nii.gz \
      --moving_suffix "_T2_nonlinear.nii.gz" \
      --out_suffix "_T2_nonlinear.nii.gz"

For T1 (only names change):
  python 05_ants_nonlinear_template_iteration.py \
      --base_dir /data/T1/linear_registered \
      --prev_iter_label T1_NL2 \
      --prev_iter_index 2 \
      --next_iter_label T1_NL3 \
      --next_iter_index 3 \
      --template_name template_nonlinear_iter2.nii.gz \
      --moving_suffix "_T1_nonlinear.nii.gz" \
      --out_suffix "_T1_nonlinear.nii.gz"

Notes:
  - The script searches for files in {base_dir}/{prev_iter_label} that end with {moving_suffix}
  - Outputs are written to {base_dir}/{next_iter_label}
  - The new template is saved as "template_nonlinear_iter{next_iter_index}.nii.gz"
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import ants


def parse_args():
    p = argparse.ArgumentParser(
        description="Run one nonlinear SyN (antsRegistrationSyNQuick[s]) iteration and build the next template."
    )
    p.add_argument(
        "--base_dir", required=True, type=str,
        help="Base directory containing iteration folders (e.g., linear_registered/)."
    )
    p.add_argument(
        "--prev_iter_label", required=True, type=str,
        help="Folder name of the previous nonlinear iteration (e.g., T2_NL4)."
    )
    p.add_argument(
        "--next_iter_label", required=True, type=str,
        help="Folder name to create for this iteration's outputs (e.g., T2_NL5)."
    )
    p.add_argument(
        "--prev_iter_index", required=True, type=int,
        help="Previous iteration index (e.g., 4). Used to check template name consistency."
    )
    p.add_argument(
        "--next_iter_index", required=True, type=int,
        help="Next iteration index (e.g., 5). Used to name the new template."
    )
    p.add_argument(
        "--template_name", required=True, type=str,
        help="Filename of the previous iteration template within prev_iter_label "
             "(e.g., template_nonlinear_iter4.nii.gz)."
    )
    p.add_argument(
        "--moving_suffix", default="_T2_nonlinear.nii.gz", type=str,
        help="Suffix pattern to select moving images in prev_iter folder (default: _T2_nonlinear.nii.gz). "
             "Use _T1_nonlinear.nii.gz for T1."
    )
    p.add_argument(
        "--out_suffix", default="_T2_nonlinear.nii.gz", type=str,
        help="Suffix to append to warped outputs in next_iter folder (default: _T2_nonlinear.nii.gz)."
    )
    p.add_argument(
        "--transform", default='antsRegistrationSyNQuick[s]', type=str,
        help='ANTs transform type (default: "antsRegistrationSyNQuick[s]").'
    )
    p.add_argument(
        "--save_transforms", action="store_true",
        help="If set, also save forward/inverse transforms (useful for debugging/resampling)."
    )
    p.add_argument(
        "--threads", type=int, default=0,
        help="Number of threads for ANTs. 0 uses ANTs default."
    )
    return p.parse_args()


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def main():
    args = parse_args()

    # Set threads if requested
    if args.threads and args.threads > 0:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
        log(f"Set ITK threads to {args.threads}")

    prev_dir = os.path.join(args.base_dir, args.prev_iter_label)
    next_dir = os.path.join(args.base_dir, args.next_iter_label)
    os.makedirs(next_dir, exist_ok=True)

    # Validate template path
    template_path = os.path.join(prev_dir, args.template_name)
    if not os.path.isfile(template_path):
        log(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    # Load template
    log(f"Loading previous iteration template: {template_path}")
    template = ants.image_read(template_path)

    # Gather moving images from prev iteration
    pattern = os.path.join(prev_dir, f"*{args.moving_suffix}")
    moving_paths = sorted(glob.glob(pattern))
    if not moving_paths:
        log(f"ERROR: No moving images found with pattern: {pattern}")
        sys.exit(1)
    log(f"Found {len(moving_paths)} moving images.")

    # Nonlinear registration loop
    warped_images = []
    for moving_path in moving_paths:
        basename = os.path.basename(moving_path)
        subject_id = basename.replace(args.moving_suffix, "")

        log(f"Registering subject: {subject_id}")
        moving_img = ants.image_read(moving_path)

        reg = ants.registration(
            fixed=template,
            moving=moving_img,
            type_of_transform=args.transform,
            verbose=False
        )

        # Save warped image
        warped = reg["warpedmovout"]
        out_img_path = os.path.join(next_dir, f"{subject_id}{args.out_suffix}")
        warped.to_filename(out_img_path)
        warped_images.append(warped)
        log(f"  -> Saved warped image: {out_img_path}")

        # Optionally save transforms
        if args.save_transforms:
            fwd_list = reg.get("fwdtransforms", [])
            inv_list = reg.get("invtransforms", [])
            # Copy/rename to next_dir for clarity
            for i, tf in enumerate(fwd_list):
                tf_out = os.path.join(next_dir, f"{subject_id}_fwd_{i}.h5" if tf.endswith(".h5") else
                                      os.path.join(next_dir, f"{subject_id}_fwd_{i}.nii.gz"))
                ants.write_transform(reg["fwdtransforms"][i], tf_out)
                log(f"  -> Saved forward transform: {tf_out}")
            for i, tf in enumerate(inv_list):
                tf_out = os.path.join(next_dir, f"{subject_id}_inv_{i}.h5" if tf.endswith(".h5") else
                                      os.path.join(next_dir, f"{subject_id}_inv_{i}.nii.gz"))
                ants.write_transform(reg["invtransforms"][i], tf_out)
                log(f"  -> Saved inverse transform: {tf_out}")

    # Average warped images into new template
    log("Averaging warped images to create next iteration template...")
    next_template = ants.average_images(warped_images)

    next_template_name = f"template_nonlinear_iter{args.next_iter_index}.nii.gz"
    next_template_path = os.path.join(next_dir, next_template_name)
    next_template.to_filename(next_template_path)

    log("Nonlinear iteration complete.")
    log(f"New template saved: {next_template_path}")
    log(f"Outputs directory: {next_dir}")


if __name__ == "__main__":
    main()
