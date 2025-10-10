#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kh Tohidul Islam; Monash Biomedical Imaging, Monash University, Clayton, Australia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM to NIfTI Batch Conversion Script
--------------------------------------

This script recursively converts DICOM folders into NIfTI format using the
`dicom2nifti` package. It is designed for reproducible and automated preprocessing
of MRI datasets (e.g., ultra-low-field or high-field scans).

Each first-level subdirectory within the specified input directory is treated
as an individual subject/session folder. The converted NIfTI files are saved
in the same location by default.

Requirements:
    - Python 3.8+
    - dicom2nifti  (https://pypi.org/project/dicom2nifti/)

Example:
    python convert_dicom_to_nifti.py --input_dir /path/to/dicom_root

Recommended Usage in a Pipeline:
    01_convert_dicom_to_nifti.py
    02_bias_field_correction_fast.py
    03_skull_stripping_bet.py
    04_ants_linear_registration_template.py
    05_ants_nonlinear_template_iteration.py
"""

import os
import argparse
import dicom2nifti
from datetime import datetime


def log(msg: str):
    """Print formatted log messages with timestamps."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def convert_dicom_folders(input_root: str):
    """
    Convert all first-level DICOM subfolders to NIfTI format.

    Parameters
    ----------
    input_root : str
        Path to the root directory containing DICOM subfolders.
    """
    # List all non-hidden first-level subdirectories
    subfolders = [
        f for f in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, f)) and not f.startswith('.')
    ]

    if not subfolders:
        log(f"No subfolders found in {input_root}. Nothing to convert.")
        return

    log(f"Found {len(subfolders)} subfolder(s) in: {input_root}")

    for folder_name in subfolders:
        source_path = os.path.join(input_root, folder_name)
        log(f"🔄 Converting DICOM folder: {source_path}")

        try:
            dicom2nifti.convert_directory(source_path, source_path, reorient=True)
            log(f"✅ Conversion complete for: {folder_name}")
        except Exception as e:
            log(f"❌ Failed to convert {folder_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert DICOM folders to NIfTI format using dicom2nifti."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Path to the root directory containing subject DICOM folders."
    )

    args = parser.parse_args()
    input_root = args.input_dir

    if not os.path.isdir(input_root):
        log(f"ERROR: The provided path does not exist or is not a directory: {input_root}")
        return

    log("Starting DICOM to NIfTI conversion process...")
    convert_dicom_folders(input_root)
    log("All conversions completed successfully.")


if __name__ == "__main__":
    main()
