#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kh Tohidul Islam; Monash Biomedical Imaging, Monash University, Clayton, Australia
Affine (Linear) Registration and Template Creation using ANTs
-------------------------------------------------------------
This script performs affine (linear) registration of multiple 3D brain MRI volumes 
(e.g., ultra-low-field T2-weighted scans) to a fixed reference image 
(e.g., a high-field T2-weighted template or representative subject).

All registered images are saved to disk, and an average template is computed 
from the registered outputs.

Requirements:
    - antspyx  (https://pypi.org/project/antspyx/)
    - tqdm     (for progress visualization)

Example:
    python linear_registration_template.py
"""

import ants
import os
from tqdm import tqdm

# =============================================================================
# User-defined paths
# =============================================================================

# Directory containing subject MRI images to be registered (e.g., ULF scans)
ulf_t2_dir = "/path/to/your/ULF_T2_images"

# Path to the fixed (reference) image 
# Typically, this is a representative high-field or age-matched T2-weighted image
fixed_image_path = "/path/to/your/fixed_reference_image.nii.gz"

# Output directory for registered images and template
output_dir = os.path.join(ulf_t2_dir, "linear_registered")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Load fixed (reference) image
# =============================================================================
print(f"Loading fixed reference image:\n{fixed_image_path}")
fixed = ants.image_read(fixed_image_path)

# =============================================================================
# List all moving images (to be registered)
# =============================================================================
moving_files = [f for f in os.listdir(ulf_t2_dir) if f.endswith(".nii.gz")]
print(f"Found {len(moving_files)} images for registration.")

# Storage for registered images (to compute average template)
registered_images = []

# =============================================================================
# Perform affine (linear) registration for each subject
# =============================================================================
for fname in tqdm(moving_files, desc="Performing Linear Registration"):
    moving_path = os.path.join(ulf_t2_dir, fname)
    moving = ants.image_read(moving_path)

    # Perform affine registration using ANTs
    # 'AffineFast' is an efficient 12-parameter model with mutual information optimization
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="AffineFast",
        verbose=False
    )

    # Save the registered output
    subject_id = fname.replace(".nii.gz", "")
    out_path = os.path.join(output_dir, f"{subject_id}_linear.nii.gz")
    reg['warpedmovout'].to_filename(out_path)

    # Store registered image for averaging
    registered_images.append(reg['warpedmovout'])

# =============================================================================
# Create an average (mean) image from all registered scans
# =============================================================================
print("\nCreating group-average template using ants.average_images...")
avg_img = ants.average_images(registered_images)

# Save the resulting average template
avg_path = os.path.join(output_dir, "T2_linear_template.nii.gz")
avg_img.to_filename(avg_path)

print(f"\nLinear registration completed successfully.")
print(f"Average template saved at:\n{avg_path}")

# =============================================================================
# Notes:
# =============================================================================
# - This script performs only linear (affine) registration. 
#   For nonlinear refinement, use ANTs SyN registration (e.g., 'antsRegistrationSyNQuick[s]').
# - All affine transformations can be reused for subsequent nonlinear steps or inverse mapping.
# - Ensure that input images are preprocessed (e.g., bias field corrected and skull-stripped).
