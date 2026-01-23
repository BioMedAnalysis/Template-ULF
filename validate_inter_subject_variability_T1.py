import os
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# USER SETTINGS (T1)
# -----------------------------
BASE_DIR = Path("/Users/tisl0004/Downloads/T1_T2_T2/Low-Field/1mm/T1_stripped")

# NOTE: your T1 linear folder is in a different tree
LINEAR_DIR = Path("/Users/tisl0004/Downloads/T1_T2_T2/Low-Field/1mm/T1_template_build/linear_registered")

# Nonlinear iterations (also in the template_build tree)
NL_DIRS = {
    "T1_NL1": Path("/Users/tisl0004/Downloads/T1_T2_T2/Low-Field/1mm/T1_template_build/NL1"),
    "T1_NL2": Path("/Users/tisl0004/Downloads/T1_T2_T2/Low-Field/1mm/T1_template_build/NL2"),
    "T1_NL3": Path("/Users/tisl0004/Downloads/T1_T2_T2/Low-Field/1mm/T1_template_build/NL3"),
    "T1_NL4": Path("/Users/tisl0004/Downloads/T1_T2_T2/Low-Field/1mm/T1_template_build/NL4"),
    "T1_NL5": Path("/Users/tisl0004/Downloads/T1_T2_T2/Low-Field/1mm/T1_template_build/NL5"),
}

OUT_DIR = BASE_DIR / "variability_validation_T1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
P_LOW, P_HIGH = 1.0, 99.0
MEAN_MASK_PERCENTILE = 20.0
EPS = 1e-6
N_SLICES_AVG = 5

# -----------------------------
# UTILITIES
# -----------------------------
def find_nii_files(folder: Path, include_key="POCEMR", verbose=False):
    folder = Path(folder)
    if not folder.exists():
        print(f"[WARNING] Folder does not exist: {folder}")
        return []

    files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))

    if verbose:
        print("All NIfTI files:")
        for f in files:
            print("  ", f.name)

    filtered = [f for f in files if include_key in f.name]

    if verbose:
        print("\nKept files:")
        for f in filtered:
            print("  ", f.name)

    return sorted(filtered)



def load_stack(files):
    imgs = []
    ref_img = None
    ref_shape = None

    for f in files:
        img = nib.load(str(f))
        data = img.get_fdata(dtype=np.float32)

        if ref_img is None:
            ref_img = img
            ref_shape = data.shape
        else:
            if data.shape != ref_shape:
                raise ValueError(f"Shape mismatch for {f}: {data.shape} vs {ref_shape}")

        imgs.append(data)

    return np.stack(imgs, axis=-1), ref_img


def robust_norm(stack):
    out = np.zeros_like(stack, dtype=np.float32)
    for i in range(stack.shape[-1]):
        v = stack[..., i]
        lo, hi = np.percentile(v, P_LOW), np.percentile(v, P_HIGH)
        if hi > lo:
            v = np.clip(v, lo, hi)
            out[..., i] = (v - lo) / (hi - lo)
    return out


def compute_maps(stack):
    mean = np.mean(stack, axis=-1)
    sd = np.std(stack, axis=-1)
    cov = sd / (mean + EPS)
    return mean, sd, cov


def make_mask(mean):
    return mean > np.percentile(mean, MEAN_MASK_PERCENTILE)


def summarize(cov, mask):
    vals = cov[mask]
    return {
        "median_CoV": float(np.median(vals)),
        "p95_CoV": float(np.percentile(vals, 95)),
        "mean_CoV": float(np.mean(vals)),
    }


def mid_slice(vol, axis):
    mid = vol.shape[axis] // 2
    sl = slice(mid - N_SLICES_AVG // 2, mid + N_SLICES_AVG // 2 + 1)

    if axis == 0:
        img = vol[sl, :, :].mean(0)
    elif axis == 1:
        img = vol[:, sl, :].mean(1)
    else:
        img = vol[:, :, sl].mean(2)

    return np.rot90(img)


def save_fig(mean, sd, cov, out_png, title):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(title)

    for j, vol in enumerate([mean, sd, cov]):
        for i in range(3):
            axs[j, i].imshow(
                mid_slice(vol, i),
                cmap="gray" if j == 0 else "viridis"
            )
            axs[j, i].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
rows = []

print("\n=== Processing T1 LINEAR ===")
lin_files = find_nii_files(LINEAR_DIR)
print(f"[T1_LINEAR] Found {len(lin_files)} files in {LINEAR_DIR}")

if len(lin_files) == 0:
    raise FileNotFoundError(f"No NIfTI files found in LINEAR_DIR: {LINEAR_DIR}")

stack, ref = load_stack(lin_files)
stack = robust_norm(stack)
mean, sd, cov = compute_maps(stack)
mask = make_mask(mean)

rows.append({
    "iteration": "LINEAR",
    "n": len(lin_files),
    **summarize(cov, mask)
})

save_fig(mean, sd, cov, OUT_DIR / "LINEAR_mean_sd_cov.png",
         "T1 Linear Registration")

# Nonlinear iterations
for name, folder in NL_DIRS.items():
    print(f"\n=== Processing {name} ===")
    files = find_nii_files(folder)

    if len(files) == 0:
        print(f"[SKIPPED] No files found in {folder}")
        continue

    print(f"[{name}] Found {len(files)} files in {folder}")

    stack, _ = load_stack(files)
    stack = robust_norm(stack)
    mean, sd, cov = compute_maps(stack)
    mask = make_mask(mean)

    rows.append({
        "iteration": name,
        "n": len(files),
        **summarize(cov, mask)
    })

    save_fig(mean, sd, cov,
             OUT_DIR / f"{name}_mean_sd_cov.png",
             f"T1 {name}")

# Save summary
df = pd.DataFrame(rows)
csv_path = OUT_DIR / "T1_inter_subject_variability_summary.csv"
df.to_csv(csv_path, index=False)

print("\n=== DONE (T1) ===")
print(df)
print(f"\nSaved summary CSV: {csv_path}")
print(f"Saved figures to: {OUT_DIR}")
