"""
Diagnose spatial alignment between predicted segmentations and ground truth labels.
Checks if the prediction shape is correct but spatially offset from the GT.
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from scipy import ndimage

# --- Adjust these paths ---
BASE = Path(r"C:\Users\esranko1\Desktop\Research_seg\Data_Aorta_Sorted")
PRED_DIR  = BASE / "nnUNet_results/Dataset021_CTAAorta/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
GT_DIR    = BASE / "nnUNet_raw/Dataset021_CTAAorta/labelsTr"
IMAGE_DIR = BASE / "nnUNet_raw/Dataset021_CTAAorta/imagesTr"


def center_of_mass_world(mask_sitk):
    arr = sitk.GetArrayFromImage(mask_sitk)
    if arr.max() == 0:
        return None
    com_vox = ndimage.center_of_mass(arr > 0)          # (z, y, x)
    com_world = mask_sitk.TransformContinuousIndexToPhysicalPoint(
        (float(com_vox[2]), float(com_vox[1]), float(com_vox[0]))
    )
    return np.array(com_world)


def dice(pred_sitk, gt_sitk):
    p = sitk.GetArrayFromImage(pred_sitk) > 0
    g = sitk.GetArrayFromImage(gt_sitk) > 0
    return 2 * np.logical_and(p, g).sum() / (p.sum() + g.sum() + 1e-8)


def spatial_header(img_sitk):
    return {
        "origin":    np.array(img_sitk.GetOrigin()),
        "spacing":   np.array(img_sitk.GetSpacing()),
        "direction": np.array(img_sitk.GetDirection()),
        "size":      np.array(img_sitk.GetSize()),
    }


results = []

pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
if not pred_files:
    print(f"No predictions found in:\n  {PRED_DIR}")
    raise SystemExit

for pred_file in pred_files:
    case = pred_file.name.replace(".nii.gz", "")
    gt_file  = GT_DIR    / f"{case}.nii.gz"
    img_file = IMAGE_DIR / f"{case}_0000.nii.gz"

    if not gt_file.exists():
        print(f"[SKIP] No GT for {case}")
        continue

    pred = sitk.ReadImage(str(pred_file))
    gt   = sitk.ReadImage(str(gt_file))

    # --- 1. Header comparison: prediction vs GT ---
    h_pred = spatial_header(pred)
    h_gt   = spatial_header(gt)

    origin_diff  = np.linalg.norm(h_pred["origin"]  - h_gt["origin"])
    spacing_diff = np.linalg.norm(h_pred["spacing"] - h_gt["spacing"])
    dir_diff     = np.linalg.norm(h_pred["direction"] - h_gt["direction"])

    # --- 2. Header comparison: prediction vs raw image (if available) ---
    img_origin_diff = None
    if img_file.exists():
        img = sitk.ReadImage(str(img_file))
        h_img = spatial_header(img)
        img_origin_diff = np.linalg.norm(h_pred["origin"] - h_img["origin"])

    # --- 3. Center-of-mass offset in world (mm) ---
    com_pred = center_of_mass_world(pred)
    com_gt   = center_of_mass_world(gt)

    if com_pred is None or com_gt is None:
        print(f"[WARN] {case}: empty mask")
        continue

    offset_vec  = com_pred - com_gt
    offset_norm = np.linalg.norm(offset_vec)
    dsc         = dice(pred, gt)

    flag = " <-- MISMATCH" if (origin_diff > 1.0 or offset_norm > 20) else ""
    print(
        f"{case}: Dice={dsc:.3f} | CoM offset={offset_norm:6.1f}mm "
        f"(x={offset_vec[0]:+.1f} y={offset_vec[1]:+.1f} z={offset_vec[2]:+.1f}) | "
        f"origin_diff(pred-gt)={origin_diff:.2f}mm"
        + (f" | origin_diff(pred-img)={img_origin_diff:.2f}mm" if img_origin_diff is not None else "")
        + flag
    )

    results.append({
        "case":         case,
        "dice":         dsc,
        "offset_mm":    offset_norm,
        "offset_vec":   offset_vec,
        "origin_diff":  origin_diff,
    })

if not results:
    print("No results — check paths above.")
    raise SystemExit

dices    = [r["dice"]      for r in results]
offsets  = [r["offset_mm"] for r in results]
mean_vec = np.mean([r["offset_vec"] for r in results], axis=0)

print("\n--- Summary ---")
print(f"Cases analysed : {len(results)}")
print(f"Mean Dice      : {np.mean(dices):.3f}  (std {np.std(dices):.3f})")
print(f"Mean CoM offset: {np.mean(offsets):.1f} mm  (std {np.std(offsets):.1f})")
print(f"Mean offset vec: x={mean_vec[0]:+.1f}  y={mean_vec[1]:+.1f}  z={mean_vec[2]:+.1f}  mm")

if np.linalg.norm(mean_vec) > 10:
    print("\n=> SYSTEMATIC spatial offset detected — predictions are shifted relative to GT.")
    print("   Most likely cause: origin or direction mismatch in the training labels.")
else:
    print("\n=> No systematic directional offset — if Dice is low, likely shape/boundary errors.")
