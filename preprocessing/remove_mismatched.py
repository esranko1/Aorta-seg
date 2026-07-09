"""
Finds image-label pairs with mismatched shapes or spacing and removes them.

Usage:
    # Preview only
    python remove_mismatched.py --dataset /path/to/Dataset021_CTAAorta

    # Actually delete mismatched pairs
    python remove_mismatched.py --dataset /path/to/Dataset021_CTAAorta --delete
"""

import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path


def check_pair(img_path: Path, lbl_path: Path) -> list[str]:
    errors = []
    img = sitk.ReadImage(str(img_path))
    lbl = sitk.ReadImage(str(lbl_path))

    if img.GetSize() != lbl.GetSize():
        errors.append(f"shape mismatch: image {img.GetSize()} vs label {lbl.GetSize()}")

    img_spacing = np.round(img.GetSpacing(), 3)
    lbl_spacing = np.round(lbl.GetSpacing(), 3)
    if not np.allclose(img_spacing, lbl_spacing, rtol=0.05):
        errors.append(f"spacing mismatch: image {img_spacing} vs label {lbl_spacing}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--delete", action="store_true", help="Delete mismatched pairs")
    args = parser.parse_args()

    images_dir = args.dataset / "imagesTr"
    labels_dir = args.dataset / "labelsTr"

    mismatched = []
    ok = 0

    label_files = sorted(labels_dir.glob("*.nii.gz"))
    print(f"Checking {len(label_files)} pairs...\n")

    for lbl in label_files:
        case_id = lbl.name.replace(".nii.gz", "")
        img = images_dir / f"{case_id}_0000.nii.gz"

        if not img.exists():
            print(f"  MISSING image for {case_id}")
            mismatched.append((None, lbl))
            continue

        errors = check_pair(img, lbl)
        if errors:
            print(f"  MISMATCH {case_id}: {'; '.join(errors)}")
            mismatched.append((img, lbl))
        else:
            ok += 1

    print(f"\nOK: {ok}  |  Mismatched: {len(mismatched)}")

    if mismatched and args.delete:
        print("\nDeleting mismatched pairs...")
        for img, lbl in mismatched:
            if img and img.exists():
                img.unlink()
                print(f"  Deleted {img.name}")
            if lbl.exists():
                lbl.unlink()
                print(f"  Deleted {lbl.name}")
        print(f"\nRemoved {len(mismatched)} pairs. {ok} remain.")
        print(f"Update dataset.json num_training_cases to {ok} and regenerate.")
    elif mismatched:
        print("\nRun with --delete to remove them.")


if __name__ == "__main__":
    main()
