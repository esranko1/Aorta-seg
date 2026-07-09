"""
Restructure per-patient folders into a flat nnU-Net inference input folder.

Input layout:
    root/
      PT008_M/<some_file>.nii.gz
      PT011_M_F4/<some_file>.nii.gz
      ...

Output layout (nnU-Net inference format):
    out/
      PT008_M_0000.nii.gz
      PT011_M_F4_0000.nii.gz
      ...

Usage:
    python restructure_for_inference.py --root /path/to/patient_folders --out /path/to/nnunet_input
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='Folder containing per-patient subfolders')
    parser.add_argument('--out', type=Path, required=True, help='Output folder for flat, renamed files')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(d for d in args.root.iterdir() if d.is_dir())
    if not patient_dirs:
        sys.exit(f"No subfolders found in {args.root}")

    n_ok = 0
    for patient_dir in patient_dirs:
        nii_files = list(patient_dir.glob('*.nii.gz'))
        if len(nii_files) == 0:
            print(f"[SKIP] {patient_dir.name}: no .nii.gz file found")
            continue
        if len(nii_files) > 1:
            print(f"[SKIP] {patient_dir.name}: found {len(nii_files)} .nii.gz files, expected 1 - {[f.name for f in nii_files]}")
            continue

        src = nii_files[0]
        dst = args.out / f"{patient_dir.name}_0000.nii.gz"
        shutil.copy2(src, dst)
        print(f"{patient_dir.name}: {src.name} -> {dst.name}")
        n_ok += 1

    print(f"\n{n_ok}/{len(patient_dirs)} patients copied to {args.out}")


if __name__ == '__main__':
    main()
