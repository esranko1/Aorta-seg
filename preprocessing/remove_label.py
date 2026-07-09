"""
Remove a label from all .nii.gz files in labelsTr and overwrite in place.

Usage:
    python remove_label.py --labels_dir /path/to/labelsTr --label 2
"""

import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path


def remove_label(file: Path, label: int) -> None:
    img = sitk.ReadImage(str(file))
    arr = sitk.GetArrayFromImage(img)
    arr[arr == label] = 0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", required=True, type=Path)
    parser.add_argument("--label", required=True, type=int)
    args = parser.parse_args()

    files = sorted(args.labels_dir.glob("*.nii.gz"))
    print(f"Found {len(files)} files. Removing label {args.label}...")

    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f.name}")
        remove_label(f, args.label)

    print("Done.")


if __name__ == "__main__":
    main()
