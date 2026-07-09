"""
Print voxel spacing for all NIfTI files in a folder.
Usage: python check_spacing.py /path/to/imagesTr
"""

import argparse
from pathlib import Path
import SimpleITK as sitk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--z_threshold', type=float, default=2.0,
                        help='Flag scans with z-spacing above this value (default: 2.0mm)')
    args = parser.parse_args()

    files = sorted(args.folder.glob('*.nii.gz'))
    if not files:
        print(f"No .nii.gz files found in {args.folder}")
        return

    flagged = []
    for f in files:
        img = sitk.ReadImage(str(f))
        sx, sy, sz = img.GetSpacing()
        flag = ' <-- THICK SLICES' if sz > args.z_threshold else ''
        print(f"{f.name:40s}  x={sx:.3f}  y={sy:.3f}  z={sz:.3f}{flag}")
        if flag:
            flagged.append(f.name)

    print(f"\n{len(flagged)}/{len(files)} scans have z-spacing > {args.z_threshold}mm")
    if flagged:
        print("Flagged:")
        for name in flagged:
            print(f"  {name}")

if __name__ == '__main__':
    main()
