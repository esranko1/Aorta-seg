"""
Resamples thick-slice scans to a target spacing using B-spline interpolation.
Good scans (z-spacing <= z_threshold) are copied as-is.
Labels are resampled with nearest-neighbour to preserve integer values.

Usage:
    python resample_dataset.py \
        --images C:/path/to/imagesTr \
        --labels C:/path/to/labelsTr \
        --out_images C:/path/to/output/imagesTr \
        --out_labels C:/path/to/output/labelsTr \
        --target_spacing 0.5 0.5 1.0 \
        --z_threshold 2.0
"""

import argparse
import shutil
from pathlib import Path

import SimpleITK as sitk


def resample_image(image: sitk.Image, target_spacing: tuple, is_label: bool) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True, help='Input imagesTr folder')
    parser.add_argument('--labels', type=Path, required=True, help='Input labelsTr folder')
    parser.add_argument('--out_images', type=Path, required=True, help='Output imagesTr folder')
    parser.add_argument('--out_labels', type=Path, required=True, help='Output labelsTr folder')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[0.5, 0.5, 1.0],
                        help='Target x y z spacing in mm (default: 0.5 0.5 1.0)')
    parser.add_argument('--z_threshold', type=float, default=2.0,
                        help='Resample scans with z-spacing above this value (default: 2.0)')
    args = parser.parse_args()

    args.out_images.mkdir(parents=True, exist_ok=True)
    args.out_labels.mkdir(parents=True, exist_ok=True)

    target_spacing = tuple(args.target_spacing)
    image_files = sorted(args.images.glob('*.nii.gz'))

    if not image_files:
        print(f"No .nii.gz files found in {args.images}")
        return

    resampled = 0
    copied = 0

    for img_path in image_files:
        # Derive matching label filename (remove _0000 suffix)
        label_name = img_path.name.replace('_0000.nii.gz', '.nii.gz')
        label_path = args.labels / label_name

        out_img_path = args.out_images / img_path.name
        out_lbl_path = args.out_labels / label_name

        img = sitk.ReadImage(str(img_path))
        z_spacing = img.GetSpacing()[2]

        if z_spacing > args.z_threshold:
            print(f"[RESAMPLE] {img_path.name}  z={z_spacing:.2f}mm → {target_spacing[2]:.2f}mm")
            resampled_img = resample_image(img, target_spacing, is_label=False)
            sitk.WriteImage(resampled_img, str(out_img_path))

            if label_path.exists():
                lbl = sitk.ReadImage(str(label_path))
                resampled_lbl = resample_image(lbl, target_spacing, is_label=True)
                sitk.WriteImage(resampled_lbl, str(out_lbl_path))
            else:
                print(f"  WARNING: no label found for {img_path.name}, skipping label")

            resampled += 1
        else:
            print(f"[COPY]     {img_path.name}  z={z_spacing:.2f}mm")
            shutil.copy2(img_path, out_img_path)
            if label_path.exists():
                shutil.copy2(label_path, out_lbl_path)
            copied += 1

    print(f"\nDone: {resampled} resampled, {copied} copied as-is.")
    print(f"Output: {args.out_images}")


if __name__ == '__main__':
    main()
