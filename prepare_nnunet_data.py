"""
Organizes AortaShape data into nnUNet-compatible imagesTr / labelsTr folders.

Folder naming supported:  PT###_M   or   PT###_M_F#
Searches recursively within each case folder for:
  - *.nii.gz              → imagesTr/<CASE>_0000.nii.gz  (copied as-is)
  - Segmentation.seg.nrrd → labelsTr/<CASE>.nii.gz       (converted via SimpleITK)

Usage:
    python prepare_nnunet_data.py \
        --src /path/to/AortaShape \
        --out /path/to/output_dataset
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import SimpleITK as sitk


FOLDER_PATTERN = re.compile(r"^PT\d+_M(_F\d+)?$")


def find_cases(src: Path) -> list[Path]:
    return sorted(
        p for p in src.iterdir()
        if p.is_dir() and FOLDER_PATTERN.match(p.name)
    )


def find_label(case_dir: Path) -> Path | None:
    return next(case_dir.rglob("Segmentation.seg.nrrd"), None)


def find_mri(case_dir: Path, label: Path | None) -> Path | None:
    # Prefer .nii.gz in the same folder as the segmentation (most likely the scanned image)
    search_dirs = []
    if label is not None:
        search_dirs.append(label.parent)
    search_dirs.append(case_dir)

    for d in search_dirs:
        candidates = [
            p for p in d.glob("*.nii.gz")
            if "segmentation" not in p.name.lower()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return candidates[0]

    return None


def copy_cases(src: Path, out: Path, dry_run: bool = False) -> None:
    images_dir = out / "imagesTr"
    labels_dir = out / "labelsTr"

    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    cases = find_cases(src)
    if not cases:
        sys.exit(f"No matching folders found in {src}")

    skipped = []
    copied = []

    for case_dir in cases:
        case_id = case_dir.name
        label_src = find_label(case_dir)
        mri_src = find_mri(case_dir, label_src)

        missing = []
        if label_src is None:
            missing.append("Segmentation.seg.nrrd (not found)")
        if mri_src is None:
            missing.append("*.nii.gz (not found)")

        if missing:
            skipped.append((case_id, missing))
            continue

        img_dst = images_dir / f"{case_id}_0000.nii.gz"
        lbl_dst = labels_dir / f"{case_id}.nii.gz"

        print(f"  {case_id}")
        print(f"    MRI   : {mri_src.relative_to(case_dir)} → imagesTr/{img_dst.name}")
        print(f"    Label : {label_src.relative_to(case_dir)} → labelsTr/{lbl_dst.name} (nrrd→nii.gz)")

        if not dry_run:
            shutil.copy2(mri_src, img_dst)
            seg = sitk.ReadImage(str(label_src))
            sitk.WriteImage(seg, str(lbl_dst))

        copied.append(case_id)

    print(f"\nDone: {len(copied)} cases copied, {len(skipped)} skipped.")
    if skipped:
        print("\nSkipped:")
        for case_id, reasons in skipped:
            print(f"  {case_id}: {', '.join(reasons)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AortaShape data for nnUNet.")
    parser.add_argument("--src", required=True, type=Path, help="Path to AortaShape folder")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset folder")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be copied without actually copying"
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        sys.exit(f"Source not found: {args.src}")

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"{mode}Source : {args.src}")
    print(f"{mode}Output : {args.out}\n")

    copy_cases(args.src, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
