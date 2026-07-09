"""
VMTK-based radius quality control for aorta segmentations.

For each predicted segmentation mask, this script:
  1. Converts the binary mask to a surface mesh
  2. Extracts the aorta centerline
  3. Computes the maximum inscribed sphere radius at each centerline point
  4. Reports: mean radius, std, max spike (largest jump between adjacent points),
     and a pass/fail flag based on a configurable spike threshold.

Usage:
    python vmtk_radius_qc.py --seg /path/to/seg.nii.gz [--threshold 0.5] [--out results.csv]

    Or process a folder of segmentations:
    python vmtk_radius_qc.py --seg_dir /path/to/segs/ [--threshold 0.5] [--out results.csv]

Requirements:
    pip install vmtk
    pip install SimpleITK numpy pandas
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import os
os.environ['VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN'] = '1'


def _require_vmtk():
    try:
        import vmtk.vmtkscripts as vmtkscripts
        return vmtkscripts
    except ImportError:
        sys.exit(
            "VMTK is not installed. Install it with:\n"
            "  conda install -c vmtk vmtk\n"
            "or see https://www.vmtk.org/download/"
        )


def mask_to_surface(seg_path: Path):
    """Convert a binary NIfTI mask to a VTK surface via marching cubes."""
    import vtk
    import vtk.util.numpy_support as vtk_np

    # Read mask and convert to VTK image
    sitk_img = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.uint8)  # (Z, Y, X)
    spacing = sitk_img.GetSpacing()   # (X, Y, Z)
    origin = sitk_img.GetOrigin()

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(origin)

    flat = arr.flatten(order='C').astype(np.float32)
    vtk_arr = vtk_np.numpy_to_vtk(flat, deep=True)
    vtk_img.GetPointData().SetScalars(vtk_arr)

    # Marching cubes at isovalue 0.5
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, 0.5)
    mc.Update()

    surface = mc.GetOutput()
    return surface, vtk_img


def extract_centerline(surface, vtk_img):
    """
    Extract aorta centerline using VMTK.
    Seed points are the two open-profile centres found after capping
    (via vmtkBoundaryReferenceSystems), fed to vmtkCenterlines explicitly.
    """
    vmtkscripts = _require_vmtk()

    # Cap the surface openings
    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Method = 'centerpoint'
    capper.Interactive = 0
    capper.Execute()
    capped = capper.Surface

    # Find the two boundary profile centres to use as source/target
    profiler = vmtkscripts.vmtkBoundaryReferenceSystems()
    profiler.Surface = capped
    profiler.Execute()
    ref = profiler.ReferenceSystems
    n = ref.GetNumberOfPoints()
    if n < 2:
        raise RuntimeError("Could not find two open profiles on the surface — capping may have failed.")
    p0 = ref.GetPoint(0)
    p1 = ref.GetPoint(n - 1)

    # Extract centerline using explicit source/target points
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = capped
    cl.SeedSelectorName = 'pointlist'
    cl.SourcePoints = list(p0)
    cl.TargetPoints = list(p1)
    cl.AppendEndPoints = 1
    cl.Execute()
    return cl.Centerline


def compute_radius_profile(centerline) -> np.ndarray:
    """Extract the MaximumInscribedSphereRadius array from the centerline."""
    radius_array = centerline.GetPointData().GetArray('MaximumInscribedSphereRadius')
    if radius_array is None:
        raise RuntimeError(
            "Centerline does not contain 'MaximumInscribedSphereRadius'. "
            "Check that VMTK centerline extraction succeeded."
        )
    n = radius_array.GetNumberOfTuples()
    radii = np.array([radius_array.GetTuple1(i) for i in range(n)])
    return radii


def analyse_radius(radii: np.ndarray, spike_threshold: float = 0.5) -> dict:
    """
    Compute summary statistics and flag spikes.

    spike_threshold: fractional change between adjacent points that counts as a spike.
                     0.5 means a 50% jump in radius triggers the flag.
    """
    if len(radii) < 2:
        return {'mean': float('nan'), 'std': float('nan'),
                'max_spike': float('nan'), 'n_spikes': 0, 'pass': False,
                'note': 'fewer than 2 centerline points'}

    jumps = np.abs(np.diff(radii)) / (radii[:-1] + 1e-6)
    max_spike = float(jumps.max())
    n_spikes = int((jumps > spike_threshold).sum())
    passed = n_spikes == 0

    return {
        'mean_radius_mm': float(radii.mean()),
        'std_radius_mm': float(radii.std()),
        'min_radius_mm': float(radii.min()),
        'max_radius_mm': float(radii.max()),
        'max_fractional_spike': max_spike,
        'n_spikes': n_spikes,
        'spike_threshold': spike_threshold,
        'pass': passed,
        'n_centerline_points': len(radii),
    }


def run_qc(seg_path: Path, spike_threshold: float = 0.5, verbose: bool = True) -> dict:
    result = {'file': str(seg_path)}
    try:
        surface, vtk_img = mask_to_surface(seg_path)
        centerline = extract_centerline(surface, vtk_img)
        radii = compute_radius_profile(centerline)
        stats = analyse_radius(radii, spike_threshold)
        result.update(stats)
        if verbose:
            status = 'PASS' if stats['pass'] else 'FAIL'
            print(f"[{status}] {seg_path.name}")
            print(f"       mean={stats['mean_radius_mm']:.2f}mm  "
                  f"std={stats['std_radius_mm']:.2f}mm  "
                  f"max_spike={stats['max_fractional_spike']:.2%}  "
                  f"n_spikes={stats['n_spikes']}")
    except Exception as e:
        result['error'] = str(e)
        result['pass'] = False
        if verbose:
            print(f"[ERROR] {seg_path.name}: {e}")
    return result


def main():
    parser = argparse.ArgumentParser(description="VMTK radius QC for aorta segmentations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seg', type=Path, help="Path to a single segmentation NIfTI file.")
    group.add_argument('--seg_dir', type=Path, help="Folder containing segmentation NIfTI files.")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Fractional radius jump that counts as a spike (default: 0.5).")
    parser.add_argument('--out', type=Path, default=None,
                        help="Optional CSV path to save results.")
    args = parser.parse_args()

    seg_files = []
    if args.seg:
        seg_files = [args.seg]
    else:
        seg_files = sorted(args.seg_dir.glob('*.nii.gz'))
        if not seg_files:
            sys.exit(f"No .nii.gz files found in {args.seg_dir}")

    results = [run_qc(f, spike_threshold=args.threshold) for f in seg_files]

    passed = sum(1 for r in results if r.get('pass', False))
    print(f"\n{passed}/{len(results)} segmentations passed QC.")

    if args.out:
        fieldnames = sorted({k for r in results for k in r.keys()})
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {args.out}")


if __name__ == '__main__':
    main()
