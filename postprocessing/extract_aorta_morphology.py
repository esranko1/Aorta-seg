import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, label


def _require_vtk():
    try:
        import vtk
        import vtk.util.numpy_support as vtk_np
        return vtk, vtk_np
    except ImportError:
        sys.exit("vtk not found. Install with: pip install vtk")


def _require_vmtk():
    try:
        import vmtk.vmtkscripts as vmtkscripts
        return vmtkscripts
    except ImportError:
        sys.exit("vmtk not found. Install with: conda install -c vmtk vmtk")


# ---------------------------------------------------------------------------
# Build surface mesh (used by both surface area and centreline)
# ---------------------------------------------------------------------------

def get_surface_mesh(arr, spacing, origin):
    vtk, vtk_np = _require_vtk()

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(origin)
    vtk_arr = vtk_np.numpy_to_vtk(arr.flatten(order='C'), deep=True)
    vtk_img.GetPointData().SetScalars(vtk_arr)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, 0.5)
    mc.Update()
    return mc.GetOutput()


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def get_volume(arr, spacing):
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    return float(arr.sum()) * voxel_volume_mm3


# ---------------------------------------------------------------------------
# Surface area
# ---------------------------------------------------------------------------

def get_surface_area(surface):
    vtk, _ = _require_vtk()
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()
    return mass.GetSurfaceArea()


# ---------------------------------------------------------------------------
# DiameterMIS
# ---------------------------------------------------------------------------

def get_diameter_mis(arr, spacing):
    mask = arr.astype(bool)
    sampling = (spacing[2], spacing[1], spacing[0])
    dist = distance_transform_edt(mask, sampling=sampling)
    return float(dist.max()) * 2.0


# ---------------------------------------------------------------------------
# Centreline metrics
# ---------------------------------------------------------------------------

def _get_array(centerline, name):
    arr = centerline.GetPointData().GetArray(name)
    n = arr.GetNumberOfTuples()
    return np.array([arr.GetTuple1(i) for i in range(n)])


def get_centerline_metrics(surface):
    vmtkscripts = _require_vmtk()

    # Cap open ends
    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Method = 'centerpoint'
    capper.Interactive = 0
    capper.Execute()

    # Find the two boundary profile centres to use as source/target
    profiler = vmtkscripts.vmtkBoundaryReferenceSystems()
    profiler.Surface = capper.Surface
    profiler.Execute()
    ref = profiler.ReferenceSystems
    n = ref.GetNumberOfPoints()
    if n < 2:
        raise RuntimeError("Could not find two open profiles on the surface — capping may have failed.")
    p0 = ref.GetPoint(0)
    p1 = ref.GetPoint(n - 1)

    # Extract centreline using explicit source/target points
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = capper.Surface
    cl.SeedSelectorName = 'pointlist'
    cl.SourcePoints = list(p0)
    cl.TargetPoints = list(p1)
    cl.AppendEndPoints = 1
    cl.Resampling = 1
    cl.ResamplingStepLength = 1.0
    cl.Execute()

    # Compute geometry along centreline
    geom = vmtkscripts.vmtkCenterlineGeometry()
    geom.Centerline = cl.Centerline
    geom.Execute()

    centerline = geom.Centerline

    # 3D coordinates of each centreline point
    points = np.array([centerline.GetPoint(i)
                       for i in range(centerline.GetNumberOfPoints())])

    # Length: sum of distances between consecutive points
    length_mm = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()

    # Tortuosity: path length / straight-line end-to-end distance
    end_to_end = np.linalg.norm(points[-1] - points[0])
    tortuosity = length_mm / end_to_end

    # Radius array → cross-section and diameter
    radii       = _get_array(centerline, 'MaximumInscribedSphereRadius')
    max_radius  = radii.max()
    max_cs_area = np.square(max_radius) * np.pi    # π r²
    diameter_ce = 2 * np.sqrt(max_cs_area / np.pi)    # 2 * sqrt(area / π)

    # Curvature and torsion
    curvature = _get_array(centerline, 'Curvature')
    torsion   = _get_array(centerline, 'Torsion')

    return {
        'length_mm':             length_mm,
        'tortuosity':            tortuosity,
        'max_cross_section_mm2': max_cs_area,
        'diameter_ce_mm':        diameter_ce,
        'mean_curvature':        np.mean(curvature),
        'max_curvature':         curvature.max(),
        'mean_torsion':          np.mean(np.abs(torsion)),    # use np.abs(torsion) first
        'max_torsion':           np.abs(torsion).max(),
    }

def keep_largest_component(arr):
    # Step 1: label all connected components
    labeled, num_features = label(arr)

    # Step 2: if there's only one component (or none), return as-is
    if num_features <= 1:
        return arr

    # Step 3: find which label has the most voxels
    # hint: ignore label 0 — that's the background
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0    # ignore background
    largest_label = sizes.argmax()

    # Step 4: return a mask with only that label
    return (labeled == largest_label).astype(np.float32)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seg',     type=Path, help='Single .nii.gz file')
    group.add_argument('--seg_dir', type=Path, help='Folder of .nii.gz files')
    parser.add_argument('--out', type=Path, help='Output CSV path')
    args = parser.parse_args()

    seg_files = [args.seg] if args.seg else sorted(args.seg_dir.glob('*.nii.gz'))

    results = []
    for seg_path in seg_files:
        print(f"Processing {seg_path.name}...")

        img     = sitk.ReadImage(str(seg_path))
        arr     = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = img.GetSpacing()
        origin  = img.GetOrigin()

        arr = keep_largest_component(arr)

        surface = get_surface_mesh(arr, spacing, origin)  # build once, reuse

        # cl_metrics = get_centerline_metrics(surface)

        result = {
            'file':             seg_path.name,
            'volume_mm3':       get_volume(arr, spacing),
            'surface_area_mm2': get_surface_area(surface),
            'diameter_mis_mm':  get_diameter_mis(arr, spacing),
        }
        # result.update(cl_metrics)   # merge centreline metrics in
        results.append(result)

        print(f"  volume:            {result['volume_mm3']:.1f} mm³")
        print(f"  surface area:      {result['surface_area_mm2']:.1f} mm²")
        print(f"  DiameterMIS:       {result['diameter_mis_mm']:.1f} mm")

    if args.out:
        fieldnames = [
            'file', 'volume_mm3', 'surface_area_mm2', 'diameter_mis_mm',
        ]
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()