# nnUNet-centerline

An nnU-Net-based pipeline for vascular segmentation from CT/MRI, extended with a
connectivity-aware training loss and a VMTK-based centerline/morphology analysis
stage. Built and validated on aortic CTA and MRI data, but the trainer, loss, and
QC tooling aren't aorta-specific — they apply to any tubular anatomical structure
segmented with nnU-Net.

## What's here

- **A custom nnU-Net trainer** ([`nnunet_custom/`](nnunet_custom)) that extends the
  standard nnU-Net v2 training loop with:
  - **clDice loss** ([`topology_losses.py`](nnunet_custom/topology_losses.py)) — a
    soft-skeleton connectivity loss added on top of the default Dice+CE loss,
    applied at full resolution only (to avoid computing it on downsampled deep
    supervision scales).
  - **Patient-grouped cross-validation** ([`nnUNetTrainerAorta.py`](nnunet_custom/nnUNetTrainerAorta.py)) —
    splits are generated with `GroupKFold` on patient ID rather than plain
    `KFold` on case ID, so multiple scans of the same patient (e.g. follow-up
    imaging) always land in the same fold. Plain K-fold would let a model see
    one scan of a patient in training and a follow-up of the *same* patient in
    validation, silently inflating validation scores.
  - Axis-restricted mirroring augmentation appropriate for an elongated,
    top-to-bottom anatomical structure.
- **Preprocessing utilities** ([`preprocessing/`](preprocessing)) for getting raw
  clinical data into nnU-Net's expected format: dataset restructuring, spacing
  audits, resampling of thick-slice scans, mismatched image/label pair
  detection, and inference-folder layout conversion.
- **Postprocessing / QC** ([`postprocessing/`](postprocessing)) built on VMTK:
  surface mesh extraction, centerline extraction, per-point radius profiling,
  and morphology metrics (length, surface area, radius statistics), plus a
  spike/discontinuity-based QC check and a spatial-alignment sanity check for
  predictions vs. ground truth.
- **Earlier baseline experiments** ([`earlier_experiments/`](earlier_experiments)) —
  standalone 2D and 3D U-Net implementations used before adopting nnU-Net as
  the primary framework.

## On the topology loss

The clDice term is meant to improve topological correctness (fewer broken/disconnected
segments), not raw voxel overlap — Dice is the wrong metric to judge it by, and
in an informal comparison on this dataset it didn't move Dice in either direction.
Fragmentation (connected-component count) was also inconclusive on the small
sample checked so far. This isn't a properly controlled ablation (same data
split, otherwise-identical trainer) yet, so treat the loss as a reasonable,
literature-motivated addition rather than a proven improvement until a proper
same-split ablation is run.

## Requirements

- Python 3.10+, PyTorch
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
- SimpleITK, NumPy, SciPy
- VTK + [VMTK](http://www.vmtk.org/) (only needed for the postprocessing/centerline
  scripts)

## Usage

1. **Prepare data** — organize raw images/labels into nnU-Net's `imagesTr`/`labelsTr`
   layout with [`preprocessing/prepare_nnunet_data.py`](preprocessing/prepare_nnunet_data.py)
   (or [`Dataset021_CTAAorta.py`](nnunet_custom/Dataset021_CTAAorta.py) for the
   CTA dataset format), after checking/fixing spacing and image-label mismatches
   with the other `preprocessing/` scripts.
2. **Plan and preprocess** with standard nnU-Net (`nnUNetv2_plan_and_preprocess`).
3. **Train** with the custom trainer, e.g.:
   ```bash
   nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerAorta
   ```
4. **Run inference** on new data restructured with
   [`restructure_for_inference.py`](preprocessing/restructure_for_inference.py),
   using standard nnU-Net inference.
5. **QC and analyze** predictions with the `postprocessing/` scripts — spatial
   alignment checks, centerline/morphology extraction, and radius-based QC.

## Reference

Segmentation label scheme and CTA data reference:
[aortaseg24.grand-challenge.org](https://aortaseg24.grand-challenge.org/)

If you use this, please also cite nnU-Net:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
> nnU-Net: a self-configuring method for deep learning-based biomedical image
> segmentation. *Nature Methods*, 18(2), 203-211.
