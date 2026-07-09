import os
import argparse
import collections

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader

from dataset import AortaDataset
from Unet_2d import UNet

CHECKPOINT   = "best_model.pth"
IMAGE_DIR    =  r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\imagesTs"  # <- fix
OUTPUT_DIR   = r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\labelsTs" # <- fix
TARGET_SIZE  = (256, 256)
BATCH_SIZE   = 16
NUM_WORKERS  = 4

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def run_inference(image_dir, output_dir, checkpoint):
    os.makedirs(output_dir, exist_ok=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"loaded {checkpoint} on {device}")

    dataset = AortaDataset(image_dir, mask_dir=None, target_size=TARGET_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # collect predicted slices per patient: {filename -> {slice_idx -> mask_array}}
    predictions = collections.defaultdict(dict)

    with torch.no_grad():
        for images, filenames, slice_indices in loader:
            images = images.to(device)
            logits = model(images)
            preds  = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()  # (B, 1, H, W)

            for pred, filename, sl in zip(preds, filenames, slice_indices):
                predictions[filename][int(sl)] = pred[0]  # (H, W)

    for filename, slice_map in predictions.items():
        # load original image to get shape and header
        orig_nii = nib.load(os.path.join(image_dir, filename))
        orig_shape = orig_nii.get_fdata().shape  # (X, Y, Z)

        volume = np.zeros(orig_shape, dtype=np.uint8)
        for sl, mask_slice in slice_map.items():
            # resize predicted mask back to original slice spatial dims
            from torch.nn.functional import interpolate
            t = torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0)
            t = interpolate(t, size=(orig_shape[0], orig_shape[1]), mode='nearest')
            volume[:, :, sl] = t.squeeze().numpy().astype(np.uint8)

        out_nii = nib.Nifti1Image(volume, orig_nii.affine, orig_nii.header)
        out_path = os.path.join(output_dir, filename)
        nib.save(out_nii, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    print("Running inference...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  default=IMAGE_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    run_inference(args.image_dir, args.output_dir, args.checkpoint)
