import os
import argparse
import numpy as np
import nibabel as nib
import torch

from Unet_3d import UNet3D

CHECKPOINT  = "best_model_3d.pth"
IMAGE_DIR   = r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\imagesTs"
OUTPUT_DIR  = r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\labelsTs_3d"
PATCH_SIZE  = (128, 128, 64)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def sliding_window_inference(model, image_volume, patch_size, stride=None):
    if stride is None:
        stride = patch_size

    pH, pW, pD = patch_size
    sH, sW, sD = stride
    H, W, D = image_volume.shape
    orig_H, orig_W, orig_D = H, W, D

    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    pad_d = max(0, pD - D)
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        image_volume = np.pad(image_volume, ((0, pad_h), (0, pad_w), (0, pad_d)))
        H, W, D = image_volume.shape

    output = np.zeros((H, W, D), dtype=np.float32)
    count  = np.zeros((H, W, D), dtype=np.float32)

    ys = list(range(0, max(1, H - pH + 1), sH))
    xs = list(range(0, max(1, W - pW + 1), sW))
    zs = list(range(0, max(1, D - pD + 1), sD))
    if ys[-1] != H - pH: ys.append(H - pH)
    if xs[-1] != W - pW: xs.append(W - pW)
    if zs[-1] != D - pD: zs.append(D - pD)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                for z in zs:
                    y_end = min(y + pH, H);  y_start = y_end - pH
                    x_end = min(x + pW, W);  x_start = x_end - pW
                    z_end = min(z + pD, D);  z_start = z_end - pD

                    patch = image_volume[y_start:y_end, x_start:x_end, z_start:z_end].copy()
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    probs = torch.sigmoid(model(patch_tensor)).squeeze().cpu().numpy()
                    output[y_start:y_end, x_start:x_end, z_start:z_end] += probs
                    count[y_start:y_end, x_start:x_end, z_start:z_end]  += 1

    output /= np.maximum(count, 1)
    return (output[:orig_H, :orig_W, :orig_D] > 0.5).astype(np.uint8)


def run_inference(image_dir, output_dir, checkpoint):
    os.makedirs(output_dir, exist_ok=True)

    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"loaded {checkpoint} on {device}")

    files = sorted(f for f in os.listdir(image_dir) if f.endswith('.nii.gz'))
    for i, filename in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {filename} ...", end=" ", flush=True)

        orig_nii = nib.load(os.path.join(image_dir, filename))
        image = np.asarray(orig_nii.get_fdata(), dtype=np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)

        pred = sliding_window_inference(model, image, PATCH_SIZE)

        out_nii = nib.Nifti1Image(pred, orig_nii.affine, orig_nii.header)
        nib.save(out_nii, os.path.join(output_dir, filename))
        print("saved")

    print(f"done. predictions in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  default=IMAGE_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    run_inference(args.image_dir, args.output_dir, args.checkpoint)
