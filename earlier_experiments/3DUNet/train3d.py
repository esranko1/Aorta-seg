import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib

from dataset3d import AortaDataset3D
from Unet_3d import UNet3D

IMAGE_DIR = r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\imagesTr"
MASK_DIR = r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\labelsTr"

PATCH_SIZE    = (128, 128, 64)
BATCH_SIZE    = 2
EPOCHS        = 15
LR            = 1e-4
VAL_FRACTION  = 0.2
NUM_WORKERS   = 4
SEED          = 42
CHECKPOINT    = "best_model_3d.pth"

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
use_amp = (device == "cuda")

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(1)
    union = probs.sum(1) + targets.sum(1)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()

bce = nn.BCEWithLogitsLoss()

def combined_loss(logits, targets):
    return bce(logits, targets) + dice_loss(logits, targets)


def patient_split(dataset, val_fraction, seed):
    patients = sorted(dataset.image_files)
    random.Random(seed).shuffle(patients)
    n_val = int(len(patients) * val_fraction)
    val_patients = set(patients[:n_val])

    n_patients = len(dataset.image_files)
    spv = dataset.samples_per_volume

    # Each training patient gets spv indices spread across the full dataset length.
    # dataset[i] maps to patient i % n_patients, so patient p's samples live at
    # indices p, p+n_patients, p+2*n_patients, ... p+(spv-1)*n_patients.
    train_idx = [
        patient_i + s * n_patients
        for s in range(spv)
        for patient_i, fn in enumerate(dataset.image_files)
        if fn not in val_patients
    ]
    # val uses one index per patient so validate()'s sliding-window loop sees each patient once
    val_idx = [
        patient_i
        for patient_i, fn in enumerate(dataset.image_files)
        if fn in val_patients
    ]

    n_train = n_patients - n_val
    print(f"patients: {n_patients} total -> {n_train} train / {n_val} val  ({len(train_idx)} train patches)")

    from torch.utils.data import Subset
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    running = 0
    for i, (image, mask) in enumerate(loader):
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if epoch == 1 and i < 5:
            print(f"  batch {i} foreground %: {mask.mean().item():.4f}")
        optimizer.zero_grad()
        with torch.amp.autocast(device_type = "cuda", enabled = use_amp):
            logits = model(image)
            loss = combined_loss(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * image.size(0)
    return running / len(loader.dataset)

def sliding_window_inference(model, image_volume, patch_size, stride=None):
    if stride is None:
        stride = patch_size  # no overlap
        
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
    count = np.zeros((H, W, D), dtype=np.float32)

    ys = list(range(0, max(1, H - pH + 1), sH))
    xs = list(range(0, max(1, W - pW + 1), sW))
    zs = list(range(0, max(1, D - pD + 1), sD))
    if ys[-1] != H - pH: ys.append(H - pH)
    if xs[-1] != W - pW: xs.append(W - pW)
    if zs[-1] != D - pD: zs.append(D - pD)

    for y in ys:
        for x in xs:
            for z in zs:
                y_end = min(y +pH, H)
                x_end = min(x +pW, W)
                z_end = min(z +pD, D)

                y_start = y_end - pH
                x_start = x_end - pW
                z_start = z_end - pD

                patch = image_volume[y_start:y_end, x_start:x_end, z_start:z_end].copy()
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, pH, pW, pD)

                with torch.no_grad():
                    logits = model(patch_tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (pH, pW, pD)
                output[y_start:y_end, x_start:x_end, z_start:z_end] += probs
                count[y_start:y_end, x_start:x_end, z_start:z_end] += 1
    output /= np.maximum(count, 1)
    return (output[:orig_H, :orig_W, :orig_D] > 0.5).astype(np.float32)



@torch.no_grad()
def validate(model, val_dataset, patch_size):
    model.eval()
    if not hasattr(validate, '_cache'):
        validate._cache = {}

    dice_scores = []
    for idx in range(len(val_dataset)):
        vol_idx = val_dataset.indices[idx]
        if vol_idx not in validate._cache:
            filename = val_dataset.dataset.image_files[vol_idx]
            image = nib.load(os.path.join(val_dataset.dataset.image_dir, filename)).get_fdata()
            image = np.asarray(image, dtype=np.float32)
            image = (image - image.mean()) / (image.std() + 1e-8)
            mask = nib.load(os.path.join(val_dataset.dataset.mask_dir, filename)).get_fdata()
            mask = (mask > 0).astype(np.float32)
            validate._cache[vol_idx] = (image, mask)

        image, mask = validate._cache[vol_idx]
        pred = sliding_window_inference(model, image, patch_size)
        inter = (pred * mask).sum()
        union = pred.sum() + mask.sum()
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        dice_scores.append(dice)

    return np.mean(dice_scores)

def main():
    torch.manual_seed(SEED)

    full_ds = AortaDataset3D(IMAGE_DIR, MASK_DIR, patch_size=PATCH_SIZE, samples_per_volume=10)
    train_ds, val_ds = patient_split(full_ds, VAL_FRACTION, SEED)

    # persistent_workers keeps worker processes alive between epochs so their volume
    # caches survive and disk reads only happen on first access per worker.
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)

    best_dice = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        val_dice = validate(model, val_ds, PATCH_SIZE)
        scheduler.step(val_dice)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {lr_now:.6f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  New best model saved with Dice: {best_dice:.4f}")
        
    
    print(f"done. best val dice: {best_dice:.4f}")

if __name__ == "__main__":
    print("Starting 3D UNet training....")
    print(f"using device: {device} | AMP: {use_amp}")
    main()