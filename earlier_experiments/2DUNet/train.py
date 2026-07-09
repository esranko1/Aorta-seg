import random
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
 
from dataset import AortaDataset
from Unet_2d import UNet
 
 
IMAGE_DIR = r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\imagesTr"
MASK_DIR = r"C:\Users\esranko1\Desktop\Research_seg\Data\Task09_Spleen\labelsTr"
BATCH_SIZE    = 32        
EPOCHS        = 100
LR            = 1e-4
VAL_FRACTION  = 0.2
NUM_WORKERS   = 0        
SEED          = 42
CHECKPOINT    = "best_model.pth"
 
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
 
 
@torch.no_grad()
def dice_score(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float().flatten(1)
    targets = targets.flatten(1)
    inter = (preds * targets).sum(1)
    union = preds.sum(1) + targets.sum(1)
    return ((2 * inter + eps) / (union + eps)).mean().item()
 
 

def patient_split(dataset, val_fraction, seed):
    patients = sorted({fn for fn, _ in dataset.slices})
    random.Random(seed).shuffle(patients)
    n_val = int(len(patients) * val_fraction)
    val_patients = set(patients[:n_val])
 
    train_idx = [i for i, (fn, _) in enumerate(dataset.slices) if fn not in val_patients]
    val_idx   = [i for i, (fn, _) in enumerate(dataset.slices) if fn in val_patients]
    print(f"patients: {len(patients)} total -> "
          f"{len(patients) - n_val} train / {n_val} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
 

def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    running = 0.0
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(image)
            loss = combined_loss(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * image.size(0)
    return running / len(loader.dataset)
 
 
@torch.no_grad()
def validate(model, loader):
    model.eval()
    running_loss, running_dice = 0.0, 0.0
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(image)
        running_loss += combined_loss(logits, mask).item() * image.size(0)
        running_dice += dice_score(logits, mask) * image.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_dice / n
 
 

def main():
    torch.manual_seed(SEED)
 
    full_ds = AortaDataset(IMAGE_DIR, MASK_DIR, skip_empty=True)
    train_ds, val_ds = patient_split(full_ds, VAL_FRACTION, SEED)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
 
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
 
    best_dice = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_loss, val_dice = validate(model, val_loader)
        scheduler.step(val_dice)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:03d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | dice {val_dice:.4f} | lr {lr_now:.1e}")
 
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  saved new best (dice {best_dice:.4f}) -> {CHECKPOINT}")
 
    print(f"done. best val dice: {best_dice:.4f}")
 
 
if __name__ == "__main__":
    print(f"using device: {device} | amp: {use_amp}")
    main()