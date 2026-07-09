import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class AortaDataset3D(Dataset):
    def __init__(self, image_dir, mask_dir=None, patch_size=(128,128,64), samples_per_volume=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.inference_mode = mask_dir is None
        self.samples_per_volume = samples_per_volume
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.nii.gz'))
        self._cache = {}  # vol_idx -> (image, mask, fg_coords), populated on first access

    def __len__(self):
        return len(self.image_files) * self.samples_per_volume

    def _get_volume(self, vol_idx):
        if vol_idx in self._cache:
            return self._cache[vol_idx]

        filename = self.image_files[vol_idx]
        pH, pW, pD = self.patch_size

        image = nib.load(os.path.join(self.image_dir, filename)).get_fdata()
        image = np.asarray(image, dtype=np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)

        mask = None
        if not self.inference_mode:
            mask = nib.load(os.path.join(self.mask_dir, filename)).get_fdata()
            mask = (mask > 0).astype(np.float32)

        H, W, D = image.shape
        pad_h = max(0, pH - H)
        pad_w = max(0, pW - W)
        pad_d = max(0, pD - D)
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, pad_d)))
            if mask is not None:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, pad_d)))

        fg_coords = np.argwhere(mask > 0) if mask is not None else np.empty((0, 3), dtype=np.int64)
        self._cache[vol_idx] = (image, mask, fg_coords)
        return self._cache[vol_idx]

    def __getitem__(self, idx):
        vol_idx = idx % len(self.image_files)
        image, mask, fg_coords = self._get_volume(vol_idx)

        H, W, D = image.shape
        pH, pW, pD = self.patch_size

        if not self.inference_mode and len(fg_coords) > 0 and random.random() < 0.8:
            anchor = fg_coords[random.randint(0, len(fg_coords) - 1)]
            anchor_y, anchor_x, anchor_z = anchor
            y = max(0, min(anchor_y - pH // 2, H - pH))
            x = max(0, min(anchor_x - pW // 2, W - pW))
            z = max(0, min(anchor_z - pD // 2, D - pD))
        else:
            y = random.randint(0, max(0, H - pH))
            x = random.randint(0, max(0, W - pW))
            z = random.randint(0, max(0, D - pD))

        image_patch = torch.from_numpy(image[y:y+pH, x:x+pW, z:z+pD].copy()).unsqueeze(0)

        if self.inference_mode:
            return image_patch, self.image_files[vol_idx], z

        mask_patch = torch.from_numpy(mask[y:y+pH, x:x+pW, z:z+pD].copy()).unsqueeze(0)
        return image_patch, mask_patch
   