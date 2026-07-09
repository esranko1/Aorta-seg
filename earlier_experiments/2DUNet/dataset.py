import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class AortaDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, target_size=(256, 256), binary=True, skip_empty=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.binary = binary
        self.inference_mode = mask_dir is None
        self.image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.nii.gz'))

        self.image_cache = {}
        self.mask_cache = {}

        self.slices = []
        for filename in self.image_files:
            img = nib.load(os.path.join(image_dir, filename)).get_fdata()
            n = img.shape[2]
            if not self.inference_mode and skip_empty:
                mask = nib.load(os.path.join(mask_dir, filename)).get_fdata()
                for i in range(n):
                    if mask[:,:,i].any():
                        self.slices.append((filename, i))
            else:
                self.slices.extend([(filename, i) for i in range(n)])
    def __len__(self):
        return len(self.slices)
    
    def _load_volume(self,cache, directory, filename):
        if filename not in cache:
            vol = nib.load(os.path.join(directory, filename)).get_fdata()
            cache[filename] = np.asarray(vol, dtype=np.float32)
        return cache[filename]
    
    def __getitem__(self, idx):
        filename, sl = self.slices[idx]
        image = self._load_volume(self.image_cache, self.image_dir, filename)[:,:,sl]

        image = (image - image.mean()) / (image.std() + 1e-8)
        image = torch.from_numpy(image).float().unsqueeze(0)
        image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        if self.inference_mode:
            return image, filename, sl

        mask = self._load_volume(self.mask_cache, self.mask_dir, filename)[:,:,sl]
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)

        if self.binary:
            mask = (mask > 0).float()
        else:
            mask = mask.squeeze(0).long()

        return image, mask

