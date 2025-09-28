# src/dataset.py
from torch.utils.data import Dataset
import cv2, os
import numpy as np

class FlameDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=(256,256)):
        self.images = sorted([os.path.join(images_dir,f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.masks = sorted([os.path.join(masks_dir,f) for f in os.listdir(masks_dir) if f.endswith('.png')])
        self.size = size
    def __len__(self):
        return min(len(self.images), len(self.masks))
    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.size)
        mask = cv2.resize(mask, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')/255.0
        mask = (mask>127).astype('float32')
        img = img.transpose(2,0,1)
        return img, mask[np.newaxis,:,:]
