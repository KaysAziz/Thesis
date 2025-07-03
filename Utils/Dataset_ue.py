import torch
import torch.nn as nn
import torch.amp
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class VialDatasetUE(Dataset):
    def __init__(self, dir):
        self.dir = dir

    def __len__(self):
        return len(os.listdir(self.dir))
    
    def __getitem__(self, idx):
        idx = int(idx)
        img_idx = sorted(os.listdir(self.dir))[idx]

        imgs = torch.load(f'{self.dir}/{img_idx}/Images.pt')
        seg_masks = torch.load(f'{self.dir}/{img_idx}/Masks.pt')
        depth_masks = torch.load(f'{self.dir}/{img_idx}/Depths.pt')
        # CamTrans = torch.load(f'{self.dir}/{img_idx}/CamTrans.pt')

        return imgs, seg_masks, depth_masks