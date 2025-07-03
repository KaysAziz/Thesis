import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
import torch.amp
import clip
from PIL import Image
import os


class VialDatasetMulti(Dataset):
    def __init__(self, img_dir, label_dir, transform = None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transforms.Compose([
                            transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias="warn"),
                            transforms.CenterCrop(size=(448, 448)),
                            transforms.Lambda(lambda image: image.convert("RGB")),
                            transforms.ToTensor()
                            ])

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        idx = int(idx)
        img_idx = sorted(os.listdir(self.img_dir))[idx]
        img_set = sorted(os.listdir(f'{self.img_dir}/{img_idx}'))[1:]
        label = torch.load(f"{self.label_dir}/{img_idx}/annotations/segmentation_vials.pt")
        images, labels = [], []
        for i in label:
            label = transforms.Resize(size=448, interpolation=InterpolationMode.NEAREST_EXACT, max_size=None, antialias="warn")(i.unsqueeze(0)).squeeze(0)
            temp_label = transforms.CenterCrop(size=(448,448))(label)
            labels.append(temp_label)
        
        for i in img_set:
            if i.endswith('.png'):
                temp_img = Image.open(f'{self.img_dir}/{img_idx}/{i}')
                images.append(self.transform(temp_img))

        images = torch.stack(images)
        labels = torch.stack(labels)

        return images, labels

