import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import OpenEXR
import Imath
import torch.amp
import cv2
import imageio
import clip
from PIL import Image
import os
import argparse
from Models.MultiheadResnet import *
from Models.ModifiedResNetSkip import *

"""
This file is designed to visualize the model accuracy.
If you want to use it on the HPC cluster then you have to comment out
the visualization lines.
"""

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
parser = argparse.ArgumentParser(description="Needed")
parser.add_argument("--model", type=str, required=True, help="Specify the model")
parser.add_argument("--path", type=str, help="Set absolute directory for model")

args = parser.parse_args()

model_names = {
    "Single": os.path.join(BASE_DIR, "Models", "Weights", "Single"),
    "Multi": os.path.join(BASE_DIR, "Models", "Weights", "Multi"),
    "Double": os.path.join(BASE_DIR, "Models", "Weights", "Double"),
    "Depth": os.path.join(BASE_DIR, "Models", "Weights", "Depth"),
}

model_name = args.model
if args.model in model_names:
    model_path_p = model_names[args.model]
else:
    raise ValueError("Model name non-existant. Choices: Single, Multi, Double, Depth")

if model_name == "Depth":
    model_path = os.path.join(model_path_p, "Depth_RMSE_Pretrained.pt")
else:
    model_path = os.path.join(model_path_p, f"{model_name}_Pretrained.pt")

if args.path is not None:
    model_path = args.path

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("RN50x64", device=device)
weights = torch.load(model_path, weights_only=True)


depth_values = [16.2188, 280,2500, 8]
trans_mat = torch.load("/home/kays_/Master/Thesis/Utils/Camera_Extrinsics.pt")
intrinsics = torch.load("/home/kays_/Master/Thesis/Utils/Camera_Intrinsics.pt")
trans_mat = trans_mat.squeeze(0).repeat(2,1,1,1) #B, 6, 4, 4
intrinsics = intrinsics.squeeze(0).repeat(2,1,1)


if model_name == "Depth":
    model = MultiHeadResnet(model,4, depth_values=depth_values, view_transformation_matrices=trans_mat, intrinsics=intrinsics, inference=True)
else:
    model = ModifiedResNet(model,4)

model.load_state_dict(weights['model_state_dict'])
model.to(device)

test_path = os.path.join(BASE_DIR, "Dataset", "Test_Set_Synthetic")



max_depth = 280.2500
min_depth = 16.2188
num_candidates = 8
depth_values = [min_depth, max_depth, num_candidates]
target_bins = 256
depth_interval = (max_depth - min_depth) / (num_candidates - 1)


depth_cands = torch.arange(0, num_candidates, requires_grad=False).reshape(1, -1).to(   
            torch.float32) * depth_interval + min_depth

depth_bins = depth_cands.view(1,num_candidates, 1, 1).expand(2, num_candidates,448,448).to(device)
depth_bins = depth_bins.unsqueeze(1) #Add temporary dimension for interpolate requirements
depth_bins_interp = nn.functional.interpolate(depth_bins, size=(target_bins,448,448), mode='trilinear', align_corners=True)  #Trilinear
depth_bins_interp = depth_bins_interp.squeeze(1) # B, target_bins, 448,448

with torch.no_grad():
    with torch.autocast(device):
        model.eval()
        for scene in os.listdir(test_path):
            img_set = torch.load(os.path.join(test_path, scene, "Images.pt"))
            seg_mask_set = torch.load(os.path.join(test_path, scene, "Masks.pt"))
            depth_mask_set = torch.load(os.path.join(test_path, scene, "Depths.pt"))
            img_set = torch.cat((img_set.unsqueeze(0), img_set.unsqueeze(0)), dim = 0)

            if args.model == "Depth":
                img_pred, depth_pred = model(img_set.to(device).view(12,3,448,448))
                img_pred = torch.softmax(img_pred, dim = 1)
                _, img_pred = torch.max(img_pred, dim = 1)
                # depth_pred = torch.softmax(depth_pred, dim = 1)

                # prob_volume = torch.softmax(depth_pred.squeeze(1), dim=1)
                # predicted_depth = torch.sum(prob_volume * depth_bins_interp, dim=1)
            else:
                img_pred = model(img_set.to(device))
                img_pred = torch.softmax(img_pred, dim = 1)
                img_pred = torch.argmax(img_pred, dim = 1)


            for i, (img, mask, depth) in enumerate(zip(img_set, seg_mask_set, depth_mask_set)):
                plt.imshow(img[i].permute(1,2,0), cmap="gray")
                plt.title("Image input")
                plt.axis("off")
                plt.show()

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                print(mask.size())
                axes[0].imshow(mask, cmap="gray")
                axes[0].set_title("Segmentation Ground Truth")
                axes[0].axis("off")
                print(img_pred.size())
                axes[1].imshow(img_pred[i].cpu(), cmap="gray")
                axes[1].set_title("Segmentation Prediction")
                axes[1].axis("off")

                plt.tight_layout()
                plt.show()


                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # axes[0].imshow(depth, cmap="viridis")
                # axes[0].set_title("Depth Ground Truth")
                # axes[0].axis("off")

                # axes[1].imshow(predicted_depth[i].cpu(), cmap="viridis")
                # axes[1].set_title("Depth Prediction")
                # axes[1].axis("off")

                # plt.tight_layout()
                # plt.show()



def get_depth(mean,std):

    transform2 = transforms.Compose([transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias="warn"),
                                transforms.CenterCrop(size=(448, 448)),
                                # transforms.Lambda(lambda image: image.convert("RGB")),  # Convert image to RGB
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                                ])
    return transform2

def get_depth2(min,max):

    transform2 = transforms.Compose([transforms.Resize(size=448, interpolation=transforms.InterpolationMode.BICUBIC, antialias="warn"),
                                transforms.CenterCrop(size=(448, 448)),
                                # transforms.ToTensor(),
                                transforms.Lambda(lambda x: (x - min) / (max - min))
                                ])
    return transform2

img_path = '/home/kays_/Master/Other_Stuff/Image2.exr'

for i in range(6):
    img_path = f'/home/kays_/Master/Other_Stuff/Image{i}.exr'

    file = OpenEXR.InputFile(img_path)
    header = file.header()


    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Get the data window size (resolution)
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)


    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = file.channels("RGBA", FLOAT)

    # Convert channels to numpy arrays
    r = np.frombuffer(channels[0], dtype=np.float32).reshape(size[1], size[0])
    g = np.frombuffer(channels[1], dtype=np.float32).reshape(size[1], size[0])
    b = np.frombuffer(channels[2], dtype=np.float32).reshape(size[1], size[0])
    d = np.frombuffer(channels[3], dtype=np.float32).reshape(size[1], size[0])
    rgb = torch.stack([torch.tensor(i) for i in [r,g,b]])

    pude = transforms.ToPILImage()

    crop_box = (500, 230, 1600, 1080)
    d = pude(d)
    rgb = pude(rgb)
    rgb = rgb.crop(crop_box)
    rgb = transforms.ToTensor()(rgb)

    d = d.crop(crop_box)
    d = transforms.ToTensor()(d).squeeze(0)
    plt.imshow(rgb.permute(1,2,0), cmap="gray")
    plt.show()
    print("Max:", d.max(), "Min:", d.min())
    
    continue