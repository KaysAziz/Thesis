import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.amp
import clip
import os
from datetime import datetime
from tensorboardX import SummaryWriter
from Utils.Dataset_ue import VialDatasetUE
from Utils.Dataset_MultiView import VialDatasetMulti
from Utils.Dataset_SingleView import VialDatasetSingle
from Utils.Utils import *
from Models.ModifiedResNetSkip import ModifiedResNet
from Models.MultiheadResnet import MultiHeadResnet as MultiHeadResNet
from Models.SingleViewModel import SVM
from Models.MultiViewModel import MVM
from Models.DepthModel import DepthModel as DM
from Models.Checkpoint import Checkpoint as cp
import argparse


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1"}:
        return True
    elif value.lower() in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
parser = argparse.ArgumentParser(description="Needed")
parser.add_argument("--model", type=str, required=True, help="Specify the model")
parser.add_argument("--pretrained", type=str_to_bool, nargs="?", const=True, default="True", help="Set --pretrained=True/False")
parser.add_argument("--path", type=str, default=os.path.join(BASE_DIR, "Dataset"), help="Set dataset parent directory")

args = parser.parse_args()

model_names = {
    "Single": os.path.join(BASE_DIR, "Models", "Weights", "Single"),
    "Multi": os.path.join(BASE_DIR, "Models", "Weights", "Multi"),
    "Double": os.path.join(BASE_DIR, "Models", "Weights", "Double"),
    "Depth": os.path.join(BASE_DIR, "Models", "Weights", "Depth"),
}


model_name = args.model
if model_name in model_names:
    model_path_p = model_names[args.model]
else:
    raise ValueError("Model name non-existant. Choices: Single, Multi, Double, Depth")



if model_name == "Depth":
    model_path = os.path.join(model_path_p, "Depth_RMSE_Pretrained.pt")
    data_path = os.path.join(args.path,"Synthetic")
else:
    model_path = os.path.join(model_path_p, f"{model_name}_Pretrained.pt")
    data_path_single = os.path.join(args.path, "Real", "Single_View")
    data_path = os.path.join(args.path, "Real", "Batched_Views")


pretrain = args.pretrained


curr_time = datetime.now().strftime("%m%d_%H%M%S")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("RN50x64", device=device)

if model_name != "Double":
    log_dir = os.path.join(BASE_DIR, "Models", "Logs", model_name)
    cp().create_folder(f"{log_dir}/{curr_time}")


batch_size_depth = 2


def train_argument(model_path, model, args):
    model_name = args.model
    pretrain = args.pretrained
    if model_name != "Depth":
        if pretrain is True:
            listed = os.listdir(model_path)
            loading = [i for i in listed if i.startswith(f"{model_name}_Pre*")]
            weights = torch.load(f"{model_path}/{loading}", weights_only=False)
            modres = ModifiedResNet(model, 4)
            modres.load_state_dict(weights['model_state_dict'])
            modres.to(device)
            optimizer = optim.Adam(modres.parameters(), lr=1e-4, fused=True, weight_decay=1e-4)
            optimizer.load_state_dict(weights["optimizer_state_dict"])
            scores = weights['Segmentation_Scores']
            return modres, optimizer, scores
        else:
            modres = ModifiedResNet(model, 4)
            modres.to(device)
            optimizer = optim.Adam(modres.parameters(), lr=1e-4, fused=True, weight_decay=1e-4)
            scores = [0,0]
            return modres, optimizer, scores
    else:
        trans_mat = torch.load(os.path.join(BASE_DIR, "Utils", "Camera_Extrinsics.pt"), weights_only=False)
        intrinsics = torch.load(os.path.join(BASE_DIR, "Utils", "Camera_Intrinsics.pt"), weights_only=False)
        trans_mat = trans_mat.squeeze(0).repeat(batch_size_depth,1,1,1) #B, 6, 4, 4
        intrinsics = intrinsics.squeeze(0).repeat(batch_size_depth,1,1) #B, 3, 3
        if pretrain is True:
            listed = os.listdir(model_path)
            loading = [i for i in listed if i.startswith("Depth_model_rmse*")]
            weights = torch.load(f"{model_path}/{loading}", weights_only=False)
            modres = MultiHeadResNet(model, 4, depth_map=depth_values, view_transformation_matrices=trans_mat, intrinsics=intrinsics)
            modres.load_state_dict(weights['model_state_dict'])
            modres.to(device)
            optimizer = optim.Adam(modres.parameters(), lr=1e-4, fused=True, weight_decay=1e-4)
            optimizer.load_state_dict(weights["optimizer_state_dict"])
            scores = weights['Depth_Scores']
            return modres, optimizer, scores
        else:
            modres = MultiHeadResNet(model, 4, depth_map=depth_values, view_transformation_matrices=trans_mat, intrinsics=intrinsics)
            modres.to(device)
            optimizer = optim.Adam(modres.parameters(), lr=1e-4, fused=True, weight_decay=1e-4)
            scores = [10e+10,10e+10,10e+10]
            return modres, optimizer, scores


def get_dataset(data_path, name):
    dataloaders = []
    if name == 'Depth':
        batch_size = 2
        dataset = VialDatasetUE(data_path)
        train_size = int(0.8*len(dataset))
        val_size = int(len(dataset)-train_size)
        train_data, val_data = random_split(dataset, [train_size,val_size])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) #DROPLAST
        val_dataloader = DataLoader(val_data, batch_size=int(batch_size/2), shuffle=True, drop_last=True)
        dataloaders.append(train_dataloader)
        dataloaders.append(val_dataloader)
    elif name == 'Single':
        batch_size = 20
        dataset = VialDatasetSingle(data_path_single)
        train_size = int(0.8*len(dataset))
        val_size = int(len(dataset)-train_size)
        train_data, val_data = random_split(dataset, [train_size,val_size])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) #DROPLAST
        val_dataloader = DataLoader(val_data, batch_size=int(batch_size/2), shuffle=True, drop_last=True)
        dataloaders.append(train_dataloader)
        dataloaders.append(val_dataloader)
    elif name == 'Multi':
        batch_size = 4
        dataset = VialDatasetMulti(data_path)
        train_size = int(0.8*len(dataset))
        val_size = int(len(dataset)-train_size)
        train_data, val_data = random_split(dataset, [train_size,val_size])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) #DROPLAST
        val_dataloader = DataLoader(val_data, batch_size=int(batch_size/2), shuffle=True, drop_last=True)
        dataloaders.append(train_dataloader)
        dataloaders.append(val_dataloader)
    elif name == 'Double':
        batch_size = 20
        dataset = VialDatasetSingle(data_path_single)
        train_size = int(0.8*len(dataset))
        val_size = int(len(dataset)-train_size)
        train_data, val_data = random_split(dataset, [train_size,val_size])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) #DROPLAST
        val_dataloader = DataLoader(val_data, batch_size=int(batch_size/2), shuffle=True, drop_last=True)
        dataloaders.append(train_dataloader)
        dataloaders.append(val_dataloader)

        batch_size = 4
        dataset = VialDatasetMulti(data_path)
        train_size = int(0.8*len(dataset))
        val_size = int(len(dataset)-train_size)
        train_data, val_data = random_split(dataset, [train_size,val_size])
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) #DROPLAST
        val_dataloader = DataLoader(val_data, batch_size=int(batch_size/2), shuffle=True, drop_last=True)
        dataloaders.append(train_dataloader)
        dataloaders.append(val_dataloader)

    return dataloaders


# Depth candidates here are only used to create interpolated depth bins matching final depth output
# Depth candidates are created directly in MatchingVolume.get_volume for the 3D matching volume to dynamically update spatial resolution and num_candidates
max_depth = 280.2500
min_depth = 16.2188
num_candidates = 8
depth_values = [min_depth, max_depth, num_candidates]
target_bins = 256
depth_interval = (max_depth - min_depth) / (num_candidates - 1)


depth_cands = torch.arange(0, num_candidates, requires_grad=False).reshape(1, -1).to(   
            torch.float32) * depth_interval + min_depth

depth_bins = depth_cands.view(1,num_candidates, 1, 1).expand(batch_size_depth, num_candidates,448,448).to(device)
depth_bins = depth_bins.unsqueeze(1) #Add temporary dimension for interpolate requirements
depth_bins_interp = nn.functional.interpolate(depth_bins, size=(target_bins,448,448), mode='trilinear', align_corners=True)  #Trilinear
depth_bins_interp = depth_bins_interp.squeeze(1) # B, target_bins, 448,448



modres, optimizer, scores = train_argument(model_path, model, args)
dataloaders = get_dataset(data_path,model_name)
epochs = 10
model_dir = os.path.join(BASE_DIR, "Model", "Weights", "New_Weights")

if model_name == "Single":
    net = SVM(device, modres, dataloaders[0], log_dir=log_dir, val_dataloader=dataloaders[1], optimizer=optimizer,
            multi=False, scores=scores, model_dir=model_dir)
    net.train(epochs)
elif model_name == "Multi":
    net = MVM(device, modres, dataloaders[0], log_dir=log_dir,
                val_dataloader=dataloaders[1], optimizer=optimizer, model_view=False, scores=scores, model_dir=model_dir) #SCORE OR NO
    net.train(epochs)
elif model_name == "Double":
    log_dir = os.path.join(BASE_DIR, "Models", "Logs", model_name)
    cp().create_folder(f"{log_dir}/{curr_time}", "Stage1")
    net_stage1 = SVM(device, modres, dataloaders[0], log_dir, val_dataloader=dataloaders[1], optimizer=optimizer,
            multi=True, multi_log=f'{log_dir}/Stage1', scores=scores, model_dir=model_dir)
    net_stage1.train(epochs)

    cp().create_folder(f"{log_dir}/{curr_time}", "Stage2")
    net_stage2 = MVM(device, modres, dataloaders[2], f"{log_dir}/Stage2",
                val_dataloader=dataloaders[3], optimizer=optimizer, model_view=True, scores=scores, model_dir=model_dir) #SCORE OR NO
    net_stage2.train(epochs)
elif model_name == "Depth":
    net_depth = DM(device, modres, dataloaders[0], log_dir, val_dataloader=dataloaders[1], optimizer=optimizer,
            depth_bins=depth_bins_interp, scores_depth=scores, model_dir=model_dir)
    net_depth.train(epochs)





