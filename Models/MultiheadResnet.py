import torch
import torch.nn as nn
import torch.amp
import clip
from Utils.Utils import *
from torch.utils.checkpoint import checkpoint as cp


class MultiHeadResnet(nn.Module):
    """
    ResNet encoder-decoder architecture. Passes batches of images through CLIP's
    encoder network which can then be used for the Semantic Segmentation and Depth Estimation head.
    Skip connections are instantiated and are used directly in the Segmentation head for upsampling.
    Depth skip connections are created by first creating 3D matching volumes which are then concatenated
    and used as skip connections to get upsampled back to original resolution.
    for classification.
    Args:
        clip_model (CLIP.Object): A loaded clip network.
        num_classes (int): An integer value representing number of classes + background.
        depth_values (list): List containing min depth, max depth and number of depth candidates, in that order.
        MatchingVolClass (MatchingVolume): Matching volume class.
        view_transformation_matrices (torch.Tensor): Tensor containing the camera extrinsic matrix for all each view.
        intrinsics (torch.Tensor): Tensor containing the camera intrinsic matrix.
    """
    def __init__(self, clip_model, num_classes=4, depth_values = None, MatchingVolClass = None, view_transformation_matrices = None, intrinsics = None,
                inference = False):
        super().__init__()
        self.depth_values = depth_values
        self.trans_mats = view_transformation_matrices
        self.intrinsics = intrinsics
        self.match = MatchingVolClass
        self.clip_model = clip_model
        self.num_classes = num_classes
        self.inference = inference
        self.width = 128
        self.embed_dim = self.width*32
        self.upsample1 = nn.ConvTranspose2d(self.embed_dim, self.width * 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(self.width * 16)
        self.upsample2 = nn.ConvTranspose2d(self.width * 16, self.width * 8, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(self.width * 8)
        self.upsample3 = nn.ConvTranspose2d(self.width * 8, self.width * 4, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(self.width * 4)
        self.upsample4 = nn.ConvTranspose2d(self.width * 4, self.width, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(self.width)
        self.upsample5 = nn.ConvTranspose2d(self.width, self.width//2, kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(self.width // 2)
        self.upsample6 = nn.ConvTranspose2d(self.width//2, self.width//4, kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(self.width//4)


        self.upsample1_D = nn.ConvTranspose3d(self.embed_dim, self.width * 16, kernel_size=2, stride=2)
        self.upsample1_D_skip = nn.ConvTranspose3d(self.embed_dim, self.width * 16, kernel_size=1, stride=1)
        self.bn1_D = nn.BatchNorm3d(self.width * 16)
        self.upsample2_D = nn.ConvTranspose3d(self.width * 16, self.width * 8, kernel_size=2, stride=2)
        self.upsample2_D_skip = nn.ConvTranspose3d(self.width * 16, self.width * 8, kernel_size=1, stride=1)
        self.bn2_D = nn.BatchNorm3d(self.width * 8)
        self.upsample3_D = nn.ConvTranspose3d(self.width * 8, self.width * 4, kernel_size=2, stride=2)
        self.upsample3_D_skip = nn.ConvTranspose3d(self.width * 8, self.width * 4, kernel_size=1, stride=1)
        self.bn3_D = nn.BatchNorm3d(self.width * 4)
        self.upsample4_D = nn.ConvTranspose3d(self.width * 4, self.width, kernel_size=1, stride=1)
        self.upsample4_D_skip = nn.ConvTranspose3d(self.width * 2, self.width, kernel_size=1, stride=1)
        self.bn4_D = nn.BatchNorm3d(self.width)
        self.upsample5_D = nn.ConvTranspose3d(self.width, self.width//2, kernel_size=2, stride=2)
        self.upsample5_D_skip = nn.ConvTranspose3d(self.width, self.width//2, kernel_size=1, stride=1)
        self.bn5_D = nn.BatchNorm3d(self.width//2)
        self.upsample6_D = nn.ConvTranspose3d(self.width//2, self.width//4, kernel_size=2, stride=2)
        self.upsample6_D_skip = nn.ConvTranspose3d(self.width//2, self.width//4, kernel_size=1, stride=1)
        self.bn6_D = nn.BatchNorm3d(self.width//4)
       
        # Final Conv layer to reduce to one channel
        self.final_depth = nn.Conv3d(self.width//4, 1, kernel_size=1, stride=1)
        
        # Final Conv layer to reduce to amount of class channels
        self.final_seg = nn.Conv2d(self.width//4, self.num_classes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        if self.inference:
            for param in self.clip_model.parameters(): #Freezing clip parameters
                param.requires_grad = False
        else:
            for param in self.clip_model.parameters(): #Unfreezing clip parameters
                param.requires_grad = True


    def forward(self,x):
        x1 = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
        x1 = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x1)))
        x1 = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x1)))
        x1 = self.clip_model.visual.avgpool(x1)

        x2 = self.clip_model.visual.layer1(x1)
        x3 = self.clip_model.visual.layer2(x2)
        x4 = self.clip_model.visual.layer3(x3)
        x5 = self.clip_model.visual.layer4(x4)
        x_depth = None
        if not self.inference:
            #depth skip connections spatial dimensions adjusted for MatchingVolume class
            x5_view = cp(lambda v: v.view(4,-1,4096,14,14), x5)
            x4_view = cp(lambda v: v.view([4,-1,*x4.size()[1:]]), x4)
            x3_view = cp(lambda v: v.view([4,-1,*x3.size()[1:]]), x3)
            x2_view = cp(lambda v: v.view([4,-1,*x2.size()[1:]]), x2)
            x1_view = cp(lambda v: v.view([4,-1,*x1.size()[1:]]), x1)

            mv = MatchingVolume()
            matching_volume = cp(lambda v: mv.create_volume(v, self.trans_mats, self.intrinsics, depth_candidates=self.depth_values, device = "cuda"), x5_view)       

            x4_match = cp(lambda v: mv.create_volume(v, self.trans_mats, self.intrinsics, depth_candidates=self.depth_values, device = "cuda", scaling=2), x4_view)
            x3_match = cp(lambda v: mv.create_volume(v, self.trans_mats, self.intrinsics, depth_candidates=self.depth_values, device = "cuda", scaling=4), x3_view)
            x2_match = cp(lambda v: mv.create_volume(v, self.trans_mats, self.intrinsics, depth_candidates=self.depth_values, device = "cuda", scaling=8), x2_view)
            x1_match = cp(lambda v: mv.create_volume(v, self.trans_mats, self.intrinsics, depth_candidates=self.depth_values, device = "cuda", scaling=8), x1_view)

            #Depth head
            x_depth = cp(lambda v: self.relu(self.bn1_D(self.upsample1_D(v))), matching_volume)

            x_depth = cp(lambda v, e: torch.cat((v, e), dim=1), x_depth, x4_match)
            x_depth = cp(lambda v: self.relu(self.bn2_D(self.upsample2_D(self.upsample1_D_skip(v)))), x_depth)

            x_depth = cp(lambda v, e: torch.cat((v, e), dim=1), x_depth, x3_match)
            x_depth = cp(lambda v: self.relu(self.bn3_D(self.upsample3_D(self.upsample2_D_skip(v)))), x_depth)

            x_depth = cp(lambda v, e: torch.cat((v, e), dim=1), x_depth, x2_match)
            x_depth = cp(lambda v: self.relu(self.bn4_D(self.upsample4_D(self.upsample3_D_skip(v)))), x_depth)

            x_depth = cp(lambda v, e: torch.cat((v, e), dim=1), x_depth, x1_match)
            x_depth = cp(lambda v: self.relu(self.bn5_D(self.upsample5_D(self.upsample4_D_skip(v)))), x_depth)

            x_depth = cp(lambda v: self.relu(self.bn6_D(self.upsample6_D(v))), x_depth)

            x_depth = self.final_depth(x_depth)

        #Segmentation head
        x = self.upsample1(x5)
        x = self.bn1(x)
        x = self.relu(x)

        x = x + x4
        x = self.upsample2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x + x3
        x = self.upsample3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x + x2
        x = self.upsample4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = x + x1
        x = self.upsample5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.upsample6(x)
        x = self.bn6(x)
        x = self.relu(x)

        return self.final_seg(x), x_depth


def check_for_nans(tensor, name):
    """
    Helper function, mainly used in Single, Multi, Double.
    Args:
        tensor (torch.Tensor): Any tensor.
        name (Str): String name to output to error message.
    """
    if torch.isnan(tensor).any():
       print(f"NaNs found in {name}")
       return True
    elif torch.isinf(tensor).any():
       print(f"INFs found in {name}")
       return True
    return False
