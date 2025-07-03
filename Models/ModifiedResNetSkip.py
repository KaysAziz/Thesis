import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.amp
import clip


class ModifiedResNet(nn.Module):
    """
    ResNet encoder-decoder architecture. Passes batches of images through CLIP's
    encoder network which then gets upsampled back to original resolution
    for classification through the use of skip connections.
    Args:
        clip_model (CLIP.Object): A loaded clip network.
        num_classes (Int): An integer value representing number of classes + background.
    """
    def __init__(self, clip_model, num_classes=4):
        super().__init__()
        self.clip_model = clip_model
        self.width = 128
        self.num_classes = num_classes
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
        
        self.final_conv = nn.Conv2d(self.width//4, self.num_classes, kernel_size=1)
        self.final_depth = nn.Conv2d(self.width//4, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        for param in self.clip_model.parameters(): #Freezing clip parameters
            param.requires_grad = False

    def forward(self,x):
        with torch.no_grad():
            x1 = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
            x1 = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x1)))
            x1 = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x1)))
            x1 = self.clip_model.visual.avgpool(x1)

            x2 = self.clip_model.visual.layer1(x1)
            x3 = self.clip_model.visual.layer2(x2)
            x4 = self.clip_model.visual.layer3(x3)
            x5 = self.clip_model.visual.layer4(x4)

        #Residual blocks
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

        x = self.final_conv(x)
        return x



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