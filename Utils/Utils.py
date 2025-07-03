import albumentations as A
import cv2
import torch
import torch.nn as nn

class AugClass():
    def __init__(self) -> None:
        self.transform_both = A.Compose([
        A.OneOf([A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                    A.Flip(p=0.5),
                    A.Rotate(limit=(-45,45), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, crop_border=False),
                    A.VerticalFlip(p=0.5),
                    A.D4(p=0.8),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_NEAREST, p=.75),
                    ]),

        
        A.Perspective(scale=(0.05,0.1), keep_size=True,pad_mode=0,pad_val=0,mask_pad_val=0,fit_output=False,interpolation=cv2.INTER_NEAREST,p=0.5),

        ])
        self.transform_img = A.Compose([
                    A.OneOf(
                    [A.Blur(blur_limit=3),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=None, p=0.6),
                    A.Defocus(radius=(3, 10),alias_blur=(0.1, 0.5),p=0.5),
                    A.GlassBlur(sigma=0.7,max_delta=4,iterations=2,mode="fast",p=0.1),
                    A.GridDistortion(num_steps=5,distort_limit=(-0.3, 0.3), interpolation=3,border_mode=4,p=0.5),
                    A.MedianBlur(blur_limit=5,p=0.5),
                    A.MotionBlur(blur_limit=7,allow_shifted=True,p=0.4),
                    ]),

                    A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2,0.2),brightness_by_max=True,p=0.4),
                    A.RandomGamma(gamma_limit=(80,300), p=0.5),
                    A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5),src_radius=250, angle_lower=0.5, p=0.5)       
        
                    ])
    def augment(self, img, label):
        augmented_both = self.transform_both(image=img, mask=label)
        img = augmented_both["image"]
        augmented_img = self.transform_img(image=img)
        return augmented_img["image"], augmented_both["mask"]


def scale_intrinsics(intrinsics, orig_width, orig_height, new_width, new_height):
    """
    Scales the intrinsic matrix for a new resolution.

    Args:
        intrinsics (torch.Tensor): Original intrinsic matrix [B, 3, 3].
        orig_width (float): Original image width.
        orig_height (float): Original image height.
        new_width (float): New image width.
        new_height (float): New image height.

    Returns:
        torch.Tensor: Scaled intrinsic matrix [B, 3, 3].
    """
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    # Clone the intrinsic matrix to avoid modifying the original
    scaled_intrinsics = intrinsics.clone()
    scaled_intrinsics[:, 0, 0] *= scale_w  # Scale fx
    scaled_intrinsics[:, 1, 1] *= scale_h  # Scale fy
    scaled_intrinsics[:, 0, 2] *= scale_w  # Scale px
    scaled_intrinsics[:, 1, 2] *= scale_h  # Scale py

    return scaled_intrinsics


class MatchingVolume(nn.Module):
    def __init__(self):
        super().__init__()

    def create_volume(self, feats, trans_mat, intrinsics, depth_candidates=None, device="gpu", scaling=None):

        pixel_coords = self.set_id_grid(*feats.size()[-2:]).to(device)
        f = self.get_volume(feats, depth_candidates, intrinsics, trans_mat, pixel_coords, scaling=scaling)
        return f


    def set_id_grid(self, h, w):
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).to(torch.float32)
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).to(torch.float32)
        ones = torch.ones(1, h, w).to(torch.float32)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)
        return pixel_coords

    def get_volume(self, feat, depth_values, intrinsics, transforms, pixel_coords, scaling = None):

        views, B, C, H, W = feat.size()
        min_depth = depth_values[0]
        max_depth = depth_values[1]
        if scaling is None:
            num_candidates = depth_values[2]
        else:
            num_candidates = depth_values[2]*scaling
        depth_interval = (max_depth - min_depth) / (num_candidates - 1)

        #Create depth candidates
        depth_cands = torch.arange(0, num_candidates, requires_grad=False).reshape(1, -1).to(
            torch.float32) * depth_interval + min_depth
        depth_candidates_norm = 2 * ((depth_cands - min_depth) / depth_interval) / (num_candidates - 1) - 1
        depth_candidates = depth_candidates_norm.view(num_candidates, 1, 1).expand(num_candidates, H, W).to(feat.device)
        depth_candidates_norm = depth_candidates.unsqueeze(0).repeat(B,1,1,1) #B,num_candidates,H,W
        ref_feat = feat[0]

        #Scale intrinsics to current resolution of features
        intrinsics = scale_intrinsics(intrinsics, 448, 448, H, W)

        extrinsics_ref = torch.inverse(transforms[:,0,:,:]) #trans: B,V,4,4 -> B,4,4

        pix_coords = pixel_coords.repeat(B, 1, 1, 1).view(B, 3, -1) #B,3,H*W

        _, DC, _, _ = depth_candidates_norm.size() # depth [B, Ndepth, H, W]

        ref_volume = ref_feat.unsqueeze(2).repeat(1,1,DC,1,1)
        cost_volume = torch.zeros_like(ref_volume).to(ref_volume.dtype).to(ref_volume.device)

        with torch.no_grad():
            for supp_view in range(views):
                if supp_view == 0:
                    continue
                
                extrinsics_supp = torch.inverse(transforms[:,supp_view,:,:]) #trans: B,V,4,4 -> B,4,4
                supp_feat = feat[supp_view] # B,C,H,W

                supp_proj = extrinsics_supp.clone()
                supp_proj[:,:3,:] = torch.matmul(intrinsics, extrinsics_supp[:,:3,:])               


                ref_proj = extrinsics_ref.clone()
                ref_proj[:,:3,:] = (torch.matmul(intrinsics, extrinsics_ref[:,:3,:])).clone()
 
                full_proj = torch.matmul(supp_proj, torch.inverse(ref_proj))

                rot = full_proj[:,:3,:3]
                trans = full_proj[:,:3,3:4]

                rot_pix = torch.matmul(rot, pix_coords) # B,3,H*W

                pix_depth = rot_pix.unsqueeze(2).repeat(1, 1, DC, 1) * depth_candidates_norm.view(B, 1, DC, -1) # B,3,dc,H*W
                trans_pix = pix_depth + trans.view(B, 3, 1, 1) # B,3,dc,H*W                

                proj_2d = trans_pix[:,:2,:,:] / (trans_pix[:,2:3,:,:] + 1e-8) #B,2,dc,H*W
                proj_x_norm = 2 * (proj_2d[:,0,:,:] - proj_2d[:,0,:,:].min()) / (proj_2d[:,0,:,:].max() - proj_2d[:,0,:,:].min()) - 1
                proj_y_norm = 2 * (proj_2d[:,1,:,:] - proj_2d[:,1,:,:].min()) / (proj_2d[:,1,:,:].max() - proj_2d[:,1,:,:].min()) - 1

                grid = torch.stack((proj_x_norm, proj_y_norm), dim=3) #B,dc,H*W,2
                warped_volume = nn.functional.grid_sample(supp_feat, grid.view(B, DC*H, W, 2), mode='bilinear', padding_mode='zeros', align_corners=False).view(B,C,DC,H,W)
                x = torch.cat([ref_volume, warped_volume], dim=1)
                _, in_plane, _, _, _ = x.size()
                with torch.autocast("cuda", enabled=False):
                    conv = self.conv3d_func(in_plane).to("cuda")
                    x = conv(x)

                cost_volume = cost_volume + x

        cost_volume = cost_volume / (views - 1)
        return cost_volume

    def conv3d_func(self, C):
        """
        Applies a 3d convolution to reduce channels without changing
        spatial dimensions for application in residual block.
        Args:
            Tensor, in this case a warped volume of a reference and support view.
        """
        return torch.nn.Sequential(nn.Conv3d(C, C//2, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm3d(C//2),
                                   nn.ReLU(inplace=True))
