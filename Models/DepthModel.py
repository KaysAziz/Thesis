import numpy as np
import torch
import torch.nn as nn
import torch.amp
import torch.optim as optim
import clip
import os
import glob
from tensorboardX import SummaryWriter
from Models.Checkpoint import Checkpoint
from Models.MultiheadResnet import *
from Utils.Utils import *


class DepthModel(nn.Module):
    def __init__(self, device, model, dataloader, log_dir, model_dir=None, val_dataloader=None, optimizer = None, 
                 curr_time = None, clip_model = None, depth_bins=None, scores_seg=[0,0], scores_depth=[10e+10,10e+10,10e+10]):
        super().__init__()
        self.scores_seg = scores_seg
        self.scores_depth = scores_depth
        self.device = device
        self.clip_model = clip_model
        self.modres = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.time = curr_time
        self.optimizer = optimizer
        self.criterion_seg = nn.CrossEntropyLoss()
        self.criterion_depth = nn.SmoothL1Loss()
        #self.aug = AugClass()
        self.depth_bins = depth_bins

        self.writer = SummaryWriter(self.log_dir)


    def train(self, epochs):
        num_epochs = epochs
        scaler = torch.amp.GradScaler(self.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        best_mean_iou = self.scores_seg[0]
        best_mAP = self.scores_seg[1]
        best_mean_absolute_error = self.scores_depth[0]
        best_mean_squared_error = self.scores_depth[1]
        best_root_mean_squared_error = self.scores_depth[2]
        for epoch in range(num_epochs):
            self.modres.train()
            lr_log = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr_log, epoch)
            epoch_seg_loss = 0
            epoch_depth_loss = 0

            for i, (img, seg_mask, depth_mask) in enumerate(self.dataloader):
                img = img.to(self.device) #B,V,C,H,W
                #Taking 4 images instead of 6
                img_first3 = img[:,:3,:,:,:]
                img_5 = img[:,4,:,:,:].unsqueeze(1)
                img = torch.cat((img_first3, img_5), dim=1)
                seg_mask = seg_mask.to(self.device) #B,V,H,W
                seg_first3 = seg_mask[:,:3,:,:]
                seg_5 = seg_mask[:,4,:,:].unsqueeze(1)
                seg_mask = torch.cat((seg_first3, seg_5), dim=1)
                depth_mask = depth_mask[:,0,:,:].to(self.device) #B,H,W only taking the reference view
                B, V, C, H, W = img.size()

                img = img.view(B*V,C,H,W)

                self.optimizer.zero_grad()
                with torch.autocast(self.device): #Autocasting as CLIP uses half precision
                    
                    img_feat, depth_feat = self.modres(img)
                    seg_mask_view = seg_mask.view(B*V,H,W)

                    #Splitting up losses before summming
                    loss_seg = self.criterion_seg(img_feat, seg_mask_view.long())
                    self.writer.add_scalar('Seg_Loss/train', loss_seg, i+1),

                    #preparing depth candidates to fit depth bins
                    prob_volume = torch.softmax(depth_feat.squeeze(1), dim=1) #depth_feat size B,256,448,448
                    predicted_depth = torch.sum(prob_volume * self.depth_bins, dim=1)  # depth_bins size B,256,448,448

                    loss_depth = self.criterion_depth(predicted_depth, depth_mask) 
                    self.writer.add_scalar('Depth_Loss/train', loss_depth, i+1)
                    del img_feat
                    del prob_volume
                    del predicted_depth

                    #Using .cpu() to properly detach and be able to add to total loss
                    temp_seg_loss = loss_seg.cpu()
                    epoch_seg_loss += temp_seg_loss.data.numpy()

                    temp_depth_loss = loss_depth.cpu()
                    epoch_depth_loss += temp_depth_loss.data.numpy()

                    total_loss = loss_seg + loss_depth
                    self.writer.add_scalar('Total_Loss/train', total_loss, i+1)
                scaler.scale(total_loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.modres.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                del loss_seg, loss_depth, total_loss

            scheduler.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Seg_Loss: {epoch_seg_loss:.4f}, Depth_Loss: {epoch_depth_loss:.4f}')
            #check val loss
            with torch.no_grad():
                with torch.autocast(self.device):
                    mean_iou, mAP = self.val_score(epoch)
                    self.writer.add_scalar('Eval/mean_iou', mean_iou, epoch+1)
                    self.writer.add_scalar('Eval/mAP', mAP, epoch+1)
                    print(f'Epoch {epoch+1}/{num_epochs}, Mean IoU: {mean_iou:.4f}, mAP: {mAP:.4f}')
                    if mean_iou > best_mean_iou or mAP > best_mAP:
                        if mean_iou > best_mean_iou:
                            best_mean_iou = mean_iou
                        if mAP > best_mAP:
                            best_mAP = mAP
                    if best_mean_iou > mean_iou and best_mAP > mAP:
                        files = glob.glob(f"{os.path.join(self.model_dir, 'Depth')}/Segmentation*")
                        for file in files:
                            os.remove(file)
                        torch.save(
                            {'model_state_dict':self.modres.state_dict(), 
                                    'optimizer_state_dict':self.optimizer.state_dict(),
                                    'Segmentation_Scores':[best_mean_iou,best_mAP],
                                    }, 
                                    os.path.join(self.model_dir, 'Segmentation.pt'))
                            
            with torch.no_grad():
                with torch.autocast(self.device):
                    mean_absolute_error, mean_squared_error, root_mean_squared_error = self.depth_val()
                    self.writer.add_scalar('Eval/mean_absolute_error', mean_absolute_error, epoch+1)
                    self.writer.add_scalar('Eval/mean_squared_error', mean_squared_error, epoch+1)
                    self.writer.add_scalar('Eval/root_mean_squared_error', root_mean_squared_error, epoch+1)
                    print(f'Epoch {epoch+1}/{num_epochs}, MAE: {mean_absolute_error}, MSE: {mean_squared_error}, RMSE {root_mean_squared_error}')
                    if mean_absolute_error < best_mean_absolute_error:
                        best_mean_absolute_error = mean_absolute_error
                        files = glob.glob(f"{os.path.join(self.model_dir, 'Depth')}/Depth_MAE*")
                        for file in files:
                            os.remove(file)
                        torch.save(
                                {'model_state_dict':self.modres.state_dict(), 
                                        'optimizer_state_dict':self.optimizer.state_dict(),
                                        'Depth_Scores':[best_mean_absolute_error, best_mean_squared_error, best_root_mean_squared_error]
                                        },
                                        os.path.join(self.model_dir, 'Depth_Model_MAE.pt'))
                    if mean_squared_error < best_mean_squared_error:
                        best_mean_squared_error = mean_squared_error
                    if root_mean_squared_error < best_root_mean_squared_error:
                        best_root_mean_squared_error = root_mean_squared_error
                        files = glob.glob(f"{os.path.join(self.model_dir, 'Depth')}/Depth_RMSE*")
                        for file in files:
                            os.remove(file)
                        torch.save(
                                {'model_state_dict':self.modres.state_dict(), 
                                        'optimizer_state_dict':self.optimizer.state_dict(),
                                        'Depth_Scores':[best_mean_absolute_error, best_mean_squared_error, best_root_mean_squared_error]
                                        },
                                        os.path.join(self.model_dir, 'Depth_Model_RMSE.pt'))
        self.writer.close()   
    

    def val_score(self,e):
        self.modres.eval()

        thresholds = torch.linspace(0,1,50)
        val_len = len(self.val_dataloader)
        class_len = torch.Tensor([0, 1e-10, 1e-10, 1e-10])
        precision_at_thresholds = torch.zeros((4, len(thresholds)))
        recall_at_thresholds = torch.zeros((4, len(thresholds)))
        iou_at_thresholds = torch.zeros((4, len(thresholds)))
        eval_loss = 0
        for val_epoch, (img_v, seg_mask_gt, _) in enumerate(self.val_dataloader):
            img_v, seg_mask_gt, = img_v.to(self.device), seg_mask_gt.to(self.device)
            
            img_v_first3 = img_v[:,:3,:,:,:]
            img_v_5 = img_v[:,4,:,:,:].unsqueeze(1)
            img_v = torch.cat((img_v_first3, img_v_5), dim=1)

            seg_mask_first3 = seg_mask_gt[:,:3,:,:]
            seg_mask_5 = seg_mask_gt[:,4,:,:].unsqueeze(1)
            seg_mask_gt = torch.cat((seg_mask_first3, seg_mask_5), dim=1)

            B, V, C, H, W = img_v.size()
            img_v = img_v.view(-1,C,H,W)
            img_eval_view = img_v.view(B*V,4,-1)
            img_v, _ = self.modres(img_v)
            if check_for_nans(img_v, "nans in val imgs") == True:
                continue
            mask_eval_view = seg_mask_gt.view(B*V,-1)
            seg_mask_gt = seg_mask_gt.view([B*V,H,W])
            e_loss = self.criterion_seg(img_eval_view, mask_eval_view.long())
            e_loss = e_loss.cpu()
            eval_loss += e_loss.data.numpy()
            pred_softmax = torch.softmax(img_v, dim = 1)
            show_pred, pred = torch.max(pred_softmax, dim = 1)
            
            for i in range(3):
                contains_class = (seg_mask_gt == i+1).any(dim=(1,2))
                mask_i = seg_mask_gt[contains_class]
                predi_softmax = pred_softmax[contains_class]
                if len(mask_i) == 0:
                    continue
                class_len[i+1] += 1
                for threshold_step, t in enumerate(thresholds):
                    binarized_preds = (predi_softmax[:, i+1, :, :] > t)
                    tp = ((binarized_preds == True) & (mask_i == i+1)).sum().item()
                    fp = ((binarized_preds == True) & (mask_i != i+1)).sum().item() #incorrect class i guess in non i class pixels
                    fn = ((binarized_preds == False) & (mask_i == i+1)).sum().item() #incorrect other class or backgorund guess in class i pixels
                    union = ((binarized_preds == True) | (mask_i == i+1)).sum().item()
                    try:
                        iou = tp/union
                        precision = tp/(tp+fp)
                        recall = tp/(tp+fn)
                    except:
                        iou = 0
                        precision = 0
                        recall = 0

                    precision_at_thresholds[i+1,threshold_step] += precision
                    recall_at_thresholds[i+1,threshold_step] += recall
                    iou_at_thresholds[i+1,threshold_step] += iou

                    if t > 0.95:
                        print(f"AT EPOCH: {e}. TP: {tp}, FP: {fp}, FN: {fn}") #Make saving logic for testing
                        if tp > 50000:
                           print(f"IN EVAL!! TP: {tp}, FP:{fp}, FN:{fn}, epoch:{val_epoch}, t:{t}")
                           torch.save(pred, f"/work3/s215961/saved_tens/Depth/Seg_Pred_{self.time}epoch{val_epoch}_class{i+1}_t{t}_TP{tp}.pt")
                           torch.save(seg_mask_gt, f"/work3/s215961/saved_tens/Depth/Seg_Mask_{self.time}epoch{val_epoch}_class{i+1}_t{t}_TP{tp}.pt")

            
        class_len = class_len.unsqueeze(1)
        precision_at_thresholds[1:] /= class_len[1:]
        recall_at_thresholds[1:] /= class_len[1:]
        iou_at_thresholds[1:] /= class_len[1:]
        
        ap = torch.zeros(4)
        for i in range(3):
            recall_indices = torch.argsort(recall_at_thresholds[i+1])
            recall_sorted = recall_at_thresholds[i+1][recall_indices]
            precision_sorted = precision_at_thresholds[i+1][recall_indices]
            ap[i+1] = self.simp(recall_sorted, precision_sorted)
        
        mean_iou = torch.mean(iou_at_thresholds[1:])
        mAP = torch.mean(ap[1:])
        self.writer.add_scalar('Eval/Loss', eval_loss, e+1)
        self.writer.add_scalar('Eval/Loss_batch', eval_loss/val_len, e+1)
        return mean_iou, mAP

    def simp(self, x, y):
        n = len(x)-1

        h = ((x[-1]-x[0])/n) /3
        sums = y[0] + y[-1]
        for i in range(1,n, 2):
            sums+=4*y[i]
        for i in range(2,n, 2):
            sums+=2*y[i]

        sums = sums*h

        return sums
            
    def depth_val(self):
        self.modres.eval()

        total_abs_error = 0
        total_squared_error = 0
        total_samples = 0


        for i, (img, _, depth_gt) in enumerate(self.val_dataloader):
            img = img.to(self.device)
            img_fd = img[:,:3,:,:,:]
            img_ld = img[:,4,:,:,:].unsqueeze(1)
            img = torch.cat((img_fd, img_ld), dim=1)
            depth_gt = depth_gt[:,0,:,:].to(self.device)
            B, V, C, H, W = img.size()
            img = img.view(B*V,C,H,W)

            _, depth_feat = self.modres(img)

            prob_volume = torch.softmax(depth_feat.squeeze(1), dim=1)
            predicted_depth = torch.sum(prob_volume * self.depth_bins, dim=1)

            abs_error = torch.abs(predicted_depth - depth_gt)
            squared_error = (predicted_depth - depth_gt) ** 2

            total_abs_error += abs_error.sum().item()
            total_squared_error += squared_error.sum().item()
            total_samples += depth_gt.numel()

        mean_absolute_error = total_abs_error / total_samples
        mean_squared_error = total_squared_error / total_samples
        rmse = mean_squared_error ** 0.5

        return mean_absolute_error, mean_squared_error, rmse



