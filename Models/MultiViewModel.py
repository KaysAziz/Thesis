# import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
import torch.amp
# import clip
import os
import glob
from tensorboardX import SummaryWriter
# from Utils.Dataset_MultiView import VialDatasetMulti
from Models.ModifiedResNetSkip import *
from Utils.Utils import AugClass
# import albumentations as A


class MVM(nn.Module):
    def __init__(self, device, model, dataloader, log_dir, model_dir=None, val_dataloader=None, optimizer=None,
                model_view = False, scores=[0,0]):
        super().__init__()
        self.scores = scores
        self.device = device
        self.modres = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.model_view = model_view
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.aug = AugClass()

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)
        self.save_counter = 0

    def train(self, epoch):
        scaler = torch.amp.GradScaler(self.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)

        num_epochs = epoch
        best_mean_iou = self.scores[0]
        best_mAP = self.scores[1]
        for epoch in range(num_epochs):
            if epoch == num_epochs-1:
                self.save_counter = 1
            self.modres.train()
            lr_log = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr_log, epoch)
            seg_loss = 0
            for i, batch in enumerate(self.dataloader):
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad()
                with torch.amp.autocast(self.device): #Autocasting as CLIP uses half precision
                    B, V, C, H, W = images.size()
                    images = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(images)
                    images = images.view(-1,C,H,W)
                    img_feat = self.modres(images)
                    img_feat_view = img_feat.view((B*V, 4, -1))

                    mask_view = masks.view([B*V,-1])
                    loss = self.criterion(img_feat_view, mask_view)
                    
                    temp_seg_loss = loss.cpu()
                    seg_loss += temp_seg_loss.data.numpy()

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.modres.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                del img_feat
                del img_feat_view
                del mask_view
                del loss
                del masks
            scheduler.step()
            self.writer.add_scalar('Train/Loss', seg_loss, epoch+1)
            self.writer.add_scalar('Train/Loss_batch', seg_loss/len(self.dataloader), epoch+1)

            #check val loss
            with torch.no_grad():
                with torch.autocast(self.device):
                    mean_iou, mAP = self.val_score(epoch)
                    self.writer.add_scalar('Eval/mean_iou', mean_iou, epoch+1)
                    self.writer.add_scalar('Eval/mAP', mAP, epoch+1)
                    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {seg_loss.item():.4f}, Mean IoU: {mean_iou:.4f}, mAP: {mAP:.4f}')
                    if mean_iou > best_mean_iou or mAP > best_mAP:
                        if mean_iou > best_mean_iou:
                            best_mean_iou = mean_iou
                        if mAP > best_mAP:
                            best_mAP = mAP
                        if self.model_view == False and best_mean_iou > mean_iou and best_mAP > mAP:
                            files = glob.glob(os.path.join(self.model_dir, 'Multi'))
                            for file in files:
                                os.remove(file)
                            torch.save(
                                {'model_state_dict':self.modres.state_dict(), 
                                        'optimizer_state_dict':self.optimizer.state_dict(),
                                        'Segmentation_Scores':[best_mean_iou,best_mAP]
                                        }, 
                                        os.path.join(self.model_dir, 'Multi' 'Multi_Model.pt'))
                        elif self.model_view == True and best_mean_iou > mean_iou and best_mAP > mAP:
                            files = glob.glob(os.path.join(self.model_dir, 'Double'))
                            for file in files:
                                os.remove(file)
                            torch.save(
                                {'model_state_dict':self.modres.state_dict(), 
                                        'optimizer_state_dict':self.optimizer.state_dict(),
                                        'Segmentation_Scores':[best_mean_iou,best_mAP]
                                        }, 
                                        os.path.join(self.model_dir, 'Double' 'Double_Model.pt'))
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
        for val_epoch, (img_v, mask_v) in enumerate(self.val_dataloader):
            img_v, mask_v = img_v.to(self.device), mask_v.to(self.device)
            B, V, C, H, W = img_v.size()
            img_v = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(img_v) 
            img_v = img_v.view(-1,C,H,W)
            img_v, _ = self.modres(img_v)
            img_eval_view = img_v.view(B*V,4,-1)
            if check_for_nans(img_v, "nans in val imgs") == True:
                continue
            mask_eval_view = mask_v.view(B*V,-1)
            mask_v = mask_v.view([B*V,H,W])
            e_loss = self.criterion(img_eval_view, mask_eval_view)
            e_loss = e_loss.cpu()
            eval_loss += e_loss.data.numpy()
            pred_softmax = torch.softmax(img_v, dim = 1)
            show_pred, pred = torch.max(pred_softmax, dim = 1)
            
            for i in range(3):
                contains_class = (mask_v == i+1).any(dim=(1,2))
                mask_i = mask_v[contains_class]
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
