import numpy as np
import torch
import torch.nn as nn
import torch.amp
import torch.optim as optim
import clip
import os
import glob
from tensorboardX import SummaryWriter
from Models import Checkpoint
from Models.ModifiedResNetSkip import *
from Utils.Utils import AugClass


class SVM(nn.Module):
    def __init__(self, device, model, dataloader, log_dir=None, model_dir=None, val_dataloader=None, optimizer = None, multi=False, multi_log = None, scores=[0,0]):
        super().__init__()
        self.scores = scores
        self.device = device
        self.modres = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.multi_log = multi_log
        self.multi = multi
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.aug = AugClass()

        if multi is False:
            self.writer = SummaryWriter(self.log_dir)
        elif multi_log is not None:
            self.writer = SummaryWriter(self.multi_log)
        self.save_counter = 0

    def train(self, epochs):
        num_epochs = epochs
        scaler = torch.amp.GradScaler(self.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)
        best_mean_iou = self.scores[0]
        best_mAP = self.scores[1]

        for epoch in range(num_epochs):
            if epoch == num_epochs-1:
                self.save_counter=1
            self.modres.train()
            lr_log = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr_log, epoch)
            seg_loss = 0

            for i, (img, mask) in enumerate(self.dataloader):
                img = img.to(self.device)
                mask = mask.to(self.device)
                aug_img, aug_mask = [],[]

                self.optimizer.zero_grad()
                with torch.autocast(self.device): #Autocasting as CLIP uses half precision
                    for img_b, mask_b in zip(img,mask):
                        img_np = img_b.permute(1,2,0)
                        img_r, mask_r = self.aug.augment(np.array(img_np.cpu()),np.array(mask_b.cpu()))
                        aug_img.append(torch.from_numpy(img_r))
                        aug_mask.append(torch.from_numpy(mask_r))
                    img = torch.stack(aug_img).permute(0,3,1,2)
                    img = img.contiguous()
                    img = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(img.to(self.device))
                    mask = torch.stack(aug_mask)
                    mask = mask.to(self.device).long().contiguous()
                    img_feat = self.modres(img)
                    img_feat_view = img_feat.view([*img_feat.size()[:-2],-1])
                    mask_view = mask.view(*[mask.size()[0],-1])

                    loss = self.criterion(img_feat_view, mask_view)
                    
                    del img_feat
                    del img_feat_view

                    temp_seg_loss = loss.cpu()
                    seg_loss += temp_seg_loss.data.numpy()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.modres.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                del loss, mask

            scheduler.step()
            self.writer.add_scalar('Train/Loss', seg_loss, epoch+1)
            self.writer.add_scalar('Train/Loss_batch', seg_loss/len(self.dataloader), epoch+1)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {seg_loss:.4f}, Best_mean_Iou: {best_mean_iou}, Best_mAP: {best_mAP}')

            #check val loss
            with torch.no_grad():
                with torch.autocast(self.device):
                    mean_iou, mAP = self.val_score(epoch)
                    print(f'Epoch {epoch+1}/{num_epochs}, Mean IoU: {mean_iou:.4f}, mAP: {mAP:.4f}')
                    self.writer.add_scalar('Eval/mean_iou', mean_iou, epoch+i)
                    self.writer.add_scalar('Eval/mAP', mAP, epoch+1)
                    
                    #Only used in Stage 1 if training Double
                    if self.multi==None:
                        if mean_iou > best_mean_iou or mAP > best_mAP:
                            if mean_iou > best_mean_iou:
                                best_mean_iou = mean_iou
                            if mAP > best_mAP:
                                best_mAP = mAP
                            if best_mean_iou > mean_iou and best_mAP > mAP: #Only saves if both are higher
                                files = glob.glob(os.path.join(self.model_dir, "Single"))
                                for file in files:
                                    os.remove(file)
                                torch.save(
                                    {'model_state_dict':self.modres.state_dict(), 
                                            'optimizer_state_dict':self.optimizer.state_dict(),
                                            'Segmentation_Scores':[best_mean_iou,best_mAP]
                                            }, 
                                            os.path.join(self.model_dir, "Single", 'Single_Model.pt'))
        self.writer.close()   
    

    def val_score(self, e):
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
            img_v = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(img_v)
            img_v, _ = self.modres(img_v)
            img_eval_view = img_v.view([*img_v.size()[:-2],-1])
            mask_eval_view = mask_v.view(*[mask_v.size()[0],-1])
            if check_for_nans(img_v, "nans in val imgs") == True:
                continue
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
            