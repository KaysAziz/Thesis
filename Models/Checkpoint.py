import torch
import os, shutil

class Checkpoint:
    def create_folder(self, log_dir, stage=None):
        # make summary folder
        if stage is not None:
            stage_folder = f'{log_dir}/{stage}'
        else:
            stage_folder = log_dir
        if not os.path.exists(stage_folder):
            os.mkdir(stage_folder)
        else:
            print('WARNING: summary folder already exists!! It will be overwritten!!')
            shutil.rmtree(stage_folder)
            os.mkdir(stage_folder)

    def save_checkpoint(self, model, optimizer, filename):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, model, optimizer, filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])