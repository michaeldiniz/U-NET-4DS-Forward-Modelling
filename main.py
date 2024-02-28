# ---------------
# GENERAL IMPORTS
# ---------------

from os.path import join as pjoin
import argparse
import torch
import pytorch_lightning as pl

# -------------
# LOCAL IMPORTS
# -------------

from Unet_classes import U_net_46x46, U_net_94x78
from load_data import *

# -------------------------
# PYTORCH LIGHTNING IMPORTS
# -------------------------

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger as TBLogger

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)



def train_u_net(out_resolution, gpu, workers, model_prefix,
              model_params, ckpt_path = None, layer='L1'):
    ckpt_file = model_prefix + '{epoch:02d}_{step}'
      
    seed_everything(42) # Common state for generating reproducible results
    dm = ShellDataModule(layer=layer, num_workers = workers, out_resolution=out_resolution)

    
    ckpt_params = {
        'dirpath':f'./checkpoint/{layer}',
        'monitor': 'val_loss',
        'filename': ckpt_file,
        'save_top_k': 1,
        'every_n_epochs': None
    }
    ckpt_callback = ModelCheckpoint(**ckpt_params)
    
    # Will save all logs in path ./lightning_logs/s4d/ <model_prefix>
    logger = TBLogger("./lightning_logs", name = 's4d', version = model_prefix)

    # General settings for training / validation
    # NOTE: If you wish to run on CPU, replace the second entry with 'gpus': 0
    train_params = {
        'max_epochs':200,
        'gpus': [gpu],
        'accelerator':'auto',
        'devices':1,
        'check_val_every_n_epoch': 1,
        'deterministic': True,
        'callbacks': [ckpt_callback],
        'logger': logger
    }
    
    trainer = pl.Trainer(**train_params)
    
    dm.setup()

    
    # If no checkpoint is provided, will train the U-net from scratch
    if ckpt_path is None:
        if out_resolution == '46x46':  
            model = U_net_46x46(**model_params )
        else:
            model = U_net_94x78(**model_params)
        trainer.fit(model, dm)

        
    # Otherwise, it only loads the checkpoint into an instance of a U-net class
    else:
        print('checkpoint file não é None')
        if out_resolution == '46x46':  
            model = U_net_46x46.load_from_checkpoint(ckpt_path,**model_params )
        else:
            model = U_net_94x78.load_from_checkpoint(ckpt_path,**model_params )
        dm.setup(layer)
        
    trainer.test(model, datamodule = dm)
    return model, dm

def main(args):
    # Parsing arguments from command line
    gpu = args.gpu
    workers = args.workers
    ckpt_path = args.cp
    layer=args.l
    out_resolution=args.r
    
    # Naming standard for all outputs (plots, scores, checkpoints, ...)
    model_prefix = f'Unet_{out_resolution}_{layer}'

    # -----------------------------------
    # FEEL FREE TO CUSTOMIZE OUTPUT PATHS
    # -----------------------------------
    
    tensor_path = './tensors'
    
    if out_resolution =='46x46':  
        data_points_path=fr'./dataset/tensores/{layer}_features/data_points.csv'
    else:
        data_points_path=fr'./dataset/tensores/{layer}_features/dRMS_data_points.csv'

    model_params = {
        'tensor_path': tensor_path,
        'data_points': data_points_path
    }

    # Training
    model, dm = train_u_net(out_resolution, gpu, workers,
                          model_prefix, model_params, ckpt_path, layer=layer) 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = int, choices = list(range(8)), default = 0)
    parser.add_argument("--workers", type = int, default = 12)
    parser.add_argument("--cp", type = str, default = None)
    parser.add_argument("--l", type = str, choices=['L1','L2','L3'], default ='L1')
    parser.add_argument("--r", type = str, choices=['46x46','94x78'], default ='94x78')

    args = parser.parse_args()
    main(args)
