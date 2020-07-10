# -*- encoding: utf-8 -*-
'''
@Filename    :train.py
@Time        :2020/07/10 23:23:18
@Author      :Kai Li
@Version     :1.0
'''

from option import parse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Lightning
import torch
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def Train(opt):
    # init Lightning Model
    light = Lightning(**opt['light_conf'])

    # mkdir the file of Experiment path
    os.makedirs(os.path.join(opt['resume']['path'],
                             opt['resume']['checkpoint']), exist_ok=True)
    checkpoint_path = os.path.join(
        opt['resume']['path'], opt['resume']['checkpoint'])
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor='val_loss', mode='min', save_top_k=1, verbose=1, save_last=True)

    # Early Stopping
    early_stopping = False
    if opt['train']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=opt['train']['patience'],
                                       mode='min', verbose=1)

    # Don't ask GPU if they are not available.
    if torch.cuda.is_available():
        gpus = len(opt['gpu_ids'])
    else:
        gpus = None
    # logger
    
    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir='./logger',
        version=1,
        name='lightning_logs'
    )
    # Trainer
    trainer = pl.Trainer(max_epochs=opt['train']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_root_dir=checkpoint_path,
                         gpus=gpus,
                         distributed_backend=opt['train']['distributed_backend'],
                         train_percent_check=1.0,  # Useful for fast experiment
                         gradient_clip_val=5.,
                         logger=logger)

    trainer.fit(light)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt)
