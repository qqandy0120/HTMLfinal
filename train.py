import numpy as np
import pandas as pd

# hparams
from opt import get_opts
# data
from dataset import BikeDataset
from torch.utils.data import DataLoader
# model
from models.gru import GRUBikePredictor
#optimizer
import torch.optim as optim
#scheduler
import torch.optim.lr_scheduler as lr_scheduler
# util
from util import *

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

MODES = ['normal', 'y_only']
class BikePredModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)  # call hparams to self
        
        assert self.hparams.mode in MODES, f'mode should be one of {MODES}!'

        self.feature_cnt = 12 if self.hparams.mode=='normal' else 1

        self.net = GRUBikePredictor(
            feature_cnt=self.feature_cnt,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
        )
        self.validation_step_outputs = []
    
    def forward(self, input):
        outputs = self.net(input)
        return outputs
    
    # set up dataset
    def setup(self, stage=None):
        print(f"------- CURRENT STATION ID: {self.hparams.station_id} --------")
        self.train_dataset = BikeDataset(self.hparams.station_id,'train', self.hparams.time_step, self.hparams.mode)
        self.valid_dataset = BikeDataset(self.hparams.station_id, 'valid', self.hparams.time_step, self.hparams.mode) 

    # pack up dataloader
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size
        )
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size
        )
    
    # define optimizer and scheduler
    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.optim)(
            self.net.parameters(),
            lr=self.hparams.lr
        )
        # if self.hparams.scheduler is not None:
        #     scheduler = getattr(lr_scheduler, self.hparams.scheduler)(optimizer, step_size=1)
        # else:
        #     scheduler = None
        # scheduler = None

        return [optimizer], []
    
    # training/validation step
    def training_step(self, batch, batch_idx):
        preds = self(batch['feature'])
        criterion = nn.MSELoss()
        loss = criterion(preds, batch['label'])
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds = self(batch['feature'])
        criterion = nn.MSELoss()
        loss = criterion(preds, batch['label'])
        self.validation_step_outputs.append(loss)
        self.log('val/loss', loss, prog_bar=True)
        return loss
    def on_validation_epoch_end(self):
        # calculate average loss
        mean_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val/loss', mean_loss, prog_bar=True)
    # for inference
    def predict_step(self, batch, batch_idx: int):
        preds = self(batch['feature'])
        return preds

if __name__ == '__main__':
    hparams = get_opts()
    hparams.ckpt_dir.mkdir(parents=True, exist_ok=True)
    hparams.log_dir.mkdir(parents=True, exist_ok=True)
    
    module = BikePredModule(hparams)

    ckpt_cb = ModelCheckpoint(
        dirpath=hparams.ckpt_dir / hparams.exp_name,
        filename='exp',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor='val/loss', mode='min', patience=hparams.patience)
    pbar = TQDMProgressBar(refresh_rate=1)  # show progress bar in terminal
    callbacks = [ckpt_cb, early_stop, pbar]

    logger = WandbLogger(
        project="HTML-Final-Project",
        save_dir=hparams.log_dir,
        name=hparams.exp_name,
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        log_every_n_steps=11,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=True,
        accelerator='auto',  # gpu? cpu? tpu?
        devices=1,
        num_sanity_val_steps=1,  # run validation step first to check correctness
        benchmark=True,
    )

    trainer.fit(module)