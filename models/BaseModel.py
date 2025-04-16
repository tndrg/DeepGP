# -*- coding: utf-8 -*-
"""
@author: Taiyu Zhu
"""

import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
from torcheval.metrics.functional import binary_accuracy,r2_score,binary_auroc

class BaseRegression(pl.LightningModule):

    def __init__(self,configs):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()
        self.lr = configs.lr
        self.opt = nn.Identity()

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        return loss
    
    def training_step(self, batch, batch_nb):
        loss = self.evaluate(batch, 'train')
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        # validation loop
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        # testing loop
        self.evaluate(batch, 'test')

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred,y
    
    

class BaseRegressionCov(BaseRegression):

    def __init__(self,configs):
        super().__init__(configs)

    
    def evaluate(self, batch, stage=None,dataloader_idx=0):
        x_all, y = batch
        y_pred = self(x_all)
        loss = self.criterion(y_pred, y)
        acc = r2_score(y_pred[:,0], y[:,0])
        if stage:
            self.log(f'{stage}_loss_{dataloader_idx}', loss, prog_bar=True)
            self.log(f'{stage}_r2_{dataloader_idx}', acc, prog_bar=True)
        return  loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # validation loop
        self.evaluate(batch, 'val', dataloader_idx)


class BaseBinary(pl.LightningModule):

    def __init__(self,configs):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCELoss()
        # self.acc = BinaryAccuracy()
        self.lr = configs.lr
        self.opt = nn.Sigmoid()

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_pred_proba = self(x)
        loss = self.criterion(y_pred_proba, y)
        acc = binary_accuracy(torch.squeeze(y_pred_proba), torch.squeeze(y))
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
            self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_nb):
        loss = self.evaluate(batch, 'train')
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
              # validation loop
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        # testing loop
        self.evaluate(batch, 'test')

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred_proba = self(x)
        y_pred = (y_pred_proba > 0.5).int()
        return y_pred_proba,y_pred,y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class BaseBinaryCov(BaseBinary):

    def __init__(self,configs):
        super().__init__(configs)

    
    def evaluate(self, batch, stage=None):
        x_all, y = batch
        y_pred_proba = self(x_all)
        loss = self.criterion(y_pred_proba, y)
        acc = binary_accuracy(y_pred_proba[:,0], y[:,0])
        auc = binary_auroc(y_pred_proba[:,0], y[:,0])
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True,sync_dist=True)
            self.log(f'{stage}_acc', acc, prog_bar=True,sync_dist=True)
            self.log(f'{stage}_auc', auc, prog_bar=True,sync_dist=True)
        return  loss
    
    def predict_step(self, batch, batch_idx):
        x_all, y = batch
        y_pred_proba = self(x_all)
        y_pred = (y_pred_proba > 0.5).int()
        return y_pred_proba,y_pred,y
