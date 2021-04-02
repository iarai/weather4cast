# Author: Pedro Herruzo
# Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch

from benchmarks.validation_metrics import LeadTimeEval
from benchmarks.unet import UNet

class FeaturesSysUNet(pl.LightningModule):
    def __init__(self, UNet_params: dict, extra_data: str, depth: int, height: int, 
                 width: int, len_seq_in: int, len_seq_out: int, bins_to_predict: int, 
                 seq_mode: str, **kwargs):
        super(FeaturesSysUNet, self).__init__()

        
        self.save_hyperparameters()
        self.model = UNet(**UNet_params)
        self.extra_data = extra_data
        self.depth = depth
        self.height = height
        self.width = width
        self.len_seq_in = len_seq_in
        self.len_seq_out = len_seq_out
        self.bins_to_predict = bins_to_predict
        self.seq_mode = seq_mode
        
        self.target_vars = kwargs['target_vars']
        self.main_metric = 'mse'
        
        self.leadTeval = LeadTimeEval(len_seq_in, bins_to_predict, len(self.target_vars))
        self.prec = 7

        loss = 'mse'
        self.loss_fn = {'smoothL1': nn.SmoothL1Loss(), 'L1': nn.L1Loss(), 'mse': F.mse_loss}
        self.loss_fn = self.loss_fn[loss]
    
    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        metric_placeholder = { metric: -1 for metric in self.target_vars}
        metric_placeholder = {**metric_placeholder, **{self.main_metric: -1}}
        self.logger.log_hyperparams(self.hparams, metric_placeholder)
        
    def forward(self, x):
        return self.model(x)
    
    def _compute_loss(self, y_hat, y, agg=True):

        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss
    
    def training_step(self, batch, batch_idx, phase='train'):

        x, y, *ignored  = batch
        y_hat = self.forward(x)
                
        loss = self._compute_loss(y_hat, y)
        self.log(f'{phase}_loss', loss)
        return loss
                
    def validation_step(self, batch, batch_idx, phase='val'):
        
        x, y, *ignored  = batch
        y_hat = self.forward(x)

        loss = self._compute_loss(y_hat, y)
        return loss

    def validation_epoch_end(self, outputs, phase='val'):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log(f'{phase}_loss_epoch', avg_loss, prog_bar=True)
        self.log(self.main_metric, avg_loss)
            
    def test_step(self, batch, batch_idx, phase='test'):

        x, y, metadata  = batch
        y_hat = self.forward(x)
                
        loss = self._compute_loss(y_hat, y, agg=False)
        self.log(f'{phase}_loss', loss.mean())
        
        # reduce spatial dims - keep batch & channels
        loss = loss.mean(dim=(-1, -2))#.detach().cpu().numpy()
        self.leadTeval.update_errors(loss.detach().cpu().numpy(), metadata)
        
        # reduce batch - keep channels
        loss = loss.mean(dim=(0)) 
        return loss
    
    def test_epoch_end(self, outputs, phase='test'):
        # ------------
        # 1. Test Channels
        # ------------
        # concat batches and compute the mean among them preserving channels
        val_set_loss = torch.stack(outputs, dim=0)
        val_set_loss = val_set_loss.mean(dim=(0))
        val_set_loss = self.leadTeval.get_numpy(val_set_loss)
        
        # log metrics
        text = f'extra data: {self.extra_data} | '
        for i, channel in enumerate(self.target_vars):
            self.log(channel, val_set_loss[i])
            v = np.format_float_positional(val_set_loss[i], precision=self.prec)
            text += f'{channel}: {v} | '
        
        v = np.format_float_positional(val_set_loss.mean(), precision=self.prec)
        text += f'mse: {v}'
        print(text)
        self.log(self.main_metric, val_set_loss.mean())
        
        # ------------
        # 2. Test Lead Time
        # ------------
        #fname = f'{self.logger.log_dir}/lead_times_mse.csv'
        region_id = self.test_dataloader().dataset.region_id
        ers, std = self.leadTeval.get_lead_time_metrics(self.logger.log_dir, text, region_id)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        
