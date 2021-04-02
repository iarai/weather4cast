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

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from w4c_dataloader import create_dataset

import pathlib
import sys
import os
module_dir = str(pathlib.Path(os.getcwd()).parent)
sys.path.append(module_dir)

import config as cf
from benchmarks.FeaturesSysUNet import FeaturesSysUNet as Model

class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params):
        super().__init__()
        self.params = params     
        self.training_params = training_params

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset, 
                        batch_size=self.training_params['batch_size'], num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, pin_memory=pin, prefetch_factor=2, persistent_workers=False)
        return dl
    
    def train_dataloader(self):
        ds = create_dataset('training', self.params)
        return self.__load_dataloader(ds, shuffle=True, pin=True)

    def val_dataloader(self):
        ds = create_dataset('validation', self.params)
        return self.__load_dataloader(ds, shuffle=False, pin=True)

def print_training(params):
    """ print pre-training info """
    
    print(f'Extra variables: {params["extra_data"]} | spatial_dim: {params["spatial_dim"]} ', 
          f'| collapse_time: {params["collapse_time"]} | in channels depth: {params["depth"]} | len_seq_in: {params["len_seq_in"]}')

def load_model(Model, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    if checkpoint_path == '':
        print('-> model from scratch!')
        model = Model(params['model_params'], **params['data_params'])            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path)
    return model

def get_trainer(gpu):
    """ get the trainer, modify here it's options:
        - save_top_k
        - max_epochs
     """
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=3, 
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')
    
    trainer = pl.Trainer(gpus=[gpu], max_epochs=20,
                         progress_bar_refresh_rate=80,
                         callbacks=[checkpoint_callback], 
                         profiler='simple',
                        )
    return trainer

def do_test(trainer, model, test_data):
    print("-----------------")
    print("--- TEST MODE ---")
    print("-----------------")
    scores = trainer.test(model, test_dataloaders=test_data)
    
def train(gpu, region_id, mode, checkpoint_path):
    """ main training/evaluation method
    """
    # ------------
    # model & data
    # ------------
    params = cf.get_params(region_id=region_id, collapse_time=True)
    data = DataModule(params['data_params'], params['training_params'])
    model = load_model(Model, params, checkpoint_path)
    
    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpu)
    print_training(params['data_params'])
    
    # ------------
    # train & final validation
    # ------------
    if mode == 'train':
        print("-----------------")
        print("-- TRAIN MODE ---")
        print("-----------------")
        trainer.fit(model, data)
    
    # validate
    do_test(trainer, model, data.val_dataloader()) 
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--gpu_id", type=int, required=False, default=1, 
                        help="specify a gpu ID. 1 as default")
    parser.add_argument("-r", "--region", type=str, required=False, default='R1', 
                        help="region_id to load data from. R1 as default")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")

    return parser

def main():
    
    parser = set_parser()
    options = parser.parse_args()

    train(options.gpu_id, options.region, options.mode, options.checkpoint)

if __name__ == "__main__":
    main()
    """ examples of usage:

    cd utils

    - a.1) train from scratch
    python 3-train-UNet-example.py --gpu_id 1 --region R1

    - a.2) fine tune a model from a checkpoint
    python 3-train-UNet-example.py --gpu_id 1 --region R1 -c '~/projects/weather4cast/lightning_logs/version_21/checkpoints/epoch=03-val_loss_epoch=0.027697.ckpt'

    - b.1) evaluate an untrained model (with random weights)
    python 3-train-UNet-example.py --gpu_id 1 --region R1 --mode val

    - b.2) evaluate a trained model from a checkpoint
    python 3-train-UNet-example.py --gpu_id 1 --region R1 --mode val -c '~/projects/weather4cast/lightning_logs/version_21/checkpoints/epoch=03-val_loss_epoch=0.027697.ckpt'
    """