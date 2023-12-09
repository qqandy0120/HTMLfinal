import typing 
import json
from typing import List, Dict
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from util import *
import random
SPLITS = ['train', 'valid', 'test_1021', 'test_1204', 'test_1211']
MODES = ['normal', 'y_only']

class BikeDataset(Dataset):
    def __init__(
        self,
        station_id: str,
        split: str,
        time_step: int = 6,
        mode: str = 'normal',
    ):

        assert station_id in get_sno_test_set(), f'station is not in test set'
        assert split in SPLITS, f'split should be one of {SPLITS}'
        assert mode in MODES, f'mode should be one of {MODES}'

        self.split = split
        self.time_step = time_step
        self.mode = mode

        self.df = get_stat_raw_data(station_id, split, self.time_step)
                
    def __len__(self):
        return len(self.df) - self.time_step
    
    def __getitem__(self, index):
        if self.mode == 'normal':
            features = self.df.iloc[index:index+self.time_step, 1:]
        if self.mode == 'y_only':
            features = self.df.iloc[index:index+self.time_step, -1]
        labels = self.df.iloc[index+self.time_step, -1]
        features_tensor = torch.tensor(features.values).to(torch.float32)
        labels_tensor = torch.tensor([labels]).to(torch.float32)
        return{
            'id': self.df.iloc[index:index+self.time_step,0].tolist(),
            'feature': features_tensor.view(self.time_step, -1),
            'label': labels_tensor,
            'pred_target': self.df.iloc[index+self.time_step,0]
        }

    @property
    def get_time_step(self):
        return self.time_step
    
    def update_with_prediction(self, pred_target, pred):
        pred_target = pred_target[0]
        pred = pred.item()
        self.df.loc[self.df['id'] == pred_target, 'sbi'] = pred
        




if __name__ == '__main__':
    random.seed(7777777)
    station_lst = get_sno_test_set()
    selected_station = random.choice(station_lst)


    train = BikeDataset(station_id=selected_station, split='train')
    print(len(train))
    print('id:')
    print(train[10]['id'])
    print('feature:')
    print(train[10]['feature'])
    print('label:')
    print(train[10]['label'])
    valid = BikeDataset(station_id=selected_station, split='valid')
    print(len(valid))
    print(valid[0]['feature'])
    print(valid[0]['label'])
    pass