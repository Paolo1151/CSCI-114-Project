import os
import sys
sys.path.append(os.path.join('..', '..'))

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from modules.base.dataset import AccidentSeverityDataset
from modules.tuning.model import TunableModel

import config as cfg

class TunableModelFactory():
    @staticmethod
    def create_new_model(params):
        activation = \
            F.relu if params['activation'] == 'relu' else\
            torch.sigmoid if params['activation'] == 'sigmoid' else\
            torch.tanh if params['activation'] == 'tanh' else\
            None
        
        layers = [{ 'name': 'Input', 'layer': nn.Linear(params['input_dim'], params['start_nodes']), 'activation': activation }]
        nodes = params['start_nodes']
        for i in range(params['num_layers']):
            next_nodes = int(nodes * params['decay_rate'])
            new_layer = nn.Linear(nodes, next_nodes)
            layers.append({
                'name': f'Layer_{i}',
                'layer': new_layer,
                'activation': activation
            })
            nodes = next_nodes
        layers.append({
                'name': f'Output',
                'layer': nn.Linear(nodes, params['output_dim']),
                'activation': torch.sigmoid
            })

        model = TunableModel(layers)

        return model


        