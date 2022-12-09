#!/usr/bin/env python3
import os
import sys
from pprint import pprint
sys.path.append('..')

import torch

from modules.base.model import BaseModel
from modules.base.trainer import AccidentSeverityModelTrainer

import config as cfg

mt = AccidentSeverityModelTrainer(BaseModel, 11, 3)
model, results = mt.train_new_model(epochs=100)

pprint(results)

state = { 'state_dict': model.state_dict() }
torch.save(state, os.path.join(cfg.MODELS_PATH, 'AccidentSeverityModel_Base.pth'))

