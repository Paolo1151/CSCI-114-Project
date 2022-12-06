import os
import sys
sys.path.append('..')

import torch

from modules.base.model import BaseModel
from modules.base.trainer import AccidentSeverityModelTrainer

import config as cfg

mt = AccidentSeverityModelTrainer(BaseModel, 88, 3)
model = mt.train_new_model(epochs=100)

state = { 'state_dict': model.state_dict() }
torch.save(state, os.path.join(cfg.MODELS_PATH, 'AccidentSeverityModel_Base.pth'))

