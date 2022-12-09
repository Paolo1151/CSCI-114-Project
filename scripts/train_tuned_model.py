#!/usr/bin/env python3
import os
import sys
sys.path.append('..')

import torch
import joblib

from modules.base.model import BaseModel
from modules.base.trainer import AccidentSeverityModelTrainer
from modules.tuning.trainer import AccidentSeverityModelTuner

import config as cfg


mt = AccidentSeverityModelTuner(11, 3)
params = mt.get_tuned_params()

print(params)

joblib.dump(params, os.path.join(cfg.MODELS_PATH, 'BestParams.joblib'))