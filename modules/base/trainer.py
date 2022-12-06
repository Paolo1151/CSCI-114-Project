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

import config as cfg

df_train = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned_train.csv'))
df_test = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned_test.csv'))

class AccidentSeverityModelTrainer():
    def __init__(self, model_type, input_dim, output_dim, batch_size=5):
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.device = 'cpu'

        X_training = df_train.drop(columns=['Accident_severity']).values
        y_training = AccidentSeverityModelTrainer._one_hot_encoding(df_train['Accident_severity'].values)

        X_training = torch.tensor(X_training).float().to(self.device)
        y_training = torch.tensor(y_training).float().to(self.device)

        train_dataset = AccidentSeverityDataset(X_training, y_training)

        X_validation = df_test.drop(columns=['Accident_severity']).values
        y_validation = AccidentSeverityModelTrainer._one_hot_encoding(df_test['Accident_severity'].values)

        self.y_test = y_validation

        X_validation = torch.tensor(X_validation).float().to(self.device)
        y_validation = torch.tensor(y_validation).float().to(self.device)

        self.x_test = X_validation

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

    @staticmethod
    def _one_hot_encoding(nd_array):
        ohe = []
        classes = max(nd_array) + 1
        for n in nd_array:
            encoding = [0] * classes
            encoding[n] = 1
            ohe.append(encoding)
        return np.array(ohe)

    @staticmethod
    def _convert_prob_to_deterministic(nd_array):
        one_hot_encoding_predictions = nd_array

        for i in range(len(nd_array)):
            max_pred = max(nd_array[i])

            for j in range(len(nd_array[i])):
                one_hot_encoding_predictions[i][j] = 1 if nd_array[i][j] == max_pred else 0

        return one_hot_encoding_predictions


    def train_new_model(self, epochs=10):
        model = self.model_type(self.input_dim, self.output_dim)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(epochs):
            # Training Step
            losses = []
            accuracies = []
            roc_auc = []
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                predictions = model.forward(data)
                loss = criterion(predictions, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prob_pred = model.forward(self.x_test).detach().cpu().numpy()
                det_pred = AccidentSeverityModelTrainer._convert_prob_to_deterministic(
                    prob_pred
                )

                losses.append(loss.item())
                accuracies.append(accuracy_score(self.y_test, det_pred))
                roc_auc.append(roc_auc_score(self.y_test, prob_pred))

                avg_loss = sum(losses) / len(losses)
                avg_acc = sum(accuracies) / len(accuracies)
                avg_ra = sum(roc_auc) / len(roc_auc)

                print(f'Epoch: {epoch+1}\t-\tLoss: {avg_loss:.4f}\t-\tAccuracy: {avg_acc:.4f}\t-\tROC-AUC: {avg_ra:.4f}', end='\r')
            

        return model

