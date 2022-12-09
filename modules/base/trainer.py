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
from sklearn.model_selection import StratifiedKFold

from modules.base.dataset import AccidentSeverityDataset

import config as cfg

df_train = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned_train.csv'))
df_test = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned_test.csv'))

class AccidentSeverityModelTrainer():
    def __init__(self, model_type, input_dim, output_dim, batch_size=5):
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_training = df_train.drop(columns=['Accident_severity']).values
        y_training = AccidentSeverityModelTrainer._one_hot_encoding(df_train['Accident_severity'].values)

        self.x_train = X_training
        self.y_train = y_training

        X_test = df_test.drop(columns=['Accident_severity']).values
        y_test = AccidentSeverityModelTrainer._one_hot_encoding(df_test['Accident_severity'].values)

        self.x_test = X_test
        self.y_test = y_test

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


    def train_new_model(self, epochs=10, model=None):
        splits = StratifiedKFold(5, shuffle=True, random_state=42)

        results = dict()
        ds = AccidentSeverityDataset(self.x_train, self.y_train)

        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(ds)), np.argmax(self.y_train, axis=1))):
            X_train = self.x_train[train_idx]
            y_train = self.y_train[train_idx]

            X_val = self.x_train[val_idx]
            y_val = self.y_train[val_idx]

            X_train = torch.tensor(X_train).float().to(self.device)
            y_train = torch.tensor(y_train).float().to(self.device)

            X_val = torch.tensor(X_val).float().to(self.device)

            curr_ds = AccidentSeverityDataset(X_train, y_train)
            curr_tl = DataLoader(curr_ds, batch_size=5, shuffle=False, drop_last=False)

            model = self.model_type(self.input_dim, self.output_dim).to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss().to(self.device)

            for epoch in range(epochs):
                # Training Step
                losses = []
                accuracies = []
                roc_auc = []
                for batch_idx, (data, targets) in enumerate(curr_tl):
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    predictions = model.forward(data)
                    loss = criterion(predictions, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    prob_pred = model.forward(X_val).detach().cpu().numpy()
                    det_pred = AccidentSeverityModelTrainer._convert_prob_to_deterministic(
                        prob_pred
                    )

                    losses.append(loss.item())
                    accuracies.append(accuracy_score(y_val, det_pred))
                    roc_auc.append(roc_auc_score(y_val, prob_pred))

                    avg_loss = sum(losses) / len(losses)
                    avg_acc = sum(accuracies) / len(accuracies)
                    avg_ra = sum(roc_auc) / len(roc_auc)

                    print(f'Fold: {fold+1}\t-\tEpoch: {epoch+1}\t-\tLoss: {avg_loss:.4f}\t-\tAccuracy: {avg_acc:.4f}\t-\tROC-AUC: {avg_ra:.4f}', end='\r')

            results[fold] = dict(
                avg_loss=avg_loss,
                avg_acc=avg_acc,
                avg_ra=avg_ra
            )

            print()

        return model, results

