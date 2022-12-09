import os
import sys
sys.path.append(os.path.join('..', '..'))

import pandas as pd
import numpy as np

import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


from modules.base.dataset import AccidentSeverityDataset
from modules.base.trainer import AccidentSeverityModelTrainer
from modules.tuning.model import TunableModel

import config as cfg

df_train = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned_train.csv'))
df_test = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned_test.csv'))

class AccidentSeverityModelTuner():
    def __init__(self, input_dim, output_dim, batch_size=5):        
        self.model_type = TunableModel
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

    def objective(self, params, epochs=100):
        splits = StratifiedKFold(5, shuffle=True, random_state=42)

        results = []
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
            curr_tl = DataLoader(curr_ds, batch_size=int(params['batch_size']+1), shuffle=False, drop_last=False)

            model = self.model_type(self.input_dim, self.output_dim, params).to(self.device) 

            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
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
                    det_pred = AccidentSeverityModelTuner._convert_prob_to_deterministic(
                        prob_pred
                    )

                    losses.append(loss.item())
                    accuracies.append(accuracy_score(y_val, det_pred))
                    roc_auc.append(roc_auc_score(y_val, prob_pred))

                    avg_loss = sum(losses) / len(losses)
                    avg_acc = sum(accuracies) / len(accuracies)
                    avg_ra = sum(roc_auc) / len(roc_auc)

            results.append(dict(
                avg_loss=avg_loss,
                avg_acc=avg_acc,
                avg_ra=avg_ra
            ))
        
            print()

        avg_roc_auc = np.mean([x['avg_ra'] for x in results])

        return -1 * avg_roc_auc

    def train_tuned_model(self, params, epochs=100):
        splits = StratifiedKFold(5, shuffle=True, random_state=42)

        results = []
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
            curr_tl = DataLoader(curr_ds, batch_size=int(params['batch_size']+1), shuffle=False, drop_last=False)

            model = self.model_type(self.input_dim, self.output_dim, params).to(self.device) 

            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.CrossEntropyLoss().to(self.device)

            vcriterion = nn.CrossEntropyLoss()

            tloss_curve = []
            vloss_curve = []
            for epoch in range(epochs):
                # Training Step
                tlosses = []
                vlosses = []
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

                    tlosses.append(loss.item())

                    prob_pred = model.forward(X_val).detach().cpu()
                    det_pred = AccidentSeverityModelTuner._convert_prob_to_deterministic(
                        prob_pred.numpy()
                    )

                    v_test = torch.tensor(y_val).float()

                    loss = vcriterion(prob_pred, v_test)
                    
                    vlosses.append(loss.item())
                    accuracies.append(accuracy_score(y_val, det_pred))
                    roc_auc.append(roc_auc_score(y_val, prob_pred.numpy()))

                    tavg_loss = sum(tlosses) / len(tlosses)
                    vavg_loss = sum(vlosses) / len(vlosses)
                    avg_acc = sum(accuracies) / len(accuracies)
                    avg_ra = sum(roc_auc) / len(roc_auc)

                    print(f'Fold: {fold+1}\t-\tEpoch: {epoch+1}\t-\tLoss: {vavg_loss:.4f}\t-\tAccuracy: {avg_acc:.4f}\t-\tROC-AUC: {avg_ra:.4f}', end='\r')

                tloss_curve.append(sum(tlosses) / len(tlosses))
                vloss_curve.append(sum(vlosses) / len(vlosses))

            results.append(dict(
                train_loss_curve=tloss_curve,
                val_loss_curve = vloss_curve,
                avg_loss=vavg_loss,
                avg_acc=avg_acc,
                avg_ra=avg_ra,
                model=model
            ))
        
            print()

        best_model_partition = None
        best_model_auc = -1
        for result in results:
            if best_model_auc < result['avg_ra']:
                best_model_partition = result
                best_model_auc = result['avg_ra']

        return best_model_partition

    def test_tuned_model(self, model):
        curr_x_test = torch.tensor(self.x_test).float().to(self.device)

        prob_pred = model.forward(curr_x_test).detach().cpu().numpy()
        det_pred = AccidentSeverityModelTuner._convert_prob_to_deterministic(
            prob_pred
        )

        cm = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(det_pred, axis=1))
        ra = roc_auc_score(self.y_test, prob_pred)

        joblib.dump(cm, os.path.join(cfg.MODELS_PATH, 'AccidentSeverityConfusionMatrix.joblib'))
        joblib.dump(ra, os.path.join(cfg.MODELS_PATH, 'AccidentSeverityTestROCAUC.joblib'))


    def get_tuned_params(self):
        params_list = {
            # Model Architecture 
            'decay_rate': hp.uniform('decay_rate', 0.001, 1),
            'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
            'num_layers': hp.randint('num_layers', 10),
            'start_nodes': hp.randint('start_nodes', 4096),

            'input_dim': self.input_dim,
            'output_dim': self.output_dim, 

            # external parameters
            'learning_rate': hp.uniform('learning_rate', 1e-4, 1e-3),
            'batch_size': hp.randint('batch_size', 31)
        }

        best_params = fmin(fn=self.objective, space=params_list, max_evals=24, algo=tpe.suggest)
        
        best_params['input_dim'] = self.input_dim
        best_params['output_dim'] = self.output_dim

        results = self.train_tuned_model(best_params)
        model = results['model']

        self.test_tuned_model(model)

        joblib.dump(results['train_loss_curve'], os.path.join(cfg.MODELS_PATH, 'AccidentSeverityTrainLossCurve.joblib'))
        joblib.dump(results['val_loss_curve'], os.path.join(cfg.MODELS_PATH, 'AccidentSeverityValLossCurve.joblib'))

        state = { 'state_dict': model.state_dict() }
        torch.save(state, os.path.join(cfg.MODELS_PATH, 'AccidentSeverityModel_Tuned.pth'))

        return best_params

    

