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

from modules.generation.autoencoder import AccidentSeverityVariationalAutoencoder
from modules.generation.dataset import AccidentSeverityAutoencoderDataset

import config as cfg

df_train = pd.read_csv(os.path.join(cfg.DATA_PATH, 'cleaned_train.csv'))

class AccidentSeverityBalancer():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 0.001

        self.balanced_limit = 150

        self.undersample_and_prepare()

    def undersample_and_prepare(self, num_classes=3):
        dfs = []
        for i in range(num_classes):
            df = df_train[df_train['Accident_severity']==i].drop(columns=['Accident_severity'])
            
            csample = min(self.balanced_limit, df.shape[0])
            subject_resample = csample != self.balanced_limit

            df = df.sample(csample)
            n_entries = df.shape[0]
            odf = df.copy()
            df = torch.tensor(df.values).float().to(self.device)

            dl = DataLoader(
                AccidentSeverityAutoencoderDataset(df),
                batch_size=5,
                shuffle=False,
                drop_last=False 
            )

            dfs.append({
                'class': i,
                'loader': dl,
                'resample': subject_resample,
                'n_entries': csample,
                'odf': odf
            })

        self.dfs = dfs
    
    @staticmethod
    def _final_loss(bce_loss, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce_loss + KLD


    def generate_balanced_train(self, epochs=100):
        model = AccidentSeverityVariationalAutoencoder().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss(reduction='sum').to(self.device)

        balanced_dfs = []
        for df in self.dfs:
            if df['resample']:
                print('OverSampling  Class {}'.format(df['class']))

                for epoch in range(epochs):
                    losses = []
                    for batch_idx, (data, targets) in enumerate(df['loader']):
                        data = data.to(self.device)
                        targets = targets.to(self.device)

                        optimizer.zero_grad()
                        reconstruction, mu, logvar = model.forward(data)
                        bce_loss = criterion(reconstruction, data)
                        loss = AccidentSeverityBalancer._final_loss(bce_loss, mu, logvar)
                        loss.backward()
                        optimizer.step()

                        losses.append(loss.item())

                        avg_loss = sum(losses) / len(losses)

                        print(f'Epoch: {epoch+1}\t-\tLoss: {avg_loss:.4f}', end='\r')
                    
                print()
                # Generate balance_limit - current shape
                sampled_mu = torch.Tensor(np.array([np.zeros(25)])).to(self.device)
                sampled_logvar = torch.Tensor(np.array([np.zeros(25)])).to(self.device)
                
                bal_df = df['odf']
                for i in range(self.balanced_limit - df['odf'].shape[0]):    
                    reconstruction = model.sample(sampled_mu, sampled_logvar)
                    generated_data_point = reconstruction[0].detach().cpu().numpy()
                    
                    dhi = 1 - generated_data_point
                    dlo = 0 - generated_data_point
                    idx = dhi + dlo < 0
                    rounded = generated_data_point + np.where(idx, dhi, dlo)

                    curr_df = pd.DataFrame(rounded.astype(int), index=df['odf'].columns).T
                    bal_df = pd.concat([bal_df, curr_df])
            else:
                bal_df = df['odf']

            # Combine datasets
            labels = np.ones(self.balanced_limit) * df['class']
            bal_df['Accident_severity'] = labels.astype(int)

            balanced_dfs.append(bal_df)
        
        df_train = pd.concat(balanced_dfs)

        df_train.to_csv(os.path.join(cfg.DATA_PATH, 'cleaned_train_balanced.csv'), index=False)

        print('Balanced Dataset Generated!')



        


