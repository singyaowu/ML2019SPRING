import numpy as np
import torch
import torch.nn as nn


class MyGRU(nn.Module):
    def __init__(self, vecSize):
        super(MyGRU, self).__init__()

        self.rnn1 = nn.GRU(
            input_size=vecSize,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential( 
            #nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        rnn1_out, _ = self.rnn1(x, None)
        mean_pool = torch.mean(rnn1_out, dim=1)
        out = self.fc(mean_pool)
        return out.view(-1)

class MyLSTM(nn.Module):
    def __init__(self, vecSize):
        super(MyLSTM, self).__init__()

        self.rnn1 = nn.LSTM(
            input_size=vecSize,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential( 
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        rnn1_out, _ = self.rnn1(x, None)
        mean_pool = torch.mean(rnn1_out, dim=1)
        out = self.fc(mean_pool)
        return out.view(-1)
