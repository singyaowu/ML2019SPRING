import numpy as np
import torch
import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, vecSize):
        super(MyLSTM, self).__init__()

        self.rnn1 = nn.LSTM(
            input_size=vecSize,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential( 
            #nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        rnn1_out, (_,_) = self.rnn1(x, None)
        out = self.fc(rnn1_out[:, -1, :].view(-1, 256))
        return out.view(-1)
