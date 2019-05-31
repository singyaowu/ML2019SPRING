import numpy as np
import sys
import torch
import csv
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import *

class MyMobileCNN(nn.Module):
    def __init__(self):
        super(MyMobileCNN, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.conv = nn.Sequential(
            conv_bn(1, 32, 1),
            nn.Dropout2d(0.1),
            conv_bn(32, 32, 1),
            nn.MaxPool2d(kernel_size=2),

            conv_dw(32, 64, 1),
            nn.Dropout2d(0.1),
            conv_dw(64, 64, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.1),

            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),

            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),

            conv_dw(64, 128, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),

            conv_dw(128, 128, 1),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 7),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(           
            nn.Conv2d(in_channels=1,out_channels=64,
                kernel_size=3,stride=1,padding=1,), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),
            nn.Conv2d(in_channels=64,out_channels=64,
            kernel_size=5,stride=1,padding=2,),    
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),                 # output shape(16, 22, 22)
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=64,out_channels=128,
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=128,out_channels=128, 
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.Dropout2d(0.08),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 10, 10)
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=128,out_channels=256, 
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.08),

            nn.Conv2d(in_channels=256,out_channels=256,  
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 4, 4)
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=256,out_channels=512,  
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=512,out_channels=512,   # output shape(32, 8, 8)
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 4, 4)
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=512,out_channels=512,   # output shape(32, 8, 8)
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.05),

            nn.Conv2d(in_channels=512,out_channels=512,   # output shape(32, 8, 8)
                kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 4, 4)
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.5),
            nn.Linear(1000, 7),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x