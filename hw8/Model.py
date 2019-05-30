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
        super(MyCNN, self).__init__()
        def conv_bn(inp, oup, kernel_sz, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_sz, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, kernel_sz, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_sz, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.conv = nn.Sequential(
            conv_bn(1, 64, 3, 1),
            #nn.Dropout2d(0.05),
            conv_dw(64, 64, 3, 1),
            nn.MaxPool2d(kernel_size=2),                 # output shape(16, 22, 22)
            nn.Dropout2d(0.05),

            conv_dw(64, 128, 3, 1),
            #nn.Dropout2d(0.1),
            conv_dw(128, 128, 3, 1),
            nn.Dropout2d(0.08),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 10, 10)
            #nn.Dropout2d(0.05),

            conv_dw(128, 256, 3, 1),
            #nn.Dropout2d(0.08),
            conv_dw(256, 256, 3, 1),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 4, 4)
            nn.Dropout2d(0.05),

            conv_dw(256, 512, 3, 1),
            #nn.Dropout2d(0.05),
            conv_dw(512, 512, 3, 1),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 4, 4)
            nn.BatchNorm2d(512),
            #nn.Dropout2d(0.05),

            conv_dw(512, 512, 3, 1),
            #nn.Dropout2d(0.05),
            conv_dw(512, 512, 3, 1),
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
        #self.out = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x