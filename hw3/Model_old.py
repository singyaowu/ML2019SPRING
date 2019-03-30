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

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(                     # input shape(1, 48, 48)
            nn.Conv2d(in_channels=1,out_channels=32,
                kernel_size=5,stride=1,padding=2,),     # output shape(16, 44, 44)
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),                 # output shape(16, 22, 22)
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,   # output shape(32, 18, 18)
                kernel_size=3,stride=1,padding=1,),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 10, 10)
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,   # output shape(32, 8, 8)
                kernel_size=3,stride=1,padding=1,),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 4, 4)
            nn.Dropout(0.4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=512,   # output shape(32, 8, 8)
                kernel_size=3,stride=1,padding=1,),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),                 # output shape(32, 4, 4)
            nn.Dropout(0.4)
        )
        self.conv5 = nn
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 7),
        )
        self.out = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        output = self.out(x)
        return output