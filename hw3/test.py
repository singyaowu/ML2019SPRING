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

def parse_csv(label_path):
    raw_data_fp = open(label_path,'r')
    lines = raw_data_fp.readlines()[1:]
    num_data = len(lines)
    raw_imgs = np.empty(shape=(num_data,1,48*48), dtype=float)
    raw_y = np.zeros(shape=(num_data),dtype=np.int64)
    #raw_y = np.zeros(shape=(num_data,7),dtype=np.int64)
    for i, line in enumerate(lines):
        nums = line.split(',')
        raw_y[i] = int(nums[0])
        #raw_y[i][int(nums[0])] = 1
        raw_imgs[i,:,:] = np.array([float(num) for num in nums[1].split(' ')]) /255
    
    raw_imgs = raw_imgs.reshape((num_data,1,48,48))
    
    #raw_y = raw_y.reshape((num_data,1))
    return raw_imgs, raw_y

if __name__ == "__main__":
    model = MyCNN()
    model.load_state_dict(torch.load('model_params.pkl'))
    test_imgs, ids = parse_csv(sys.argv[1])
    test_imgs = torch.tensor(test_imgs).type(torch.FloatTensor)
    num_test_data = test_imgs.size()[0]
    print('num_test_data=', num_test_data)
    #test_imgs = test_imgs.to(device, dtype=torch.float)
    predict = model(test_imgs)
    predict_y = torch.max(predict, 1)[1]
    output_file = open(sys.argv[2], 'w')
    output_file.write("id,label\n")
    #print(predict_y.size())
    #print(predict_y)
    for i in range(num_test_data):
        output_file.write( str(i) + ',' + str(int(predict_y[i])) + '\n')