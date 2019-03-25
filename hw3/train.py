import numpy as np
import sys
import torch
import csv
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#import matplotlib.pyplot as plt
# parameters
EPOCH = 200
BATCH_SIZE = 25
LEARNING_RATE = 0.0001
'''
class MyDataset(Dataset):
    def __init__(self, label_path):
        self.labels = parse_csv(label_path)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label, img = self.labels[idx]
        return torch.tensor(img), torch.tensor(label)
'''
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Sequential(                     # input shape(1, 48, 48)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,),                            # output shape(16, 44, 44)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                 # output shape(16, 22, 22)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,   # output shape(32, 18, 18)
                kernel_size=3,stride=1,padding=0,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                 # output shape(32, 10, 10)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,   # output shape(32, 8, 8)
                kernel_size=3,stride=1,padding=0,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                 # output shape(32, 4, 4)
        )

        self.out = nn.Linear(64 * 4 * 4, 7)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

def parse_csv(label_path):
    raw_data_fp = open(label_path,'r')
    lines = raw_data_fp.readlines()[1:]
    num_data = len(lines)
    raw_imgs = np.empty(shape=(num_data,1,48*48), dtype=float)
    raw_y = np.zeros(shape=(num_data,1),dtype=np.int64)
    #raw_y = np.zeros(shape=(num_data,7),dtype=np.int64)
    for i, line in enumerate(lines):
        nums = line.split(',')
        raw_y[i][0] = int(nums[0])
        #raw_y[i][int(nums[0])] = 1
        raw_imgs[i,:,:] = np.array([float(num) for num in nums[1].split(' ')])
    raw_imgs = raw_imgs.reshape((num_data,1,48,48))
    #raw_y = raw_y.reshape((num_data,1))
    return torch.tensor(raw_imgs).type(torch.FloatTensor), torch.tensor(raw_y).type(torch.LongTensor)

if __name__ == "__main__":
    raw_imgs, raw_y = parse_csv(sys.argv[1])
    print('raw_imgs:', raw_imgs.size())
    print('raw_y:', raw_y.size())
    training_set = Data.TensorDataset(raw_imgs, raw_y)# MyDataset(sys.argv[1])
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    
    # train
    device = torch.device('cuda')
    model = MyCNN()

    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    
    print('start training...')
    model.train()

    for epoch in range(EPOCH):
        train_loss, train_acc = [], []
        for step, (img, target) in enumerate(train_loader):
            #print(img.size(), target.size())           
            img_cuda = img.to(device, dtype=torch.float)
            target_cuda = target.to(device)

            optimizer.zero_grad()
            output = model(img_cuda)
            #print(output.size(), target_cuda.size())
            loss = loss_func(output, target_cuda.view(-1))
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((target_cuda == predict).cpu().numpy())
            if step % 500 == 0:
                print('accuracy: ', acc)
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
    
    # save parameters
    
    # test
    test_imgs, ids = parse_csv(sys.argv[2])
    predict_y = model(test_imgs) 