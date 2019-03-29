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
    #torch.tensor(raw_imgs).type(torch.FloatTensor), torch.tensor(raw_y).type(torch.LongTensor)

# parameters

EPOCH = 700
BATCH_SIZE = 256
LEARNING_RATE = 0.001

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
def augment_data(imgs, y):
    new_imgs = imgs.copy()
    new_y = y.copy()
    num_data = new_imgs.shape[0]
    for i in range(num_data):
        new_imgs[i,0,:,:] = new_imgs[i,0,:,::-1]
    return new_imgs, new_y

if __name__ == "__main__":
    raw_imgs, raw_y = parse_csv(sys.argv[1])
    aug_imgs, aug_y = augment_data(raw_imgs, raw_y)
    train_imgs = np.concatenate((raw_imgs, aug_imgs), axis=0)
    train_y = np.concatenate((raw_y, aug_y), axis=0)
    print(train_imgs.shape, train_y.shape)
    train_imgs = torch.tensor(train_imgs).type(torch.FloatTensor)
    train_y = torch.tensor(train_y).type(torch.LongTensor)
    #print('raw_imgs:', raw_imgs.size())
    #print('raw_y:', raw_y.size())
    num_valid_data = 0#raw_imgs.size()[0] // 4
    
    val_imgs = train_imgs[:num_valid_data,:,:]
    val_y = train_y[:num_valid_data]
    training_set = Data.TensorDataset(train_imgs[num_valid_data:,:,:], train_y[num_valid_data:])
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_set = Data.TensorDataset(train_imgs[num_valid_data:,:,:], train_y[num_valid_data:])
    # train
    device = torch.device('cuda')
    model = MyCNN()

    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    #if num_valid_data > 0:
    #    val_imgs = val_imgs.to(device, dtype=torch.float)
    #    val_y = val_y.to(device)

    print('start training...')
    model.train()


    for epoch in range(EPOCH):
        train_loss, train_acc = [], []
        torch.cuda.empty_cache()
        for step, (img, target) in enumerate(train_loader):
            #print(img.size(), target.size())           
            img_cuda = img.to(device, dtype=torch.float)
            target_cuda = target.to(device)

            optimizer.zero_grad()
            output = model(img_cuda)
            #print(output.size(), target_cuda.size())
            loss = loss_func(output, target_cuda)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((target_cuda == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        val_acc = 0
        if num_valid_data != 0:
            model.eval()
            predict = torch.max(model(val_imgs), 1)[1]
            val_acc = np.mean((val_y == predict).cpu().numpy())
            model.train()
        # torch.cuda.empty_cache()
        print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, Val Acc: {:.4f}"\
            .format(epoch + 1,
             np.mean(train_loss), np.mean(train_acc), val_acc
             ))
    model.to('cpu')
    model.eval()
    # save parameters
    # torch.save(model, 'model.pkl') # entire net
    torch.save(model.state_dict(), 'model_params.pkl') # parameters
    # validation
    # test
    test_imgs, ids = parse_csv(sys.argv[2])
    test_imgs = torch.tensor(test_imgs).type(torch.FloatTensor)
    num_test_data = test_imgs.size()[0]
    print('num_test_data=', num_test_data)
    #test_imgs = test_imgs.to(device, dtype=torch.float)
    predict = model(test_imgs)
    predict_y = torch.max(predict, 1)[1]
    output_file = open(sys.argv[3], 'w')
    output_file.write("id,label\n")
    #print(predict_y.size())
    #print(predict_y)
    for i in range(num_test_data):
        output_file.write( str(i) + ',' + str(int(predict_y[i])) + '\n')
    print('finish')
