import sys
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tf
import Model
import Mydataset
from skimage import io

import matplotlib.pyplot as plt

LEARNING_RATE = 0.0003
BATCH_SIZE = 128
EPOCH = 30
validation = False
#bash cluster.sh <images path> <test_case.csv path> <prediction file path>
def save_imgs(imgs, filename):
    num_imgs = len(imgs)
    imgs = np.transpose(imgs, (0,2,3,1))
    #print(imgs.shape)

    for i in range(0, num_imgs):
        plt.subplot(16, num_imgs//16, i+1)
        plt.axis('off')
        plt.imshow(imgs[i])
    plt.savefig(filename)
    plt.close()
if __name__ == "__main__":
    num_imgs = 40000
    images_path = sys.argv[1]
    train_dataset = Mydataset.TrainDataset( (1,num_imgs+1), images_path)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model = Model.MyAutoEncoder()
    model.cuda()
    model.train()
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print('=== start training ===')
    for epoch in range(EPOCH):
        train_loss_list = []
        val_loss_list = []
        for _, (imgs) in enumerate(train_loader):
            imgs_cuda = imgs.cuda()
            #print(imgs.size())
            output = model(imgs_cuda)
            loss = criterion(output, imgs_cuda)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
        train_loss = np.mean(train_loss_list)
        val_loss = 0
        if validation:
            model.eval()
            for _, (imgs) in enumerate(val_loader):
                imgs_cuda = imgs[0].cuda()
                output = model(imgs_cuda)
                loss = criterion(output, imgs)
                val_loss_list.append(loss.item())
            val_loss = np.mean(val_loss_list)
            model.train()
        if epoch % 5 == 0:
            torch.save(model, 'model_tmp.pkl')
            save_imgs(imgs.numpy(), '%d_original.jpg'%epoch)
            save_imgs(output.cpu().detach().numpy(), '%d_decode.jpg'%epoch)
        print("Epoch: {}| Train Loss: {:.6f}| Val Loss: {:.6f}"\
            .format(epoch + 1, train_loss, val_loss))
        
    torch.save(model, 'model_finish.pkl')