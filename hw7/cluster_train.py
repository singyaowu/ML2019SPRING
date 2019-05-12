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
import torchvision.utils as vutils
import Model
import Mydataset
from skimage import io

import matplotlib.pyplot as plt

LEARNING_RATE = 0.0001
BATCH_SIZE = 256
EPOCH = 1000
validation = False
#bash cluster.sh <images path> <test_case.csv path> <prediction file path>
def save_imgs(imgs, filename):
    imgs = imgs*0.5+0.5
    num_imgs = len(imgs)
    imgs = np.transpose(vutils.make_grid(imgs, padding=2, normalize=True), (1,2,0))
    #print(imgs.shape)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    plt.imshow(imgs)
    plt.savefig(filename)
    plt.close()
if __name__ == "__main__":
    num_imgs = 40000
    images_path = sys.argv[1]
    if validation:
        start_val = (num_imgs * 3) // 4+1
        train_dataset = Mydataset.TrainDataset( (1,start_val), images_path)
        val_dataset = Mydataset.TrainDataset( (start_val,num_imgs+1), images_path)
        val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    else: train_dataset = Mydataset.TrainDataset( (1,num_imgs+1), images_path)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = Model.MyAutoEncoder()
    model.cuda()
    model.train()
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print('=== start training ===')
    for epoch in range(EPOCH):
        train_loss_list = []
        val_loss_list = []
        ss = 0
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
                imgs_cuda = imgs.cuda()
                output = model(imgs_cuda)
                loss = criterion(output, imgs_cuda)
                
                val_loss_list.append(loss.item())
            val_loss = np.mean(val_loss_list)
            model.train()
        if epoch % 32 == 0:
            torch.save(model, 'model_tmp.pkl')
            save_imgs(imgs, '%d_original.jpg'%epoch)
            save_imgs(output.cpu().detach(), '%d_decode.jpg'%epoch)
            torch.save(model, 'model_tmp%d.pkl'%epoch)
        print("Epoch: {}| Train Loss: {:.6f}| Val Loss: {:.6f}"\
            .format(epoch + 1, train_loss, val_loss))
        
    torch.save(model, 'model_finish.pkl')