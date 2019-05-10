import sys
import csv
import os

import numpy as np
import pandas as pd
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
from sklearn.cluster import KMeans

BATCH_SIZE = 512
num_imgs = 40000
model_name = 'model_finish.pkl'
images_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
if __name__ == "__main__":
    try:
        img_codes = np.load('img_codes.npy')
    except:


        test_dataset = Mydataset.TrainDataset( (1,num_imgs+1), images_path)
        test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = torch.load(model_name)
        model.cuda()
        model.eval()
        img_codes = np.zeros(shape=(1, 16*14*14), dtype=float)
        for step, (imgs) in enumerate(test_loader):
            imgs_cuda = imgs.cuda()
            output = model.encoder(imgs_cuda)[0]
            out_codes = output.view(len(output), -1).cpu().detach().numpy()
            #print(out_codes.shape)
            img_codes = np.concatenate((img_codes, out_codes), axis=0)
        img_codes = img_codes[1:]
        np.save('img_codes.npy', img_codes)
    print(img_codes.shape)
    
    print('=== start clustering ===')
    clf = KMeans(n_clusters=2)
    clf.fit(img_codes)
    print(clf.labels_)
    test_df = pd.read_csv(test_path, sep=',', dtype= {'id': int, 'image1_name':int, 'image2_name':int})
    #print(test_df)
    print('=== predicting ===')
    with open(output_path, 'w') as output_file: 
        output_file.write('id,label\n')
        for idx, row in test_df.iterrows():
            #print(row)
            predict = clf.labels_[row['image1_name']-1] == clf.labels_[row['image2_name']-1]
            output_file.write( str(row['id']) + ',' + str(1 if predict else 0) + '\n')
    print('writing output finish')
