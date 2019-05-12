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
model_name = 'model_tmp416.pkl'
images_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
if __name__ == "__main__":

    test_dataset = Mydataset.TestDataset( (1,num_imgs+1), images_path)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = torch.load(model_name)
    model.cuda()
    model.eval()
    img_codes = np.zeros(shape=(1, 64), dtype=float)
    for step, (imgs) in enumerate(test_loader):
        imgs_cuda = imgs.cuda()
        output = model.encode(imgs_cuda)
        #print(output.size())
        #output = model.encoder2(output)[0]   
        out_codes = output.cpu().detach().numpy()
        #print(len(imgs_cuda))
        #print(out_codes.shape)
        #if img_codes is None: img_codes = out_codes.copy()
        img_codes = np.concatenate((img_codes, out_codes), axis=0)
        #print(img_codes)
    img_codes = img_codes[1:]

    print(img_codes.shape)
    #print(img_codes)
    print('=== start clustering ===')
    clf = KMeans(n_clusters=2, random_state=0)
    clf.fit(img_codes)
    print(type(clf.labels_))
    print('clf.labels_\'s len: %d'%len(clf.labels_))
    
    #print(test_df)
    print('=== predicting ===')
    test_df = pd.read_csv(test_path, sep=',', dtype= {'id': int, 'image1_name':int, 'image2_name':int})
    #print(test_df)
    predict = (clf.labels_[(test_df['image1_name']-1)] == clf.labels_[(test_df['image2_name']-1)]).astype(int) 
    
    predict = np.concatenate((np.arange(len(predict)).reshape(-1,1),predict.reshape(-1,1)), axis=1)
    #print(predict)
    df_predict = pd.DataFrame(data=predict, columns=['id','label'])
    #print(df_predict)
    df_predict.to_csv(output_path, sep=',', index=False)
    #with open(output_path, 'w') as output_file: 
    #    output_file.write('id,label\n')
    #    for idx, row in test_df.iterrows():
    #        #print(row)
    #        predict = clf.labels_[row['image1_name']-1] == clf.labels_[row['image2_name']-1]
    #        output_file.write( str(row['id']) + ',' + str(1 if predict else 0) + '\n')
    print('writing output finish')
