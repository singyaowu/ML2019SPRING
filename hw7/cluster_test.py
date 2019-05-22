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
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt

BATCH_SIZE = 128
num_imgs = 40000
model_name = 'model_finish_32dim.pkl'
images_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
cluster_method = 't-SNE'
if __name__ == "__main__":

    test_dataset = Mydataset.TestDataset( (1,num_imgs+1), images_path)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    device = torch.device('cuda')
    model = torch.load(model_name)
    model.cuda()
    model.eval()
    latent_vec = np.zeros(shape=(1, 32), dtype=float)#torch.tensor([]).to(device, dtype=torch.float)
    for step, (imgs) in enumerate(test_loader):
        imgs_cuda = imgs.to(device, dtype=torch.float)
        output = model.encode(imgs_cuda)
        #latent_vec = torch.cat((latent_vec, output), dim=0)
        out_codes = output.cpu().detach().numpy()
        latent_vec = np.concatenate((latent_vec, out_codes), axis=0)
    latent_vec = latent_vec[1:]#latent_vec.cpu().detach().numpy()
    print(latent_vec.shape)
    #print(latent_vec)
    print('=== start clustering ===')
    print('perform %s'%cluster_method)
    if cluster_method == 'pca':
        pca = PCA(n_components=2, copy=False, whiten=True, svd_solver='full')
        latent_vec = pca.fit_transform(latent_vec)
    elif cluster_method == 't-SNE':
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_jobs=8, random_state=0)
        #tsne = TSNE(n_components=2,init='pca', random_state=0)
        latent_vec = tsne.fit_transform(latent_vec)
    
    clf = KMeans(n_clusters=2, random_state=0)
    clf.fit(latent_vec)
    print('clf.labels_\'s len: %d'%len(clf.labels_))
    

    #code_min, code_max = code_tsne.min(0), img_tsne.max(0)
    #code_norm = (code_tsne-code_min)/(code_max-code_min)

    #color = [['r', 'b'][i] for i in clf.labels_]
    #plt.scatter(code_norm[:,0], code_norm[:,1], s=1, c=color)

    #print(test_df)
    print('=== predicting ===')
    test_df = pd.read_csv(test_path, sep=',', dtype= {'id': int, 'image1_name':int, 'image2_name':int})
    predict = (clf.labels_[(test_df['image1_name']-1)] == clf.labels_[(test_df['image2_name']-1)]).astype(int) 
    predict = np.concatenate((np.arange(len(predict)).reshape(-1,1),predict.reshape(-1,1)), axis=1)
    df_predict = pd.DataFrame(data=predict, columns=['id','label'])
    #print(df_predict)
    df_predict.to_csv(output_path, sep=',', index=False)
    print('writing output finish')
