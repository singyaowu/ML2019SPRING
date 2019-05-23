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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt

class TestDataset(Dataset):
    def __init__(self):
        self.imgs = np.load("../tmp/visualization.npy")
        self.transform = tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx]).type(torch.FloatTensor)
        return img
BATCH_SIZE = 128
num_imgs = 40000
model_name = 'model_finish_32dim.pkl'
#model_name = 'model_finishCNN.pkl'

cluster_method = 't-SNE'
if __name__ == "__main__":

    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    device = torch.device('cuda')
    model = torch.load(model_name)
    model.cuda()
    model.eval()
    latent_vec = np.zeros(shape=(1, 32), dtype=float)
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

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_jobs=8, random_state=0)
    latent_vec = tsne.fit_transform(latent_vec)
    
    clf = KMeans(n_clusters=2, random_state=0)
    clf.fit(latent_vec)
    labels = clf.labels_

    code_min, code_max = latent_vec.min(0), latent_vec.max(0)
    code_norm = (latent_vec-code_min)/(code_max-code_min)
    
    color = [['b', 'r'][i] for i in labels]
    plt.scatter(code_norm[:,0], code_norm[:,1], s=1, c=color)
    plt.show()
