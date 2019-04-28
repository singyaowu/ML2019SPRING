#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import sys
import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import jieba
import emoji
from gensim.models import Word2Vec, KeyedVectors

import Model

EPOCH = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001

x_path = 'train_x.csv'
y_path = 'train_y.csv'
dict_txt_path = 'dict.txt.big'
vecSize = 150
senSize = 64
jieba.load_userdict(dict_txt_path)
wv = KeyedVectors.load("word2vec.model", mmap='r')

print('====== Reading train_x.csv ======')
x = pd.read_csv(x_path, sep=',', dtype={'id': int, 'comment':str}, index_col=0)
print('====== Reading train_y.csv ======')
y = pd.read_csv(y_path, sep=',', dtype={'id': int, 'label':int}, index_col=0)

print('====== Building Training data.csv ======')
x_train = np.zeros(shape=(len(x), senSize, vecSize), dtype=float)
for i_row, sen in enumerate(x['comment']):
    for i_w, w in enumerate(list(jieba.cut(emoji.demojize(sen), cut_all=False))):
        if i_w >= senSize: break
        x_train[i_row, i_w, :] = wv[w]

y_train = y['label'].values
del x, y
print('====== Building NN Model ======')
# model
model = Model.LSTM()
device = torch.device('cuda')
model.to(device)
model.train()
print(model)

trainDataset = Data.TensorDataset(
        torch.tensor(x_train).type(torch.FloatTensor), 
        torch.tensor(y_train).type(torch.LongTensor))
train_loader = Data.DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss() 

print('====== Start Training ======')
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x_cuda = x.to(device, dtype=torch.float)
        y_cuda = y.to(device)
        print(x.Size())
        x = x.view(-1, senSize, vecSize)   # reshape x to (batch, time_step, input_size)

        output = model(x)               # rnn output
        loss = loss_func(output, y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
torch.save(model.state_dict(), 'model_params.pkl') # parameters