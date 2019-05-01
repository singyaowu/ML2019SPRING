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

EPOCH = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

#bash hw6_test.sh <test_x file> <dict.txt.big file> <output file>
#bash hw6_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
x_path = 'train_x.csv'
y_path = 'train_y.csv'
dict_txt_path = 'dict.txt.big'
vecSize = 150
senSize = 100
shuffle = False

jieba.load_userdict(dict_txt_path)
wv = KeyedVectors.load("word2vec%d.model"%vecSize, mmap='r')


print('====== Reading Training data.csv ======')
try:
    assert 1==0
    x_train = np.load('x_config.npy')
    assert x_train[2] == vecSize
    assert x_train[1] == senSize
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
 
except:
    '''
    
    x = pd.read_csv(x_path, sep=',', dtype={'id': int, 'comment':str}, index_col=0)
    print('====== Reading train_y.csv ======')
    y = pd.read_csv(y_path, sep=',', dtype={'id': int, 'label':int}, index_col=0)
    
    x_train = np.zeros(shape=(len(x), senSize, vecSize), dtype=float)
    for i_row, sen in enumerate(x['comment']):
        #for i_w, w in enumerate(list(jieba.cut(emoji.demojize(sen), cut_all=False))):
        for i_w, w in enumerate(list(jieba.cut(sen, cut_all=False))):
            if i_w >= senSize: break
            x_train[i_row, i_w, :] = wv[w]
    np.save('x_train.npy', x_train)
    y_train = y['label'].values
    np.save('y_train.npy', y_train)

    np.save('x_config.npy', x_train.shape)
    '''
    x = []
    y = []
    print('====== Reading train_x.csv ======')
    with open(x_path, newline='') as x_fp:
        count = 0
        for i in x_fp.readlines()[1:]:
            c = i.split(',', 1)
            c = list(jieba.cut(c[1]))
            x.append(c)
            count += 1

    with open(y_path, newline='') as x_fp:
        for i in R.readlines()[1:]:
            c = i.split(',')
            y.append(int(c[1]))

    x = np.array(x)
    y = np.array(y)

    w2v_model = Word2Vec.load('word2vec%d.model'%vecSize)

    for i in range(count):
        if(len(x[i]) > senSize):
            x[i] = x[i][:senSize]

    tmp = np.zeros((count, senSize, vecSize), dtype=float)
    for i in range(count):

        for j in range(len(x[i])):
            tmp[i, j, :] = w2v_model.wv[x[i][j]]

    x = tmp
    x_train = x
    y_train = y
    #del x, y


print('====== Extract Verification data.csv ======')
if shuffle:
    x_shape, y_shape = x_train.shape, y_train.shape
    c =  np.concatenate((x_train.reshape(len(x_train), -1), y_train.reshape(len(y_train),1)), axis=1)
    np.random.shuffle(c)
    x_train = (c[:, :-1]).reshape(x_shape)
    y_train = (c[:, -1]).reshape(y_shape)
num_total = x_train.shape[0]
num_val = num_total // 10
print('total: %d'%num_total)
if num_val > 0:
    x_val = x_train[-num_val:]
    y_val = y_train[-num_val:]

    x_train = x_train[:-num_val]
    y_train = y_train[:-num_val]
    val_set = Data.TensorDataset(
            torch.tensor(x_val).type(torch.FloatTensor), 
            torch.tensor(y_val).type(torch.LongTensor))
    val_loader = Data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

train_set = Data.TensorDataset(
        torch.tensor(x_train).type(torch.FloatTensor), 
        torch.tensor(y_train).type(torch.LongTensor))
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

print('numTrain: %d, numVal: %d'%(train_set.__len__(), val_set.__len__()))
print('====== Building NN Model ======')
# model
#try: model = torch.load('model.pkl')
model = Model.MyLSTM(vecSize)
#device = torch.device('cuda')
device = torch.device('cuda')
model.to(device)
model.train()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.BCELoss()

print('====== Start Training ======')
for epoch in range(EPOCH):
    train_loss, train_acc_list = [], []
    train_acc = 0
    #torch.cuda.empty_cache()
    for _, (batch_x, batch_y) in enumerate(train_loader):
        x_cuda = batch_x.to(device, dtype=torch.float)
        y_cuda = batch_y.to(device)

        output = model(x_cuda)
        #print(output.size())        
        loss = loss_func(output, y_cuda.type(torch.cuda.FloatTensor))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predict = (output >= 0.5).type(torch.cuda.LongTensor)#torch.max(output, 1)[1]
        #print(predict)
        cal = (y_cuda == predict).cpu().numpy()
        train_acc +=np.sum(cal)
        #acc = np.mean(cal)
        #train_acc_list.append(acc)
        train_loss.append(loss.item())
    #acc = np.mean(train_acc_list)
    train_acc /= train_set.__len__()
    val_acc = 0
    if num_val > 0:
        model.eval()
        for _, (batch_x, batch_y) in enumerate(val_loader):
            x_cuda = batch_x.to(device, dtype=torch.float)
            y_cuda = batch_y.to(device)
            output = model(x_cuda)
            predict = (output >= 0.5).type(torch.cuda.LongTensor)
            #predict = torch.max(output, 1)[1]
            val_acc += np.sum( (y_cuda == predict).cpu().numpy())
        val_acc /= val_set.__len__()
        model.train()
        
    print("Epoch: {}| Loss: {:.4f}| Acc: {:.4f}| Val Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), train_acc, val_acc))
    if epoch % 5 == 0:
        torch.save(model, 'model_%s.pkl'%str(epoch))
        
#torch.save(model.state_dict(), 'model_params.pkl') # parameters
torch.save(model, 'model.pkl')
