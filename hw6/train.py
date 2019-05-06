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
BATCH_SIZE = 64
LEARNING_RATE = 0.0003

#bash hw6_test.sh <test_x file> <dict.txt.big file> <output file>
#bash hw6_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
x_path = 'train_x.csv'
y_path = 'train_y.csv'
dict_txt_path = 'dict.txt.big'
vecSize = 150
senSize = 128
shuffle = False
newLine = False
jieba.load_userdict(dict_txt_path)
w2v = Word2Vec.load("word2vec%d.model"%vecSize, mmap='r')

print('====== Extracting train_x.csv ======')
x = []
if newLine:
    x_df = pd.read_csv(x_path, sep=',', dtype={'id': int, 'comment':str})
    for sen in x_df['comment']:
        tmp = list(jieba.cut(sen))
        #x.append(tmp)
        x.append([w for w in tmp if w in w2v.wv.vocab])
    del x_df
    y_df = pd.read_csv(y_path, sep=',', dtype={'id': int, 'label':int})
    y_train = y_df['label'].values
else:
    with open(x_path, newline='') as x_fp:
        for line in x_fp.readlines()[1:]:
            tmp = list(jieba.cut(line.split(',', 1)[1]))
            #x.append(tmp)
            x.append([w for w in tmp if w in w2v.wv.vocab])
    with open(y_path, newline='') as y_fp:
        y_train = [int(line.split(',')[1]) for line in y_fp.readlines()[1:] ]
        y_train = np.array(y_train)
x = np.array(x)


x_train = np.zeros((x.shape[0], senSize, vecSize), dtype=float)
for i in range(x.shape[0]):
    if(len(x[i]) > senSize):
        x[i] = x[i][:senSize]
    for j in range(len(x[i])):    
        x_train[i, j, :] = w2v.wv[x[i][j]]
del x
    #np.save('x_train.npy', x_train)
    #np.save('y_train.npy', y_train)
    #np.save('x_config.npy', x_train.shape)
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
    #x_train = x_train[:-num_val]
    #y_train = y_train[:-num_val]
    val_set = Data.TensorDataset(
            torch.tensor(x_train[-num_val:]).type(torch.FloatTensor), 
            torch.tensor(y_train[-num_val:]).type(torch.LongTensor))
    val_loader = Data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    train_set = Data.TensorDataset(
            torch.tensor(x_train[:-num_val]).type(torch.FloatTensor), 
            torch.tensor(y_train[:-num_val]).type(torch.LongTensor))
    train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
else:
    train_set = Data.TensorDataset(
            torch.tensor(x_train).type(torch.FloatTensor), 
            torch.tensor(y_train).type(torch.LongTensor))
    train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
del x_train, y_train
print('numTrain: %d, numVal: %d'%(train_set.__len__(), val_set.__len__()))

print('====== Building NN Model ======')
model = Model.MyLSTM(vecSize)
device = torch.device('cuda')
model.to(device)
model.train()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.BCELoss()

print('====== Start Training ======')
acc_val_save = 0.75
for epoch in range(EPOCH):
    train_loss, train_acc_list = [], []
    train_acc = 0
    torch.cuda.empty_cache()
    for _, (batch_x, batch_y) in enumerate(train_loader):
        x_cuda = batch_x.to(device, dtype=torch.float)
        y_cuda = batch_y.to(device)

        output = model(x_cuda)        
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
    if val_acc > acc_val_save:
        acc_val_save = val_acc
        #torch.save(model.state_dict(), 'model_%s_params.pkl'%str(epoch))
        torch.save(model, 'model_tmp.pkl')
        
#torch.save(model.state_dict(), 'model_params.pkl') # parameters
torch.save(model, 'model_finish.pkl')
