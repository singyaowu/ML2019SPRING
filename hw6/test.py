#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader
import jieba
import emoji
from gensim.models import Word2Vec, KeyedVectors

import Model

BATCH_SIZE = 128

#bash hw6_test.sh <test_x file> <dict.txt.big file> <output file>
try:
    test_x_path = sys.argv[1] # 'test_x.csv'
    dict_txt_path = sys.argv[2] #'dict.txt.big'
    output_path = sys.argv[3] #'submission.csv'
except:
    print('use default')
    test_x_path = 'test_x.csv'
    dict_txt_path = 'dict.txt.big'
    output_path = 'submission.csv'

vecSize = 150
senSize = 128
shaffle = False
model_name = 'model_tmp.pkl'
jieba.load_userdict(dict_txt_path)
w2v = Word2Vec.load("word2vec%d.model"%vecSize, mmap='r')

'''
print('====== Reading test_x.csv ======')
x = pd.read_csv(test_x_path, sep=',', dtype={'id': int, 'comment':str}, index_col=0)

x_test = np.zeros(shape=(len(x), senSize, vecSize), dtype=float)
for i_row, sen in enumerate(x['comment']):
    #for i_w, w in enumerate(list(jieba.cut(emoji.demojize(sen), cut_all=False))):
    for i_w, w in enumerate(list(jieba.cut(sen, cut_all=False))):
        if i_w >= senSize: break
        x_test[i_row, i_w, :] = wv[w]
del x
'''
x = []
print('====== Reading test_x.csv ======')
with open(test_x_path, newline='') as x_fp:
    for line in x_fp.readlines()[1:]:
        tmp = list(jieba.cut(line.split(',', 1)[1]))
        
        x.append([w for w in tmp if w in w2v.wv.vocab])
x = np.array(x)

x_test = np.zeros((x.shape[0], senSize, vecSize), dtype=float)
for i in range(x.shape[0]):
    if(len(x[i]) > senSize):
        x[i] = x[i][:senSize]
    for j in range(len(x[i])):
        x_test[i, j, :] = w2v.wv[x[i][j]]
del x
    
num_test = x_test.shape[0] 
test_set = Data.TensorDataset(
        torch.tensor(x_test).type(torch.FloatTensor) )
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = torch.load(model_name)
device = torch.device('cuda')
model.to(device)
model.eval()

predict_y = None
for step, (batch_x) in enumerate(test_loader):
    x_cuda = batch_x[0].to(device, dtype=torch.float)
    output = model(x_cuda)
    predict = (output >= 0.5).type(torch.cuda.LongTensor)
    #predict = torch.max(output, 1)[1]
    if predict_y is None:
        predict_y = predict
    else:
        predict_y = torch.cat((predict_y, predict), 0)

print(predict_y.size())

with open(output_path, 'w') as output_file: 
    output_file.write('id,label\n')
    for i in range(num_test):
        output_file.write( str(i) + ',' + str(int(predict_y[i])) + '\n')
print('finish')

