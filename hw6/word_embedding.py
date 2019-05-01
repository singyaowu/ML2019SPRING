#!/usr/bin/env python3
# coding: utf-8
from gensim.models import Word2Vec
import jieba
import emoji
import pandas as pd
import numpy as np

train_path = 'train_x.csv'
test_path = 'test_x.csv'
y_path = 'train_y.csv'
dict_txt_path = 'dict.txt.big'
x_type = {'id': int, 'comment':str}
vecSize = 150
jieba.load_userdict(dict_txt_path)
'''
train_df = pd.read_csv(train_path, sep=',', dtype=x_type, index_col=0)
test_df = pd.read_csv(test_path, sep=',', dtype=x_type, index_col=0)
sentence_df = pd.concat([train_df, test_df])

words = []
for sentence in sentence_df['comment']:
    #words.append(list(jieba.cut(emoji.demojize(sentence), cut_all=False)))
    words.append(list(jieba.cut(sentence, cut_all=False)))

model = Word2Vec(words, size=vecSize, min_count=1, iter=10)
model.save('word2vec%d.model'%vecSize)

print(model)

'''
x = []

with open(train_path, newline='') as R:
    R.readline()
    for i in R.readlines():
        c = i.split(',', 1)
        c = list(jieba.cut(c[1]))
        x.append(c)

with open(test_path, newline='') as R:
    R.readline()
    for i in R.readlines():
        c = i.split(',', 1)
        c = list(jieba.cut(c[1]))
        x.append(c)

model = Word2Vec(x, size=vecSize, min_count=1, iter=10)
model.save('word2vec%d.model'%vecSize)