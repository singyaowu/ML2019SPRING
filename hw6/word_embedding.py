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

newLine = False
sentences= []
if newLine:
    train_df = pd.read_csv(train_path, sep=',', dtype=x_type)
    test_df = pd.read_csv(test_path, sep=',', dtype=x_type)
    sentence_df = pd.concat([train_df, test_df])

    for sentence in sentence_df['comment']:
        #sentences.append(list(jieba.cut(emoji.demojize(sentence), cut_all=False)))
        sentences.append(list(jieba.cut(sentence)))
else:
    with open(train_path, newline='') as x_fp:
        for line in x_fp.readlines()[1:]:
            #c = list(jieba.cut( emoji.demojize(c[1]) ) )
            words = list(jieba.cut(line.split(',', 1)[1]))
            sentences.append(words)

    with open(test_path, newline='') as x_fp:
        for line in x_fp.readlines()[1:]:
            #c = list(jieba.cut( emoji.demojize(c[1]) ) )
            words = list(jieba.cut(line.split(',', 1)[1]))
            sentences.append(words)

print(sentences[0:3])
model = Word2Vec(sentences, size=vecSize, min_count=5, iter=10, sg=1)
model.save('word2vec%d.model'%vecSize)
print(model)