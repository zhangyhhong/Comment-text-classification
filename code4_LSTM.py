# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:13:43 2018

@author: zyh
"""

from keras.layers.core import Activation,Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


import os
os.chdir(r'D:\Desktop\textClassification\finalData')

train = pd.read_csv('no_stopword_train.csv')
test = pd.read_csv('no_stopword_test.csv') 
train['Score'][train['Score']<3]=0;train['Score'][train['Score']>3]=1
test['Score'][test['Score']<3]=0;test['Score'][test['Score']>3]=1

#转化为list
train['Words'] = train.Words.apply(lambda x:str(x).split())
test['Words'] = test.Words.apply(lambda x:str(x).split())

#train = train.append(test, ignore_index=True)
#***********************构建LSTM*****************
#统计句子最大的词数
import collections
numSent = len(train)  #训练样本总数
wordFreqs = collections.Counter()  #词频
sentenceLength = []
for sen in train['Words']:
    senLen = len(sen)
    sentenceLength.append(senLen)
    for word in sen:
        wordFreqs[word] += 1
print('句子最大词数：',max(sentenceLength),'不同的词个数：',len(wordFreqs)) #310 16558       

#画箱线图展示句子长度分布
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(sentenceLength)
plt.show()
#词数分布
plt.hist(sentenceLength, 60)
plt.xlabel('Words count')
plt.ylabel('Frequency')
plt.title('Words count of text')
plt.show()

MAX_FEATURES = 10000  #<=========用于训练的特征个数,其余的标记为UNK
MAX_SENTENCE_LENGTH = 150  #<=========每个句子的长度，长的截断，不够的补上符号‘PAD’

#将词语转化为数字表示，以及将数字转化回特征
vocab_size = min(MAX_FEATURES, len(wordFreqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(wordFreqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}


#对数据进行处理
from keras.preprocessing import sequence
def x2seq(sentenceWords):
    X = []
    i = 0
    for sen in sentenceWords:
        for word in sen:
            seqs = []
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X.append(seqs)
        i += 1
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    return X
 
X = x2seq(train['Words'])
y = train['Score']
Xtest = x2seq(test['Words'])
ytest = test['Score']
#转换为哑变量
y = pd.get_dummies(y).values;ytest=pd.get_dummies(ytest).values

Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(128, activation='relu'))  #增加隐藏层
model.add(Dense(2,activation='softmax'))
#model.add(Dense(1),Activation("sigmoid")) #改用softmax试试
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
#binary_crossentropy

BATCH_SIZE = 32
NUM_EPOCHS = 20
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xvalid, yvalid))

#单个输出sigmoid
pred = model.predict(Xtest)
pred[pred>0.5]=1
pred[pred<=0.5]=0
np.mean(pred == ytest.reshape(-1,1))  #0.8252

#两个输出
score = model.evaluate(Xtest,ytest,batch_size=20,verbose=1)                        
print('准确率为：',score[1])  #0.250


#模型可视化
from keras.utils import plot_model
plot_model(model, to_file='model.png')

from keras.utils.visualize_util import plot
plot(model, to_file='model1.png',show_shapes=True)  



