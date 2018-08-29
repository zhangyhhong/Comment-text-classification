# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:36:57 2018

@author: zyh
构造tfidf的文本表示，并用多种机器学习方法进行训练

"""

#************************机器学习文本分类************************
#划分数据集
import pandas as pd
import numpy as np
import re
import os 
import matplotlib.pylab as plt
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
import os
os.chdir('D:\\Desktop\\textClassification')

stopwords1 = open('./dict/stopwords.txt',encoding='utf-8').readlines()
stoplist = set(w.strip() for w in stopwords1)

#utf_8_sig以utf-8有BOM编码读入
data_pos = pd.read_csv('data_pos.csv',encoding='utf_8_sig')
data_neg = pd.read_csv('data_neg.csv',encoding='utf_8_sig')
data = data_pos.append(data_neg)
data = data.dropna(axis=0,how='any')  #删除na的行


def tfIdf(text_words):  ###输入经过空格分隔的文本list###注意输入格式
#    vec = CountVectorizer().fit_transform(text_words)
#    tfidf = TfidfTransformer().fit_transform(vec)
    vectorizer=CountVectorizer()   #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    vector = vectorizer.fit_transform(text_words)
    transformer=TfidfTransformer() #该类会统计每个词语的tf-idf权值 
    tfidf=transformer.fit_transform(vector)#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = list(vectorizer.get_feature_names())  #获取词袋模型中的所有词语
    #print(word)
    weight=tfidf.toarray()  #元素a[i][j]表示j词在i类文本中的tf-idf权重
    return word,weight


    
#对分词后的文档计算tfidf权重
def wordWeight(text):      
    word,weight = tfIdf(text) 
    
    weight1 = list(np.sum(weight,axis=0))
    word2weight = dict(zip(word,weight1))  #转化为词典，便于后续排序
    return word,word2weight  #返回所有分词和在所有文档中对应的权重的字典

X_train,X_test,y_train,y_test = train_test_split(data[['Comment','Words']],data['Score'],test_size = 0.2,stratify=data['Score'],random_state=123)
X_train['Words'].iloc[0]

#计算tfidf
train_word,train_weight = tfIdf(X_train['Words']) #词列表和权重矩阵
train_tfidf_weight = pd.DataFrame(train_weight,columns = train_word)
#tfidf的训练数据
train_weight1 = list(np.sum(train_weight,axis=0))
word2weight = dict(zip(train_word,train_weight1))  #转化为词典，便于后续排序
#根据权重选择前K个特征值
K = 100   
feature_weight = sorted(word2weight.items(), key = lambda x:x[1], reverse = True)[:K]
feature,imp_weight = list(zip(*feature_weight))  #获得较为重要的词及对应的权重和

#导出重要的特征
pd.DataFrame(list(feature)).to_csv(r'D:\Desktop\textClassification\word2vec\important_features.csv',index=False,encoding='utf-8')

#在数据集中取出该词所在的列
weight_data = pd.DataFrame(train_weight,columns=train_word)
x_train = weight_data[list(feature)] ###最终训练集的x

#测试集的tfidf
idf = np.log(len(x_train)/(1+np.sum(train>0,axis = 0)))  #总文档数比有该词的文档数
#idf = dict(zip(train_word,idf))

#计算词频
import collections
def cutDoc(data):
    text = []
    for i in data.iloc[:,0]:
        text.append(cutSentence(i))
    data['Words'] = text
    return data

#对每一行进行分词
def cutSentence(sents): #输入一个句子
    words = list(jieba.cut(re.sub(' ','',sents)))
    words = [w for w in words if w not in stoplist]
    return words
X_test = cutDoc(X_test) 

#计算test中每个词的词频
tf = []
for i in X_test['Words']:
    a = [sum([f==y for y in i])/len(i) for f in feature]
    tf.append(a)
x_test = np.array(tf)*list(idf)
y_train[y_train<3]=-1;y_train[y_train>3]=1
y_test[y_test<3]=-1;y_test[y_test>3]=1


#***************************各种分类方法未调参******************
#x_train,y_train,x_test,y_test
#knn分类neigh=[20,50,100]，距离
from sklearn import neighbors
import time
#p=1,2,汉明距离和欧式距离，-1 the number of CPU cores
knn = neighbors.KNeighborsClassifier(n_neighbors=50,n_jobs=-1,metric='minkowski',p=2) 
t0 = time.time()
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
t1 = time.time();t=t1-t0#t = round(t1-t0,2)
acc = (pred==y_test).mean();print('KNN准确率为：',acc);print('消耗时间(s)：%s'% t)

#随机森林 ['gini','entropy']
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=20,
                             max_features='sqrt',n_jobs=-1)
t0 = time.time()
clf.fit(x_train,y_train)
t1 = time.time();t = t1-t0
pred = clf.predict(x_test)
acc = (pred==y_test).mean();print('随机森林准确率为：',acc);print('消耗时间(s)：%s'% t)

#SVM
from sklearn.metrics import confusion_matrix
from sklearn import svm
clf = svm.SVC(C=20,kernel='rbf',gamma=0.01)
t0 = time.time()
clf.fit(x_train,y_train)
t1 = time.time();t = round(t1-t0,2)
pred = clf.predict(x_test)
acc = (pred==y_test).mean();print('SVM准确率为：',acc);print('消耗时间(s)：%s'% t)
confusion_matrix(pred,y_test)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=10,random_state=123)
t0 = time.time()
clf.fit(x_train,y_train)
t1 = time.time();t = round(t1-t0,2)
pred = clf.predict(x_test)
acc = (pred==y_test).mean();print('Adaboost准确率为：',acc);print('消耗时间(s)：%s'% t)



