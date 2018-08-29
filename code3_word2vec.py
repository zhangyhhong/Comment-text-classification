# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 15:24:19 2018

@author: zyh
构建word2vec模型

"""
import os
import pandas as pd
import numpy as np
import jieba
import logging
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from IPython.display import display
import time

os.chdir(r'D:\Desktop\textClassification\finalData')


#1.1自己生成word2vec模型
def createW2VModel():
    data = pd.read_csv('data_jd.csv')
    data.head()
    
    #生成以空格分割的txt文件
    with open('fenci_result.txt','w',encoding='utf-8') as f:
        for i in data['Comment']:
            seg_list = jieba.cut(i,cut_all=False)
            f.write(' '.join(seg_list))
        
    #构建模型 
    W = 50  #<========词向量维度
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)  
    sentences =word2vec.Text8Corpus(u"fenci_result.txt")  # 加载语料  
    #size维度，少于min_count的词忽略，window为距离目标词距离不超过
    model =word2vec.Word2Vec(sentences, size=W,window=5,min_count=5,workers=4)  #训练skip-gram模型，默认window=5  
    return model
    
#直接导入自己生成的模型
W = 50 
model = word2vec.Word2Vec.load('word2vec.model') #<=====直接导入已经保存的词向量模型


#1.2 导入网上下载的word2vec模型
import gensim
W = 256  #<========词向量维度
model = gensim.models.Word2Vec.load(r'E:\BaiduYunDownload\word2vec_from_weixin\word2vec\word2vec_wx')
#不更新model时，可以只保留需要的词向量
#WV = model.wv
#del model


#导入数据
#utf_8_sig以utf-8有BOM编码读入
train = pd.read_csv('no_stopword_train.csv')
test = pd.read_csv('no_stopword_test.csv') 
train['Score'][train['Score']<3]=0;train['Score'][train['Score']>3]=1
test['Score'][test['Score']<3]=0;test['Score'][test['Score']>3]=1

#转化为list
train['Words'] = train.Words.apply(lambda x:str(x).split())
test['Words'] = test.Words.apply(lambda x:str(x).split())


#2.1.对每个文档用Word2vec进行词向量相加再平均 
def getDocVec(data):  #输入分词后的list文档，返回文本文档向量矩阵
    vec = np.array([0.0]*W)
    x_train = []
    for i in data:
        t = 1 #防止分母为0
        for j in range(len(i)):
            try:  #词向量若不存在则忽略该词
                vec += model.wv[i[j]]
                t += 1
            except KeyError:
                continue
        x_train.append(vec/t)
    return x_train


#转化为向量
x_train = np.array(getDocVec(train['Words']))  
x_test = np.array(getDocVec(test['Words']))
y_train = train['Score'];y_test = test['Score']



#2.2挑选部分重要的词的词向量相加再平均
#特征选择过程
def word_feature(word_dict, index = 10):
    r = sorted(word_dict.items(), key = lambda x:x[1], reverse = True)[:index]
    res = [x[0] for x in r]
    return res

import itertools
words = set(itertools.chain.from_iterable(list(train['Words'])))

tfidf_word = list(words)

import pickle
feature_dict_list = []
with open('./xiaoji/dict.txt', 'rb') as output:
    for i in range(10):
        feature_dict_list.append(pickle.load(output))
        
        
n_feature = [3,5,10,20,30,50]  #<====特征个数
feature_all = []
for n_fea in n_feature:        
    feature_list = []
    for i in feature_dict_list:
        feature = word_feature(i, n_fea)
        feature_list.extend(feature)
        #得到选择出的特征        
    #feature_all = list(set(feature_list) & set(tfidf_word))
    feature_all.append(list(set(feature_list) & set(tfidf_word)))


#导入选择好的特征
import pickle
with open('feat.txt', 'rb') as output:
    feature_svm = pickle.load(output)
    feature_knn = pickle.load(output) 
    feature_nb = pickle.load(output)
    
def getDocVecImpFeat(data,features):  #输入分词后的list文档，返回文本文档向量矩阵
    vec = np.array([0.0]*W)
    x_train = []
    for i in data:
        t = 1 #防止分母为0
        for j in range(len(i)):
            if i[j] in features:  #若是重要特征才相加
                try:  #词向量若不存在则忽略该词
                    vec += model.wv[i[j]]
                    t += 1
                except KeyError:
                    continue
        x_train.append(vec/t)
    return x_train

#给定不同的特征向量相加
x_train = np.array(getDocVecImpFeat(train['Words'],features))
x_test = np.array(getDocVecImpFeat(test['Words'],features))
y_train = train['Score'];y_test = test['Score']


#2.3 构建句向量
#选择词典和选择出的一部分词进行取交集
#2.3.1选择最常见的词
MAX_FEATURES = 10000
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

mostCommWords = wordFreqs.most_common(MAX_FEATURES)
word,wordNum = zip(*mostCommWords)

all_feature = list(set(model.wv.vocab) & set(word))
choseNum = len(all_feature)  #7690  最终选定的维度

#2.3.2特征选择方法
n_fea = 2000  #<=========不同的标准的特征选择个数
feature_list = []
for i in feature_dict_list:
    feature = word_feature(i, n_fea)
    feature_list.extend(feature)
    #得到选择出的特征        
feature_all = list(set(feature_list) & set(tfidf_word))

all_feature = list(set(model.wv.vocab) & set(feature_all))
choseNum = len(all_feature)  #1000->2427  2000->最终选定的维度

def getDocVec(train):
    t0 = time.time()
    trainArray = np.zeros([len(train),choseNum])
    for i in range(trainArray.shape[0]):
        if i%50==0: print('已经训练至第%s个样本'%i)
        for word in train['Words'][i]:  #对每条评论中的每个词
            if word in all_feature:
                trainArray[i][all_feature.index(word)]=1  #对应的词值为1
                simiWordValue = model.wv.most_similar(word,topn=10)
                simiword,simiValue = zip(*simiWordValue)
                simiWord = list(simiword);simiValue=list(simiValue)             #simiValue=[round(x,4) for x in simiValue]
                word2value = dict(simiWordValue)
            wordExist = [x for x in simiWord if x in all_feature]  #存在的词
            indexFeat = [all_feature.index(x) for x in wordExist]  #找到对应的需要修改的索引，即列
            for t in range(len(indexFeat)):
                j = indexFeat[t]
                value = word2value[wordExist[t]]
                if trainArray[i][j]<value:
                    trainArray[i][j]=value
    t1 = time.time();print('消耗用时：%s sec'% (t1-t0))  #6183s
    return trainArray


trainArray = getDocVec(train)  
    
testArray = getDocVec(test)             
with open('word2vectrainArray.pkl', 'wb') as output:
    pickle.dump(trainArray, output)        
with open('word2vectestArray.pkl', 'wb') as output:
    pickle.dump(testArray, output)    

with open('word2vectrainArray.pkl', 'rb') as inpu:
    trainArray = pickle.load(inpu)        
with open('word2vectestArray.pkl', 'rb') as inpu:
    testArray = pickle.load(inpu) 



#3.各种机器学习方法
#对每一列进行归一化
from sklearn import preprocessing  
x_train = preprocessing.MinMaxScaler().fit_transform(x_train) 
x_test = preprocessing.MinMaxScaler().fit_transform(x_test)

paraSpace = []
# KNN
from sklearn.neighbors import KNeighborsClassifier
clf_KNN = KNeighborsClassifier(p=2,n_jobs=-1)
space = dict(n_neighbors=[3,5,7,9,10,15,20,30],weights=['uniform','distance'])
#paraSpace.append(space)


# GaussianNB
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()


# SVM
from sklearn.svm import SVC
clf_svm = SVC(random_state=0,kernel='rbf',degree=2)
space = dict(C=[0.1,0.5,1,2,5,7,10],kernel =['linear','poly','rbf','sigmoid'],
             degree=[2,3,4],gamma=[0.001,0.01,0.1],coef0=[-1,0,1])

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf_Ada = AdaBoostClassifier(random_state=0)
#clf_Ada.fit(x_train2,y_train)
#np.mean(clf_Ada.predict(x_test2)==y_test)

# 决策树
from sklearn.tree import DecisionTreeClassifier
clf_Tree = DecisionTreeClassifier(random_state=0)
space = dict(criterion=['gini','entropy'],)
paraSpace.append(space)

# Logistic
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(random_state=0)
space = dict(penalty=['l1','l2'])
paraSpace.append(space)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(random_state=0)
space = dict(n_estimators=[5,10,20])
paraSpace.append(space)

# GBDT
from sklearn.ensemble import GradientBoostingClassifier
clf_gbdt = GradientBoostingClassifier(random_state=0)

mL=[clf_KNN,clf_NB,clf_svm,clf_Ada,clf_Tree,clf_log,clf_forest,clf_gbdt]             
mL=[clf_KNN,clf_NB,clf_forest]      
mL=[clf_Tree,clf_log,clf_forest]       


#对各种方法用不同的特征，对不同的方法选择特征
accAll = []
for features in feature_all:
    x_train = np.array(getDocVecImpFeat(train['Words'],features))
    x_test = np.array(getDocVecImpFeat(test['Words'],features))
    y_train = train['Score'];y_test = test['Score']
    x_train = preprocessing.MinMaxScaler().fit_transform(x_train) 
    x_test = preprocessing.MinMaxScaler().fit_transform(x_test)
    acc = []
    for method in mL:
        #method = mL[0]
        method.fit(x_train,y_train)
        a = np.mean(method.predict(x_test)==y_test)
        acc.append(a)
        print(method.__class__,'测试集准确率为：',a)
    accAll.append(acc)
    
#只对默认的参数进行运行，初步看结果    
acc = []    
for method in mL:
    #method = mL[0]
    method.fit(trainArray,train['Score'])
    a = np.mean(method.predict(testArray)==test['Score'])
    acc.append(a)
    print(method.__class__,'测试集准确率为：',a)
   

import pickle
with open('accAll.pkl','wb') as output:
    pickle.dump(accAll,output)


# 搜索空间
from sklearn.metrics import classification_report 
from sklearn import metrics 
import time
#scorer = make_scorer(accuracy_score)
#kfold = KFold(n_splits=5)

#对单个机器学习方法进行调参
t0 = time.time()
grid = GridSearchCV(clf_forest, n_estimators=[5,10,20], scoring='accuracy', cv=3,n_jobs=2)
grid = grid.fit(trainArray, train['Score'])
t1 = time.time()    
print('消耗的时间：',t1-t0)
print('最好的分数：',grid.best_score_)
print('最好的参数：',grid.best_params_)  
pred = grid.predict(testArray)
resultAcc = np.mean(pred == test['Score'])
print('测试集的结果：',resultAcc)



#同时对多个机器学习方法进行调参
def doGridSearch(mL,paraSpace):
    for i in range(len(mL)):
        t0 = time.time()
        grid = GridSearchCV(mL[i], paraSpace[i], scoring='accuracy', cv=3,n_jobs=2)
        grid = grid.fit(trainArray, train['Score'])
        t1 = time.time()    
        print('消耗的时间：',t1-t0)
        print('最好的分数：',grid.best_score_)
        print('最好的参数：',grid.best_params_)  
        pred = grid.predict(testArray)
        resultAcc = np.mean(pred == test['Score'])
        print('测试集的结果：',resultAcc)
doGridSearch(mL,paraSpace)

grid.grid_scores_
display(pd.DataFrame(grid.cv_results_).T)



clf_KNN = KNeighborsClassifier(n_neighbors=5,weights='distance',p=2,n_jobs=-1)
clf_KNN.fit(x_train,y_train)
pred1 = clf_KNN.predict(x_test)
np.mean(pred1 == y_test)

pred1 =clf_NB.predict(x_test1)




















#************以下为参考代码***************
#输出向量
a = model.wv['喜欢']
b = model.wv['讨厌']
c = model.wv['不错','手机']
a-b

try:  #有可能该词不存在词典中
    y1 = model.wv.similarity('不错','手机')    
except KeyError:
    y1=0
print('两个词相似度为：',y1)   

y2 = model.wv.most_similar('讨厌',topn=20);print('相似的词有',y2)

#寻找对应关系
y3 = model.wv.most_similar(['质量','不错'],['手机'],topn=3)
for i in y3:
    print(i[0],i[1])

#寻找不合群的词
y4 = model.wv.doesnt_match('手机 不错 漂亮'.split())
print('不合群的词：',y4)

model.score(['这个 手机 很 好看'.split()])

y4 = model.doesnt_match('手机 屏幕 漂亮'.split())
print('不合群的词：',y4)  #结果不令人满意

#存储模型和调用模型
model.save('word2vec.model')
model2 = word2vec.Word2Vec.load('word2vec.model')

#以c语言可以解析的形式存储词向量
model.save_word2vec_format('vec.model.bin',binary=True)
model3 = word2vec.Word2Vec.load_word2vec_format('vec.model.bin',binary=True)


import gensim
model = gensim.models.Word2Vec.load('word2vec_wx')

pd.value_counts(y)  #计算离散型的分布

#构建简单的词向量模型
sentences = [['first', 'sentence',], ['second', 'sentence'],['haha','sentence']]
model = word2vec.Word2Vec(sentences,size=5,window=5,min_count=1,workers=4)
a = set(model.wv.vocab)

#对字典的使用
a = {'a':1,'b':2}
b = {'a':1.5,'b':1.5}
value = 1
a['d'] = max(a.get('d',0),value)


#绘制ROC
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

Plot of a ROC curve for a specific class

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()












 



