# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:00:16 2018

@author: zyh
对全部数据进行分词，对积极消极文档分别进行tfidf计算并画出权重靠前的词分布
（对全部数据计算tfidf，并用于选取部分重要特征）
tfidf的计算
python中画图显示中文设置
词云的使用
pickle模块的使用
划分数据集
"""

import pandas as pd
import numpy as np
import os 
import matplotlib.pylab as plt
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
os.chdir(r'D:\ProgramExample\textClassification_zyh\finalData')
    
#画图显示中文
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  


def tfIdf(text_words):  #输入经过空格分隔的文本list
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
    return word,word2weight  #返回特征的list和所有分词和在所有文档中对应的权重的字典


#画权重为前10名的词的权重 
def word_count_plot(word_dict, title, index = 10):
    r = sorted(word_dict.items(), key = lambda x:x[1], reverse = True)[:index][::-1]
    label = list(zip(*r))[0]
    count_label = list(zip(*r))[1]
    num_label = np.arange(len(label))
    plt.figure(figsize = (10, 6))
    plt.barh(num_label, count_label,color = 'lightskyblue')
    plt.yticks(num_label, label)
    plt.xlabel('Sum of Tf-idf')
    plt.ylabel('Word')
    plt.title(title)
    for i in range(index):
        plt.text(r[i][1]+2, num_label[i], '%.0f' % r[i][1], fontsize = 9)
    plt.show()

#绘制不同评论的词的tf-idf的和
#停用词表
stoplist = open(r'stopwords.txt',encoding='utf-8').readlines()
stoplist = [w.strip() for w in stoplist]

data = pd.read_csv('data_jd.csv')
data.head()
data = data.drop(data.columns[0],axis=1) #删除第一列

#得到经过分割和去停用词后的词，
text = []
for i in data.iloc[:,0]:
    seg_list = jieba.cut(i,cut_all=False)
    seg_list = [w for w in seg_list if w not in stoplist]
    text.append(' '.join(seg_list))
data['Words'] = text   #################可用于输入计算tfidf
    
data_pos = data[data['Score']>3]
pos_word,pos_weight = wordWeight(data_pos['Words'])  
word_count_plot(pos_weight, "Positive Comment Words' Tf-idf", index = 20)

data_neg = data[data['Score']<3]
neg_word,neg_weight = wordWeight(data_neg['Words'])
word_count_plot(neg_weight, "Negative Comment Words' Tf-idf", index = 20)

data_unk = data[data['Score']==3]
unk_word,unk_weight = wordWeight(data_unk['Words'])
word_count_plot(unk_weight, "Unknown Comment Words' Tf-idf", index = 20)


data_pos.to_csv('data_pos.csv',na_rep='',index=False,encoding='utf-8')
data_neg.to_csv('data_neg.csv',na_rep='',index=False,encoding='utf-8')
data_unk.to_csv('data_unk.csv',na_rep='',index=False,encoding='utf-8')

#所有评论中选取前五十的重要词
all_word,all_weight = wordWeight(data['Words'])  
imp_all_word_weight = sorted(all_weight.items(), key = lambda x:x[1], reverse = True)[:50]
imp_all_word,imp_all_weight = list(zip(*imp_all_word_weight))
pd.DataFrame(imp_all_word).to_csv('all_tfidf_top50')  #重要特征写出来

#绘制词云
from wordcloud import WordCloud,ImageColorGenerator  #安装包需要先安装VC++
import PIL.Image as Image

#按照最终权重进行排序，tfidf求和    
imp_pos_word_weight = sorted(pos_weight.items(), key = lambda x:x[1], reverse = True)[:50]
imp_pos_word,imp_pos_weight = list(zip(*imp_pos_word_weight))
imp_pos_word = ' '.join(imp_pos_word)

imp_neg_word_weight = sorted(neg_weight.items(), key = lambda x:x[1], reverse = True)[:50]
imp_neg_word,imp_neg_weight = list(zip(*imp_neg_word_weight))
imp_neg_word = ' '.join(imp_neg_word)

imp_unk_word_weight = sorted(unk_weight.items(), key = lambda x:x[1], reverse = True)[:50]
imp_unk_word,imp_punk_weight = list(zip(*imp_unk_word_weight))
imp_unk_word = ' '.join(imp_unk_word)

#画图
def wordsClound(words):
    coloring = np.array(Image.open('huaweipho.jpg'))
    my_wordcloud = WordCloud(background_color="white", max_words=2000,
                             mask=coloring, max_font_size=60, 
                             random_state=42, scale=5,font_path='milanti.ttf')
    my_wordcloud.generate(words)
    image_colors = ImageColorGenerator(coloring) #从图片中取颜色
    plt.imshow(my_wordcloud.recolor(color_func=image_colors))
    plt.axis("off")  #去除坐标
    #plt.savefig('pink.png', dpi=300) #像素值越大越清晰
    plt.show()
    
    
wordsClound(imp_pos_word)
wordsClound(imp_neg_word)
wordsClound(imp_unk_word)

#划分数据集进行训练
data = data[data['Score']!=3]

import pickle
with open(r'D:\Desktop\textClassification\finalData\index.txt', 'rb') as output:
    trainIndex = pickle.load(output)
    testIndex = pickle.load(output) 
    valIndex = pickle.load(output)
allTrainIndex = trainIndex + valIndex
train = data.iloc[allTrainIndex,:] 
train = train.dropna(axis=0,how='any')  #删除na的行
test = data.iloc[testIndex,:]
test = test.dropna(axis=0,how='any')  #删除na的行


#train,test = train_test_split(data[['Words','Score']],test_size = 0.2,stratify=data['Score'],random_state=123)
#train.head()    
#train.to_csv('no_stopword_train.csv',na_rep='',index=False,encoding='utf-8')
#test.to_csv('no_stopword_test.csv',na_rep='',index=False,encoding='utf-8')

#导入验证集和测试集
train = data.iloc[trainIndex,:] 
valid = data.iloc[valIndex,:] 
test = data.iloc[testIndex,:] 

train.to_csv('no_stopword_train1.csv',na_rep='',index=False,encoding='utf-8')
valid.to_csv('no_stopword_valid1.csv',na_rep='',index=False,encoding='utf-8')
test.to_csv('no_stopword_test.csv',na_rep='',index=False,encoding='utf-8')

#******************挑选变量*****************
data = pd.read_csv('no_stopword_train1.csv')  #训练集原始数据
#计算词频
#得到经过分割和去停用词后的词，
text = [];all_word = set()
for i in data.iloc[:,0]:
    seg_list = jieba.cut(i,cut_all=False)
    seg_list = [w for w in seg_list if w not in stoplist]
    all_word = all_word | set(seg_list)  #得到所有的词
    text.append(' '.join(seg_list))
data['Words'] = text   

    
#用tfidf挑选变量
tfidfK = 50
all_word,all_weight = wordWeight(data['Words'])  
imp_all_word_weight = sorted(all_weight.items(), key = lambda x:x[1], reverse = True)[:tfidfK]
featureTI,imp_all_weight = list(zip(*imp_all_word_weight))
featureTI=set(featureTI)

#用互信息挑选变量，输入[Npos,Nneg,NposW,NnegW]，积极消极文档数，以及对应的文档中有该词的数
huK = 50
def getHu(Npos,Nneg,NposW,NnegW):
    N = Npos+Nneg
    W = NposW+NnegW
    res = Npos/N*np.log((NposW+1)*N/(W*Npos))+Nneg/N*np.log((NnegW+1)/(W*Nneg))
    return res
word,tfidfArray = tfIdf(text)
data['word'] = text
posTfArr = tfidfArray[data['Score']>3,:]
negTfArr = tfidfArray[data['Score']<3,:]
Npos=posTfArr.shape[0];Nneg=negTfArr.shape[0]
NposW = np.sum(posTfArr>0,axis=0);NnegW = np.sum(negTfArr>0,axis=0)
huLis = []
for i in range(len(word)):
    huLis.append(getHu(Npos,Nneg,NposW[i],NnegW[i]))
wordIndex = np.argsort(huLis)[::-1][:huK]
featureHu = set([word[i] for i in wordIndex])

#用卡方统计量挑选变量
chiK = 50
def getChi(Npos,Nneg,NposW,NnegW):
    N = Npos+Nneg
    res = N*(NposW*(Nneg-NnegW)-NnegW*(Npos-NposW))**2/(Npos*(NposW+NnegW)*Nneg*(N-NposW-NnegW))
    return res
chiLis = []
for i in range(len(word)):
    chiLis.append(getChi(Npos,Nneg,NposW[i],NnegW[i]))
wordIndexChi = np.argsort(chiLis)[::-1][:chiK]
featureChi = set([word[i] for i in wordIndexChi])

#取并集得到最终的特征
features = featureTI | featureHu | featureChi



###****************以下是草稿***************
a = np.arange(10).reshape(2,5)
b = np.arange(10).reshape(2,5)

import pickle
data1 = {'a': [1, 2.0, 3, 4+6j],
         'b': ('string', u'Unicode string'),
         'c': None}

selfref_list = [1, 2, 3]
selfref_list.append(selfref_list)


# Pickle dictionary using protocol 0.
with open('D:/data.pkl', 'wb') as output:
    pickle.dump(data1, output)
    pickle.dump(data2, output)
# Pickle the list using the highest protocol available.
pickle.dump(selfref_list, output, -1)



#使用pickle模块从文件中重构python对象

import pprint, pickle

pkl_file = open('D:/data.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

data2 = pickle.load(pkl_file)
pprint.pprint(data2)

pkl_file.close()

#*************制作PPT需要****************
a = open('D:/Desktop/11.txt').readlines()
' '.join(a).replace('\n','')

a = '这部手机真不错，屏幕大，电池耐用。'
'/'.join(jieba.cut(a,cut_all =False))  #精确模式
'/'.join(jieba.cut(a,cut_all =True))  #全模式
'/'.join(jieba.cut_for_search(a))  #搜索引擎模式

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import jieba
import pandas as pd
corpus = pd.DataFrame(['这部手机真不错，屏幕不错，电池不错。',
          '这部手机不好看，屏幕刺眼。'])
    
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
    print(u"-------这里输出第",i,u"类文本的词语tf-idf权重------"  )
    for j in range(len(word)):  
        print(word[j],weight[i][j] )


    
