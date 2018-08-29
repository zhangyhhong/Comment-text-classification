# Comment-text-classification

手机评论文本分类项目，自己独立完成的尝试项目

* [code0_crawler](code0_crawler.py)爬取评论文本

* [code1_dataAnalysis](code1_dataAnalysis.py)对全部数据进行分析，对积极消极文档分别进行tf-idf计算并画出权重靠前的词的分布，用于选取部分重要特征

* [code2_classify](code2_classify.py)构造tf-idf的文本表示，并用多种机器学习方法进行训练

* [code3_word2vec](code3_word2vec.py)构建和导入word2vec模型，向量相加取平均，向量加权取平均，`构造句向量`，运用机器学习方法和对应调参

* [code4_LSTM](code4_LSTM.py)用LSTM训练